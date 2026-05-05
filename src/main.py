"""
page-evict: LLaMA 3-8B with monkeypatched KV cache compression.

Replaces HuggingFace's DynamicCache with:
  - BufferManager: fixed-size physical K/V storage
  - MappingTable: virtual → physical index mapping
  - AIA: async importance accumulator (EMA-decayed attention scores)
  - EvictController: evicts low-importance tokens when buffer is 95% full

Usage:
    python src/main.py
"""

import sys
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, CacheLayerMixin

# Load .env from project root (one level up from src/)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
HF_TOKEN = os.getenv("HF_TOKEN")

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from buffer_manager import BufferManager
from table import MappingTable
from importance import AIA
from eviction import EvictController


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Physical buffer holds fewer slots than max context → compression
MAX_PHYSICAL_SLOTS = 4096
# Max virtual sequence length the mapping table can track
MAX_VIRTUAL_CONTEXT = 8192
# Recency window: last N tokens are protected from eviction
WINDOW_SIZE = 512
# EMA decay factor for AIA importance scoring
GAMMA = 0.99

DEVICE = "cuda"
DTYPE = torch.float16


# ═══════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════

def load_model(model_id: str = MODEL_ID):
    """
    Load LLaMA from HuggingFace with eager attention.

    We MUST use attn_implementation="eager" because:
    - Flash Attention / SDPA fuse softmax+matmul and do NOT return
      raw attention weights
    - We need those weights to feed into AIA for importance scoring
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not found. Set it in the .env file at the project root.")

    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

    print(f"Loading model from {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        device_map="auto",
        attn_implementation="eager",  # REQUIRED: need raw attention weights for AIA
        token=HF_TOKEN,
    )
    model.eval()

    print(f"Model loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_key_value_heads} KV heads, "
          f"head_dim={model.config.hidden_size // model.config.num_attention_heads}")

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════
# HuggingFace Cache shim
# ═══════════════════════════════════════════════════════════════
#
# Why this exists:
#   HF generate uses `past_key_values.get_seq_length()` for two things on
#   every decode step:
#     1. Computing `position_ids` (so RoPE encodes the new token at the
#        correct absolute position).
#     2. Deciding whether to slice `input_ids` down to just the new token
#        (`next_sequence_length = 1 if use_cache else None`).
#
#   If we set `use_cache=False`, HF re-passes the FULL growing sequence
#   on every decode step and our Python-level prefill loop reprocesses
#   every past token from scratch — catastrophically slow on long prompts.
#
#   If we set `use_cache=True` but pass no cache, HF auto-creates a
#   DynamicCache that our patched_forward never updates, so
#   get_seq_length() stays at 0 forever and the same bug returns.
#
#   So we provide a minimal Cache that owns NO K/V storage (the real
#   cache lives in BufferManager) and only ticks a per-layer counter.
#   patched_forward calls .update() once per call to advance it.

class PageEvictLayer(CacheLayerMixin):
    is_compileable = False
    is_sliding = False

    def __init__(self):
        super().__init__()
        self.seen_tokens = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.seen_tokens += key_states.shape[-2]
        return key_states, value_states

    def get_seq_length(self) -> int:
        return self.seen_tokens

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.seen_tokens + query_length, 0

    def reset(self) -> None:
        self.seen_tokens = 0


class PageEvictCache(Cache):
    def __init__(self):
        super().__init__(layer_class_to_replicate=PageEvictLayer)


# ═══════════════════════════════════════════════════════════════
# Per-Layer State
# ═══════════════════════════════════════════════════════════════

class LayerPageEvictState:
    """
    Holds the page-evict components for a single attention layer.

    Each of the 32 LLaMA layers gets its own independent:
      - BufferManager (physical K/V storage)
      - MappingTable (virtual→physical index mapping)
      - AIA (importance scores)
      - EvictController (eviction policy)

    This matches the README design decision:
      "Buffer per layer: Independent — Avoids cross-layer kernel complexity"
    """
    def __init__(self, layer_idx: int, num_kv_heads: int, head_dim: int,
                 device: str = DEVICE, dtype: torch.dtype = DTYPE):
        self.layer_idx = layer_idx
        self.buffer = BufferManager(MAX_PHYSICAL_SLOTS, num_kv_heads, head_dim, device, dtype)
        self.table = MappingTable(MAX_VIRTUAL_CONTEXT, device)
        self.aia = AIA(MAX_PHYSICAL_SLOTS, gamma=GAMMA)
        self.evictor = EvictController(MAX_PHYSICAL_SLOTS, window_size=WINDOW_SIZE)

    def reset(self):
        """Reset all state for a new sequence."""
        self.buffer = BufferManager(
            MAX_PHYSICAL_SLOTS,
            self.buffer.k_buffer.shape[1],
            self.buffer.k_buffer.shape[2],
            self.buffer.device,
            self.buffer.dtype,
        )
        self.table = MappingTable(MAX_VIRTUAL_CONTEXT, self.table.device)
        self.aia = AIA(MAX_PHYSICAL_SLOTS, gamma=GAMMA)


# ═══════════════════════════════════════════════════════════════
# Patched Forward
# ═══════════════════════════════════════════════════════════════

def _try_evict(state: LayerPageEvictState):
    """
    Check if the buffer is near capacity and evict if needed.
    Evicts down to 75% occupancy to avoid "yo-yo" thrashing.
    """
    if state.buffer.get_occupancy() < 0.95:
        return

    # Get importance scores from AIA (synchronizes the async stream)
    scores = state.aia.get_scores()

    # Get the active mapping table for scoring
    active_table = state.table.get_physical_indices()
    occupied = state.buffer.write_head - len(state.buffer.free_list)

    # Evict: returns (physical_indices_to_free, logical_indices_to_remove)
    phys_to_free, logical_to_remove = state.evictor.evict(
        scores, active_table, occupied
    )

    # Free physical slots in the buffer
    state.buffer.free(phys_to_free)

    # Reset AIA scores for freed slots
    state.aia.reset_slots(torch.tensor(phys_to_free, device=DEVICE, dtype=torch.long))

    # Remove evicted tokens from the virtual→physical mapping
    state.table.rearrange(logical_to_remove)


def make_patched_forward(original_attn, state: LayerPageEvictState):
    """
    Create a replacement forward() for a single LlamaAttention module.

    This closure captures:
      - original_attn: the HF LlamaAttention instance (for its projection weights)
      - state: the per-layer page-evict state (buffer, table, aia, evictor)

    The patched forward replaces HF's DynamicCache with our buffer-managed pipeline:
      1. Q/K/V projection + RoPE  (unchanged — uses HF's weights)
      2. Allocate new K/V tokens into BufferManager
      3. Gather full K/V cache from buffer via MappingTable
      4. Compute attention (eager matmul — needed for AIA)
      5. Update AIA importance scores (async CUDA stream)
      6. Output projection  (unchanged — uses HF's o_proj)
    """
    # Import HF's helpers — we need these inside the closure
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        repeat_kv,
    )
    from kernel import gather_attention
    from kernel import gather_attention

    def patched_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,  # see PageEvictCache: K/V stay in BufferManager;
                               # we only call .update() to tick the seq-length counter
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # ── Step 1: Q/K/V Projection ──
        # This is identical to HuggingFace's LlamaAttention.forward().
        # We use the original module's linear layers (q_proj, k_proj, v_proj)
        # which are already loaded with the pretrained weights.
        input_shape = hidden_states.shape[:-1]  # (batch, seq_len)
        hidden_shape = (*input_shape, -1, original_attn.head_dim)

        query_states = original_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = original_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = original_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # Shapes: query  = [1, num_q_heads(32), seq_len, head_dim(128)]
        #         key    = [1, num_kv_heads(8), seq_len, head_dim(128)]
        #         value  = [1, num_kv_heads(8), seq_len, head_dim(128)]

        # ── Step 2: Apply Rotary Position Embeddings ──
        # RoPE encodes positional information into Q and K.
        # Again, identical to HF's implementation.
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Advance the cache shim's per-layer seq-length counter so HF generate
        # slices input_ids correctly on the next decode step and computes
        # correct RoPE positions. The shim stores no K/V — we keep using buffer.
        if past_key_values is not None:
            past_key_values.update(key_states, value_states, state.layer_idx)

        # ── Step 3: Allocate New Tokens into Buffer ──
        # During prefill: seq_len = prompt length (many tokens)
        # During decode:  seq_len = 1 (one new token)
        #
        # We process tokens one-at-a-time to maintain causality during prefill.
        # (Token i should only attend to tokens 0..i-1, not future tokens.)
        bsz, num_kv_heads, seq_len, head_dim = key_states.shape

        for t in range(seq_len):
            # Check if eviction is needed BEFORE allocating
            _try_evict(state)

            # Extract this token's K and V vectors
            k_token = key_states[0, :, t, :]   # [num_kv_heads, head_dim]
            v_token = value_states[0, :, t, :]  # [num_kv_heads, head_dim]

            # Allocate into the physical buffer → returns slot index
            slot = state.buffer.allocate(k_token, v_token)

            if slot == -1:
                # Buffer still full after eviction — shouldn't happen with correct config
                raise RuntimeError(
                    f"Layer {state.layer_idx}: Buffer allocation failed after eviction. "
                    f"Buffer occupancy: {state.buffer.get_occupancy():.2%}"
                )

            # Record the virtual→physical mapping
            state.table.add_index(slot)

        # ── Step 4 & 5: Compute Attention ──
        physical_indices = state.table.get_physical_indices()  # [active_tokens] (long tensor)
        active_tokens = len(physical_indices)

        if seq_len == 1:
            # DECODE PHASE: Use the optimized Triton gather_attention kernel
            # The kernel reads directly from the physical buffer slots, avoiding contiguous gathering
            attn_output = gather_attention(
                query_states,
                state.buffer.k_buffer,
                state.buffer.v_buffer,
                physical_indices,
                active_tokens
            )
            # We skip updating AIA during decode to maximize throughput
            attn_weights = None
        else:
            # PREFILL PHASE: Use eager attention to get raw attention weights for AIA
            k_cache = state.buffer.k_buffer[physical_indices]  # [active_tokens, num_kv_heads, head_dim]
            v_cache = state.buffer.v_buffer[physical_indices]   # [active_tokens, num_kv_heads, head_dim]

            # Reshape for attention: [batch=1, num_kv_heads, active_tokens, head_dim]
            k_cache = k_cache.unsqueeze(0).transpose(1, 2)
            v_cache = v_cache.unsqueeze(0).transpose(1, 2)

            # GQA expansion: replicate 8 KV heads → 32 Q heads
            k_cache = repeat_kv(k_cache, original_attn.num_key_value_groups)
            v_cache = repeat_kv(v_cache, original_attn.num_key_value_groups)
            # k_cache, v_cache: [1, 32, active_tokens, 128]

            attn_weights = torch.matmul(
                query_states, k_cache.transpose(2, 3)
            ) * original_attn.scaling
            # attn_weights: [1, num_q_heads, seq_len, active_tokens]

            # Apply causal mask during prefill
            causal_mask = torch.triu(
                torch.full((seq_len, active_tokens), float("-inf"), device=DEVICE, dtype=query_states.dtype),
                diagonal=active_tokens - seq_len + 1,
            )
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_weights = attn_weights.to(query_states.dtype)

            attn_output = torch.matmul(attn_weights, v_cache)
            # attn_output: [1, num_q_heads, seq_len, head_dim]

            # Update AIA Importance Scores (Async)
            if state.aia.async_stream is not None:
                with torch.cuda.stream(state.aia.async_stream):
                    attn_scores_to_add = attn_weights[0].sum(dim=(0, 1))
                    state.aia.update(physical_indices, attn_scores_to_add.detach())
            else:
                attn_scores_to_add = attn_weights[0].sum(dim=(0, 1))
                state.aia.update(physical_indices, attn_scores_to_add.detach())

        # ── Step 7: Output Projection ──
        # Identical to HF's implementation.
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = original_attn.o_proj(attn_output)

        return attn_output, attn_weights

    return patched_forward


# ═══════════════════════════════════════════════════════════════
# Apply Monkeypatch
# ═══════════════════════════════════════════════════════════════

def apply_page_evict(model) -> tuple[list[LayerPageEvictState], PageEvictCache]:
    """
    Replace every LlamaAttention.forward() with the page-evict version.

    Returns:
      - layer_states: per-layer buffer/table/AIA/evictor state
      - cache: a PageEvictCache shim to pass to model.generate(past_key_values=..., use_cache=True)
    """
    config = model.config
    num_kv_heads = config.num_key_value_heads   # 8 for LLaMA 3-8B
    head_dim = config.hidden_size // config.num_attention_heads  # 128
    num_layers = config.num_hidden_layers  # 32

    print(f"\nApplying page-evict monkeypatch to {num_layers} layers...")
    print(f"  Physical buffer: {MAX_PHYSICAL_SLOTS} slots per layer")
    print(f"  Virtual context: {MAX_VIRTUAL_CONTEXT} max tokens")
    print(f"  Recency window:  {WINDOW_SIZE} tokens protected")
    print(f"  AIA gamma:       {GAMMA}")
    print(f"  Per-token KV:    2 × {num_kv_heads} × {head_dim} × 2B = "
          f"{2 * num_kv_heads * head_dim * 2 / 1024:.1f} KB")
    print(f"  Buffer per layer: {MAX_PHYSICAL_SLOTS * 2 * num_kv_heads * head_dim * 2 / 1024 / 1024:.1f} MB")

    layer_states = []
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        state = LayerPageEvictState(layer_idx, num_kv_heads, head_dim, DEVICE, model.dtype)

        # Replace the forward method with our patched version
        layer.self_attn.forward = make_patched_forward(layer.self_attn, state)

        layer_states.append(state)

    cache = PageEvictCache()

    print(f"  Total buffer memory: "
          f"{num_layers * MAX_PHYSICAL_SLOTS * 2 * num_kv_heads * head_dim * 2 / 1024 / 1024 / 1024:.2f} GB")
    print("Monkeypatch applied.\n")

    return layer_states, cache


def reset_all_states(layer_states: list[LayerPageEvictState], cache: PageEvictCache | None = None):
    """Reset all per-layer states (and the HF cache shim) for processing a new sequence."""
    for state in layer_states:
        state.reset()
    if cache is not None:
        cache.reset()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    # ── Load model ──
    model, tokenizer = load_model()

    # ── Apply the monkeypatch ──
    layer_states, cache = apply_page_evict(model)

    # ── Tokenize prompt ──
    prompt = "The key insight behind attention mechanisms is"
    print(f"Prompt: {prompt!r}")
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    print(f"Input tokens: {input_len}\n")

    # ── Generate ──
    print("Generating...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,           # greedy decoding for reproducibility
            use_cache=True,            # required: lets HF slice input_ids to 1 token on decode steps
            past_key_values=cache,     # PageEvictCache shim — owns no K/V, just tracks seq length
        )

    # ── Decode and print ──
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n{'═' * 60}")
    print(f"Generated text:")
    print(f"{'═' * 60}")
    print(generated_text)
    print(f"{'═' * 60}")

    # ── Print diagnostics ──
    print(f"\nDiagnostics:")
    for i, state in enumerate(layer_states):
        occ = state.buffer.get_occupancy()
        active = state.table.write
        if i == 0 or i == len(layer_states) - 1 or occ > 0.9:
            print(f"  Layer {i:2d}: buffer occupancy={occ:.1%}, "
                  f"active_tokens={active}, "
                  f"write_head={state.buffer.write_head}, "
                  f"free_list_len={len(state.buffer.free_list)}")

    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    print(f"\n  Peak GPU memory: {peak_mem:.2f} GB")


if __name__ == "__main__":
    main()
