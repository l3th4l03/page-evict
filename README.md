# page-evict

**Page Table, Intelligent Eviction Policy, and Custom Triton Kernel for Fragmented KV Cache Attention**

A hybrid KV cache compression system for LLM inference that reduces DRAM usage by up to 4× without significantly degrading model quality. Validated on Llama-3-8B with LongBench, Needle-in-Haystack, and PG19 benchmarks.

---

## Overview

During autoregressive generation, LLMs store Key/Value (KV) tensors for every past token. At 32K context length on Llama-3-8B, this cache consumes **~128 MB per sequence per layer-set**, making memory the primary bottleneck for concurrent serving.

**page-evict** solves this by:
1. Tracking per-token importance via an **Asynchronous Importance Accumulator (AIA)** using EMA-decayed attention scores
2. **Evicting** low-importance tokens when the buffer reaches 95% capacity, while protecting a sliding recency window
3. Maintaining a **Virtual-to-Physical Mapping Table** so eviction never requires data movement
4. Using a custom **Triton Gather Attention Kernel** that performs block-sparse attention over the fragmented buffer

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Modified LlamaAttention.forward()           │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────┐   │
│  │  Buffer  │──▶│   Virtual    │──▶│  Gather Attention  │   │
│  │  Manager │   │ Mapping Table│   │  Kernel (Triton)   │   │
│  └──────────┘   └──────────────┘   └────────────────────┘   │
│       ▲                                      │              │
│       │              ┌──────────┐            │              │
│       └──────────────│ Eviction  │◀──────────┘              │
│                      │ Controller│◀── AIA (async stream)    │
│                      └──────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
page-evict/
├── README.md
├── page_evict/                  # Core library
│   ├── __init__.py
│   ├── config.py                # Hyperparameters & system config
│   ├── buffer_manager.py        # Physical K/V buffer allocation & free-list
│   ├── mapping_table.py         # Virtual-to-physical index mapping
│   ├── importance.py            # Async Importance Accumulator (AIA)
│   ├── eviction.py              # Eviction controller & compaction logic
│   ├── kernels/
│   │   ├── __init__.py
│   │   └── gather_attention.py  # Triton gather attention kernel
│   └── integration/
│       ├── __init__.py
│       └── llama_attention.py   # HuggingFace LlamaAttention override
├── tests/                       # Unit & integration tests
│   ├── test_buffer_manager.py
│   ├── test_mapping_table.py
│   ├── test_eviction.py
│   ├── test_gather_kernel.py
│   └── test_end_to_end.py
├── benchmarks/                  # Evaluation scripts
│   ├── run_longbench.py
│   ├── run_needle.py
│   └── run_perplexity.py
└── requirements.txt
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Importance tracking | Per-layer scalar (sum across heads) | Balances granularity vs. metadata overhead |
| Decay method | EMA (γ = 0.99) | Prevents "Attention Sink" bias toward early tokens |
| Sliding window | Fixed 512 tokens | Preserves local context (grammar, pronouns) |
| Compaction trigger/target | 95% → 75% | Avoids "yo-yo" effect of frequent compactions |
| Eviction selection | Top-K lowest scores outside window | Efficient O(N log K) on GPU |
| Buffer per layer | Independent | Avoids cross-layer kernel complexity |
| K/V storage | Separate buffers, shared mapping | A token's K and V always share the same fate |
| Kernel strategy | Block-sparse (64-token blocks) | Hardware-friendly memory coalescing |
| Softmax | Online (FlashAttention-style) | Required for 16K+ context performance |
| Buffer budget | 25–50% of max context | Enables 2–4× more concurrent sequences |

## Llama-3-8B Target Parameters

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 4096 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 (GQA) |
| `head_dim` | 128 |
| `num_hidden_layers` | 32 |
| Per-token KV size | 2 × 8 × 128 × 2B = **4 KB** |

## Benchmarks & Success Criteria

| Metric | Target |
|--------|--------|
| Peak GPU memory | ≤ 50% of dense KV cache |
| Tokens/sec (generation) | ≥ 90% of dense baseline |
| LongBench score | ≥ 95% of dense baseline |
| Needle-in-Haystack accuracy | ≥ 90% at all depths |
| PG19 perplexity | ≤ 5% degradation vs. dense |

**Baselines:** Dense KV cache, StreamingLLM, H2O (Heavy-Hitter Oracle)

## Setup

```bash
pip install -r requirements.txt
```

## License

MIT
