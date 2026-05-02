"""
Tests for the Async Importance Accumulator (AIA).
Run with: python3 -m pytest testing/importance_test.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.importance import AIA

DEVICE = "cpu"  # use "cuda" when GPU is available
GAMMA = 0.99


# --- Helper to build fake attention weights ---
def make_attn_weights(per_token_scores: list[float]) -> torch.Tensor:
    """
    Simulate attention_weights shape [1, num_heads, 1, seq_len].
    Each token gets uniform attention across heads so that
    sum(dim=1) yields the target per-token score.
    """
    seq_len = len(per_token_scores)
    num_heads = 8  # arbitrary, sum across heads = num_heads * per_head_val
    attn = torch.zeros(1, num_heads, 1, seq_len, device=DEVICE)
    for i, s in enumerate(per_token_scores):
        attn[0, :, 0, i] = s / num_heads  # each head contributes equally
    return attn


# =========================================================
# 1. EMA formula correctness
# =========================================================
class TestEMAFormula:
    def test_single_update_from_zero(self):
        """Score starts at 0, one update with attention=1.0 → score = 1.0."""
        aia = AIA(max_slots=4, gamma=GAMMA, device=DEVICE)
        indices = torch.tensor([0, 1], device=DEVICE)
        attn = make_attn_weights([1.0, 0.5])

        aia.update(indices, attn)
        scores = aia.get_scores()

        assert torch.isclose(scores[0], torch.tensor(1.0)), f"Expected 1.0, got {scores[0]}"
        assert torch.isclose(scores[1], torch.tensor(0.5)), f"Expected 0.5, got {scores[1]}"

    def test_two_updates_ema(self):
        """Verify exact EMA math over two steps."""
        aia = AIA(max_slots=4, gamma=GAMMA, device=DEVICE)
        indices = torch.tensor([0], device=DEVICE)

        # Step 1: score = 0.99 * 0 + 1.0 = 1.0
        aia.update(indices, make_attn_weights([1.0]))
        # Step 2: score = 0.99 * 1.0 + 0.5 = 1.49
        aia.update(indices, make_attn_weights([0.5]))
        scores = aia.get_scores()

        expected = GAMMA * 1.0 + 0.5  # 1.49
        assert torch.isclose(scores[0], torch.tensor(expected)), \
            f"Expected {expected}, got {scores[0]}"


# =========================================================
# 2. Decay test — unattended tokens decrease over iterations
# =========================================================
class TestDecay:
    def test_unattended_scores_decrease(self):
        """
        Token 0 gets attention once, then is repeatedly updated
        with zero attention. Its score should decrease each step.
        """
        aia = AIA(max_slots=4, gamma=GAMMA, device=DEVICE)
        idx = torch.tensor([0], device=DEVICE)

        # Give initial attention
        aia.update(idx, make_attn_weights([5.0]))
        prev_score = aia.get_scores()[0].item()

        # 10 steps with zero attention — score should decay each time
        for _ in range(10):
            aia.update(idx, make_attn_weights([0.0]))
            curr_score = aia.get_scores()[0].item()
            assert curr_score < prev_score, \
                f"Score should decrease: {curr_score} >= {prev_score}"
            prev_score = curr_score

        # After 10 decays: 5.0 * 0.99^10 ≈ 4.52
        expected = 5.0 * (GAMMA ** 10)
        assert abs(prev_score - expected) < 1e-4, \
            f"Expected ~{expected:.4f}, got {prev_score:.4f}"


# =========================================================
# 3. Score ordering — frequent attention > rare attention
# =========================================================
class TestScoreOrdering:
    def test_high_attention_ranks_higher(self):
        """Token receiving more attention should have a higher score."""
        aia = AIA(max_slots=4, gamma=GAMMA, device=DEVICE)
        indices = torch.tensor([0, 1], device=DEVICE)

        for _ in range(20):
            aia.update(indices, make_attn_weights([1.0, 0.1]))

        scores = aia.get_scores()
        assert scores[0] > scores[1], \
            f"High-attention token should rank higher: {scores[0]} vs {scores[1]}"


# =========================================================
# 4. Reset slots — evicted slots are properly zeroed
# =========================================================
class TestResetSlots:
    def test_reset_zeroes_correct_slots(self):
        """After reset, evicted slots should be 0 and others unchanged."""
        aia = AIA(max_slots=8, gamma=GAMMA, device=DEVICE)
        all_indices = torch.tensor([0, 1, 2, 3, 4], device=DEVICE)

        # Give all 5 tokens some attention
        for _ in range(5):
            aia.update(all_indices, make_attn_weights([1.0, 2.0, 3.0, 4.0, 5.0]))

        scores_before = aia.get_scores()

        # Evict slots 1 and 3 (simulating what eviction controller would return)
        evicted = torch.tensor([1, 3], device=DEVICE)
        aia.reset_slots(evicted)

        scores_after = aia.get_scores()

        # Evicted slots should be zero
        assert scores_after[1].item() == 0.0, f"Slot 1 should be 0, got {scores_after[1]}"
        assert scores_after[3].item() == 0.0, f"Slot 3 should be 0, got {scores_after[3]}"

        # Non-evicted slots should be unchanged
        assert scores_after[0] == scores_before[0], "Slot 0 should be unchanged"
        assert scores_after[2] == scores_before[2], "Slot 2 should be unchanged"
        assert scores_after[4] == scores_before[4], "Slot 4 should be unchanged"

    def test_reset_then_reuse(self):
        """After reset + new attention, slot starts fresh from 0."""
        aia = AIA(max_slots=4, gamma=GAMMA, device=DEVICE)
        idx = torch.tensor([0], device=DEVICE)

        # Build up a score
        for _ in range(50):
            aia.update(idx, make_attn_weights([1.0]))
        old_score = aia.get_scores()[0].item()
        assert old_score > 10, f"Should have accumulated a high score, got {old_score}"

        # Reset and add a single small attention
        aia.reset_slots(idx)
        aia.update(idx, make_attn_weights([0.1]))
        new_score = aia.get_scores()[0].item()

        # Should be 0.1, not old_score * gamma + 0.1
        assert torch.isclose(torch.tensor(new_score), torch.tensor(0.1)), \
            f"After reset, score should be 0.1, got {new_score}"


# =========================================================
# 5. Partial index update — untouched slots stay unchanged
# =========================================================
class TestPartialUpdate:
    def test_untouched_indices_unchanged(self):
        """Updating indices [0,1,2] should not affect indices [3,4,5]."""
        aia = AIA(max_slots=8, gamma=GAMMA, device=DEVICE)

        # Update only slots 0, 1, 2
        indices = torch.tensor([0, 1, 2], device=DEVICE)
        aia.update(indices, make_attn_weights([1.0, 1.0, 1.0]))

        scores = aia.get_scores()

        # Slots 3-7 should still be exactly 0
        for i in range(3, 8):
            assert scores[i].item() == 0.0, \
                f"Slot {i} should be 0.0, got {scores[i].item()}"


# =========================================================
# 6. get_scores returns a true copy
# =========================================================
class TestCopySemantics:
    def test_modifying_copy_does_not_affect_internal(self):
        """Mutating the returned scores should not change AIA internals."""
        aia = AIA(max_slots=4, gamma=GAMMA, device=DEVICE)
        idx = torch.tensor([0], device=DEVICE)
        aia.update(idx, make_attn_weights([5.0]))

        returned = aia.get_scores()
        original_val = returned[0].item()

        # Mutate the returned tensor
        returned[0] = 999.0

        # Internal score should be unchanged
        fresh = aia.get_scores()
        assert fresh[0].item() == original_val, \
            f"Internal score should be {original_val}, got {fresh[0].item()}"


# =========================================================
# 7. Edge case — empty indices
# =========================================================
class TestEdgeCases:
    def test_empty_indices_no_crash(self):
        """Calling update/reset with empty tensors should not crash."""
        aia = AIA(max_slots=4, gamma=GAMMA, device=DEVICE)
        empty = torch.tensor([], dtype=torch.long, device=DEVICE)
        empty_attn = torch.zeros(1, 8, 1, 0, device=DEVICE)

        aia.update(empty, empty_attn)
        aia.reset_slots(empty)
        scores = aia.get_scores()

        # All scores should still be zero
        assert torch.all(scores == 0.0), "All scores should remain 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
