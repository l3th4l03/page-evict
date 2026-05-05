import torch

class EvictController:
    def __init__(self, max_slots: int, window_size: int):
        self.max_slots = max_slots
        self.window_size = window_size
    
    def evict(self, scores: torch.Tensor, table: torch.Tensor, occupied: int) -> tuple[list[int], list[int]]:
        tokens_removing = occupied - int(0.75 * self.max_slots)
        # Only score tokens outside the recency window (protect recent tokens)
        # We must index scores using the logical-to-physical table!
        evictable_scores = scores[table[:-self.window_size]]
        _, remove_arr_logic = torch.topk(evictable_scores, k=tokens_removing, largest=False)
        
        # logical indices relative to the active table (same as topk since we sliced from 0)
        remove_arr_logic = remove_arr_logic.tolist()
        # Map logical indices → physical indices via the mapping table
        remove_arr_phys = table[remove_arr_logic].long().tolist()

        return (remove_arr_phys, remove_arr_logic)