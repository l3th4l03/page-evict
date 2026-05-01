import torch

class EvictController:
    def __init__(self, max_slots: int, window_size: int):
        self.max_slots = max_slots
        self.window_size = window_size
    
    def evict(self, scores: torch.Tensor, table: torch.Tensor, occupied: int) -> list[int]:
        tokens_removing = occupied - int(0.75 * self.max_slots)
        scores = scores[:len(table) - self.window_size]
        scores_v, scores_i = torch.topk(scores, k=tokens_removing, largest=False)
        remove_arr = [table[token] for token in scores_i]

        return remove_arr
            

    