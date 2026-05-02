import torch

class EvictController:
    def __init__(self, max_slots: int, window_size: int):
        self.max_slots = max_slots
        self.window_size = window_size
    
    def evict(self, scores: torch.Tensor, table: torch.Tensor, occupied: int) -> list[int]:
        tokens_removing = occupied - int(0.75 * self.max_slots)
        scores = scores[:len(table) - self.window_size]
        scores_v, remove_arr_logic = torch.topk(scores, k=tokens_removing, largest=False)
        remove_arr_phys = [table[token] for token in remove_arr_logic]

        return (remove_arr_phys, remove_arr_logic)
            

    