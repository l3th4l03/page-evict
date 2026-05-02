import torch


class MappingTable:
    def __init__(self, max_slots: int, device: str = "cuda"):
        self.max_slots = max_slots
        self.device = device
        self.map_tensor = torch.zeros(max_slots, dtype=torch.long, device=device)
        # index where you will write the next token
        self.write = 0

    def add_index(self, buffer_loc: int):
        self.map_tensor[self.write] = buffer_loc
        self.write += 1

    def get_physical_indices(self) -> torch.Tensor:
        """Return the active virtual→physical mapping as a long tensor."""
        return self.map_tensor[:self.write]

    def rearrange(self, remove_arr_logic: list[int]):
        mask_tensor = torch.ones(self.write, dtype=torch.bool, device=self.device)
        mask_tensor[remove_arr_logic] = False
        temp_map = self.map_tensor[:self.write]
        temp_map = temp_map[mask_tensor]
        self.write = len(temp_map)
        self.map_tensor[:self.write] = temp_map