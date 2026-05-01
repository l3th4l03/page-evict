import torch


class MappingTable:
    def __init__(self, max_slots: int):
        self.max_slots = max_slots
        self.map_tensor = torch.zeros(max_slots)
        # index where you will write the next token
        self.write = 0

    def add_index(self, buffer_loc):
        self.map_tensor[self.write] = buffer_loc
        self.write += 1

    def rearrange(self, remove_arr_logic: list[int]):
        mask_tensor = torch.ones(self.write, dtype=torch.bool)
        mask_tensor[remove_arr_logic] = False
        temp_map = self.map_tensor[:self.write]
        temp_map = temp_map[mask_tensor]
        self.write = len(temp_map)
        self.map_tensor[:self.write] = temp_map
        
    
    