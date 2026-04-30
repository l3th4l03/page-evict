import torch


class BufferManager:
    def __init__ (self, max_slots: int, num_kv_heads: int, head_dim: int):
        self.max_slots = max_slots
        self.k_buffer = torch.zeros(max_slots, num_kv_heads, head_dim)
        self.v_buffer = torch.zeros(max_slots, num_kv_heads, head_dim)
        # slots/data that is freed after being occupied
        self.free_list = []
        # next spot to write data in
        self.write_head = 0

    # how much space used -> if 95% of buffer used, evict control will be called in forward pass
    def get_occupancy(self) -> float:
        return float(self.write_head - len(self.free_list)) / float(self.max_slots)

    # find the open memory slot and write incoming kv caches
    def allocate(self, key: torch.Tensor, value: torch.Tensor):
        # first check if there's enough space -> deal with evicting (evict controller)
        if self.get_occupancy() >= 0.95 and len(self.free_list) == 0:
            return -1

        #otherwise return open slot
        if len(self.free_list) > 0:
            curr_write = self.free_list.pop()
        else:
            curr_write = self.write_head
            self.write_head += 1

        self.k_buffer[curr_write] = key
        self.v_buffer[curr_write] = value
        return curr_write


    # free up all required tokens
    def free(self, indices: list[int]):
        for i in indices: self.free_list.append(i)