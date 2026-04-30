import torch


class BufferManager:
    def __init__ (self, max_slots, num_kv_heads, head_dim):
        self.k_buffer = torch.zeros(max_slots, num_kv_heads, head_dim)
        self.v_buffer = torch.zeros(max_slots, num_kv_heads, head_dim)
        # slots/data that is freed after being occupied
        self.free_list = []
        # next spot to write data in
        self.write_head = 0

    # define this func
    def allocate(self):
        