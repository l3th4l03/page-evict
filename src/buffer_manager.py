import torch


class BufferManager:
    def __init__ (self, max_slots, num_kv_heads, head_dim):
        self.max_slots = max_slots
        self.k_buffer = torch.zeros(max_slots, num_kv_heads, head_dim)
        self.v_buffer = torch.zeros(max_slots, num_kv_heads, head_dim)
        # slots/data that is freed after being occupied
        self.free_list = []
        # next spot to write data in
        self.write_head = 0

    # how much space used
    def get_occupancy(self):
        return self.write_head / self.max_slots

    # find the open memory slot for incoming kv caches
    def allocate(self) -> int:
        # first check if there's enough space -> deal with evicting (evict controller)
        if self.get_occupancy() >= 0.95 and len(self.free_list) == 0:
            return -1

        #otherwise return open slot
        if len(self.free_list) > 0:
            curr_write = self.free_list.pop()
        else:
            curr_write = self.write_head
            self.write_head += 1

        return curr_write


    # write the kv cache into memory
    def write(self, key, value):
        write_loc = self.allocate()
        
        