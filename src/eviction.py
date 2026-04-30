class EvictController:
    def __init__(self, max_slots: int, window_size: int):
        self.max_slots = max_slots
        self.window_size = window_size
    