#Asnyc Importance Accumlator (AIA)
import torch

class AIA:
    def __init__ (self, max_physical_slots: int, gamma: float = 0.99):
        self.gamma = gamma
        self.max_physical_slots = max_physical_slots
        self.scores = torch.zeros(max_physical_slots, dtype=torch.float32, device="cuda")
        self.async_stream = torch.cuda.Stream() #seperate CUDA stream to keep it async

    def update():
        
    def reset_slots():
        
    def get_scores():
        return self.scores
    
        
