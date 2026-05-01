#Asnyc Importance Accumlator (AIA)
import torch

class AIA:
    def __init__ (self, max_physical_slots: int, gamma: float = 0.99):
        self.gamma = gamma
        self.max_physical_slots = max_physical_slots
        self.scores = torch.zeros(max_physical_slots, dtype=torch.float32, device="cuda")
        self.async_stream = torch.cuda.Stream() #seperate CUDA stream to keep it async

    def update(self, physical_indices: torch.Tensor, attention_weights: torch.Tensor):
        # have to do all ts in its own cuda stream so its async
        with torch.cuda.stream(self.async_stream):
            # formula is New_Score = (Old_Score * gamma) + Summed_Attention_For_Current_Step
            summed_attention = torch.sum(attention_weights, dim=1).view(-1)
            self.scores[physical_indices] *= self.gamma
            self.scores[physical_indices] += summed_attention

    def reset_slots(self, physical_indices: torch.Tensor):
        self.async_stream.synchronize()
        self.scores[physical_indices] = 0.0
        
    def get_scores(self):
        self.async_stream.synchronize()
        return self.scores.clone() #this is a copy
    
        
