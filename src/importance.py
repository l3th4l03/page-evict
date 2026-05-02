#Asnyc Importance Accumlator (AIA)
import torch

class AIA:
    def __init__ (self, max_slots: int, gamma: float = 0.99, device: str = "cuda"):
        self.gamma = gamma
        self.max_slots = max_slots
        self.device = device
        self.scores = torch.zeros(max_slots, dtype=torch.float32, device=device)
        # separate CUDA stream to keep it async (only available on GPU)
        self.async_stream = torch.cuda.Stream() if device == "cuda" else None

    def _run_on_stream(self, fn):
        """Run fn on async CUDA stream if available, otherwise run synchronously."""
        if self.async_stream is not None:
            with torch.cuda.stream(self.async_stream):
                fn()
        else:
            fn()

    def update(self, physical_indices: torch.Tensor, attention_weights: torch.Tensor):
        def _update():
            # formula is New_Score = (Old_Score * gamma) + Summed_Attention_For_Current_Step
            summed_attention = torch.sum(attention_weights, dim=1).view(-1)
            self.scores[physical_indices] *= self.gamma
            self.scores[physical_indices] += summed_attention
        self._run_on_stream(_update)

    def reset_slots(self, physical_indices: torch.Tensor):
        if self.async_stream is not None:
            self.async_stream.synchronize()
        self.scores[physical_indices] = 0.0
        
    def get_scores(self):
        if self.async_stream is not None:
            self.async_stream.synchronize()
        return self.scores.clone() #this is a copy
    
        
