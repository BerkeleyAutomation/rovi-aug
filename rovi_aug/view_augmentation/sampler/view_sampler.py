import torch

class ViewSampler:
    def sample(self, start_view: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Samples SE(3) elements and returns as a batch tensor of size (B, ...)
        """
        raise NotImplementedError