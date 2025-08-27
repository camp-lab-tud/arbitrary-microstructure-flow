from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Normalizer(ABC, nn.Module):

    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def normalize(self, *args):
        pass

    @abstractmethod
    def inverse(self, *args):
        pass


class MaxNormalizer(Normalizer):

    def __init__(
        self,
        scale_factors: tuple | list = (1,)
    ):
        """
        `scale_factors`: scale factors for each channel.
        """
        super().__init__()
        
        assert isinstance(scale_factors, (tuple, list))

        self.scale_factors = nn.Parameter(
            torch.Tensor(scale_factors),
            requires_grad=False
        )

    def forward(self, x: torch.Tensor):
        """
        `x`: tensor with shape: (samples, channels, height, width)
        """
        return self.normalize(x)
        
    def normalize(self, x: torch.Tensor):
        assert x.dim() == 4

        # normalize
        out = x / self.scale_factors.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return out

    def inverse(self, x: torch.Tensor):
        assert x.dim() == 4

        # inverse normalize
        out = x * self.scale_factors.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return out
