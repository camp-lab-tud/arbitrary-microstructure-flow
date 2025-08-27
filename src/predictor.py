from abc import ABC, abstractmethod
from typing import Union, Any

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage

from .normalizer import Normalizer, MaxNormalizer
from .unet.models import UNet


_model_type = Union[UNet, Any]


class Predictor(ABC, nn.Module):

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict,
        distance_transform: bool,
    ):
        super().__init__()

        self.model_name = model_name
        self.model: _model_type = eval(model_name)(**model_kwargs)

        in_channels = model_kwargs['in_channels']
        out_channels = model_kwargs['out_channels']
        self.normalizer: dict[str, Normalizer] = self.init_normalizer(
            in_channels=in_channels,
            out_channels=out_channels
        )

        self.distance_transform = nn.Parameter(
            torch.Tensor([distance_transform]),
            requires_grad=False
        )

        print(f'Trainable parameters: {self.trainable_params}.')

    @property
    def trainable_params(self):
        total = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )
        return total
    
    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def predict(self, *args):
        pass

    @abstractmethod
    def pre_process(self, *args):
        pass

    def init_normalizer(self, in_channels: int, out_channels: int):
        """Initialize normalizers for model input & output"""

        self.normalizer = nn.ModuleDict({
            'input': MaxNormalizer(scale_factors=[1 for _ in range(in_channels)]),
            'output': MaxNormalizer(scale_factors=[1 for _ in range(out_channels)])
        })
        return self.normalizer
    
    def set_normalizer(
        self,
        norm_dict: dict[str, Union[tuple, list, None]]
    ):
        """Set parameters for normalizers."""

        for key, val in norm_dict.items():
            if val is not None:
                self.normalizer[key] = MaxNormalizer(scale_factors=val)

        return self.normalizer


class VelocityPredictor(Predictor):
    """
    Model for predicting velocity field in microstructures.
    """

    def __init__(
        self,
        model_name='UNet',
        model_kwargs: dict = {},
        distance_transform = True,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            distance_transform=distance_transform
        )

    def forward(self, img: torch.Tensor):
        """
        `x`: (binary) microstructure images with 1 in fluid areas. Shape: (batch, 1, height, width).
        """

        mask = img.clone()

        feats = self.pre_process(img)
        out = self.model(feats)

        out = out * mask # multiply by mask

        return out

    def predict(self, img: torch.Tensor):
        
        pred = self(img)
        out = self.normalizer['output'].inverse(pred)
        return out

    def pre_process(self, img: torch.Tensor):
        """
        Pre-process inputs.

        `img`: (binary) microstructure images, with 1 in fluid areas. Shape: (batch, 1, height, width).
        """
        assert img.dim() == 4
        assert img.shape[1] == 1 # only 1 channel

        if self.distance_transform:
            img = apply_distance_transform(img)

        features = self.normalizer['input'](img)

        return features


class PressurePredictor(Predictor):
    """
    Model for predicting pressure field in microstructures.
    """

    def __init__(
        self,
        model_name='UNet',
        model_kwargs: dict = {},
        distance_transform = False,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            distance_transform=distance_transform
        )

    def forward(
        self,
        img: torch.Tensor,
        x_length: torch.Tensor
    ):
        """
        `img`: (binary) microstructure images with 1 in fluid areas. Shape: (batch, 1, height, width).\n
        `x_length`: Microstructure length. Shape: (batch, 1) or (batch, 1, height, width).
        """

        mask = img.clone()

        feats = self.pre_process(img, x_length)
        out = self.model(feats)

        out = out * mask # multiply by mask

        return out

    def predict(
        self,
        img: torch.Tensor,
        x_length: torch.Tensor
    ):
        
        pred = self(img, x_length)
        out = self.normalizer['output'].inverse(pred)
        return out
    

    def pre_process(
        self,
        img: torch.Tensor,
        x_length: torch.Tensor
    ):
        """
        Pre-process inputs.

        `img`: (binary) microstructure images with 1 in fluid areas. Shape: (batch, 1, height, width).\n
        `x_length`: Microstructure length. Shape: (batch, 1) or (batch, 1, height, width).
        """

        x_length = self._process_microstructure_length(x_length, img.shape)
        assert img.dim() == 4
        assert x_length.dim() == 4
        img_copy = img.clone()

        if self.distance_transform:
            img = apply_distance_transform(img)

        features = torch.cat(
            (img, x_length),
            dim=1
        ) # shape: (samples, 2, height, width)

        features = self.normalizer['input'](features)


        """Additional modifications"""

        # multiply microstructure by fiber volume fraction
        fiber_vf = self._compute_fiber_vf(img_copy)
        features[:, [0]] = features[:, [0]] * fiber_vf

        # use inverse of microstructure length
        features[:, [1]] = 1 / features[:, [1]]

        return features


    @staticmethod
    def _process_microstructure_length(
        x_length: torch.Tensor,
        shape: tuple
    ):
        """
        Evaluate whether microstructures' length is passed as scalars or matrices.\n
        Returns a 4D tensor.
        
        `x_length`: microstructre length.\n
        `shape`: (samples, 1, height, width).
        """
        if x_length.dim() == 4:
            # shape: (samples, 1, height, width)
            pass
        else:
            x_length = x_length.squeeze() # shape: (samples,)
            x_length = x_length.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # shape: (samples, 1, 1, 1)
            x_length = x_length * torch.ones(shape, device=x_length.device)

        return x_length

    @staticmethod
    def _compute_fiber_vf(img: torch.Tensor):
        """
        Compute fiber volume fraction
        
        `img`: microstructure image. Shape: (samples, 1, height, width).
        """

        fluid_vf = torch.sum(
            img,
            dim=[-1,-2]
        ) / (img.shape[-1] * img.shape[-2])

        fiber_vf = 1 - fluid_vf
        fiber_vf = fiber_vf.unsqueeze(-1).unsqueeze(-1) # shape: (samples, 1, height, width)

        return fiber_vf
        





def apply_distance_transform(imgs: torch.Tensor):
    """
    Perform distance tranform of input images.

    `imgs`: batch of images of shape (n_img, 1, x, x)
    """
    device = imgs.device

    imgs = imgs.cpu().numpy()

    tmp_list = []
    for im in imgs:
        im = im[0]
        im_tr = ndimage.distance_transform_edt(im)
        tmp_list.append([[im_tr]])
    
    out = torch.from_numpy(
        np.concatenate(tmp_list)
    ).float()
    return out.to(device)