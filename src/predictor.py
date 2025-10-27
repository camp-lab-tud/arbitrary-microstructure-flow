from abc import ABC, abstractmethod
from typing import Union, Any
import json
import os
import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage

from .normalizer import Normalizer, MaxNormalizer
from .unet.models import UNet
from utils.zenodo import download_data, unzip_data, is_url


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
    
    @property
    @abstractmethod
    def type(self) -> str:
        pass

    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def predict(self, *args):
        pass

    @abstractmethod
    def pre_process(self, *args):
        pass

    @property
    def trainable_params(self):
        total = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )
        return total
    
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

    def load_weights(self, model_path: str, device: str):
        """Load model parameters from `.pt` file."""
        self.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device(device)
            )
        )
        print(f'Loaded weights from "{model_path}".')

    @classmethod
    def from_directory(cls, folder: str, device: str) -> 'Predictor':
        """
        Load trained ML model from folder.
        
        Args:
            folder: Directory containing `model.pt` and `log.json` files created during training.
            device: Device (e.g. "cuda:0" or "cpu") to map the model to.
        """

        log_file = osp.join(folder, 'log.json')
        model_path = osp.join(folder, 'model.pt')

        with open(log_file) as fp:
            log_data = json.load(fp)
        
        param_dict = log_data['params']
        predictor_type = param_dict['training']['predictor_type']
        predictor_kwargs = param_dict['training']['predictor']

        if predictor_type == 'velocity':
            predictor_class = VelocityPredictor
        elif predictor_type == 'pressure':
            predictor_class = PressurePredictor
        else:
            raise ValueError(f'Unknown predictor type: {predictor_type}')

        predictor = predictor_class(**predictor_kwargs)
        predictor.to(device)
        predictor.load_weights(model_path, device=device)
        return predictor

    @classmethod
    def from_url(cls, url: str, device: str) -> 'Predictor':
        """
        Load trained ML model from URL. Pre-trained models are hosted here: https://doi.org/10.5281/zenodo.17306446.
        
        Args:
            url: URL pointing to a zipped folder containing `model.pt` and `log.json` files created during training.
            device: Device (e.g. "cuda:0" or "cpu") to map the model to.
        """

        _folder = 'pretrained'
        if not osp.exists(_folder): os.mkdir(_folder)

        # download pre-trained weights
        zip_path = download_data(url=url, save_dir=_folder)

        # unzip data
        folder_path = unzip_data(zip_path=zip_path, save_dir=_folder)

        predictor = cls.from_directory(folder_path, device=device)
        return predictor

    @classmethod
    def from_directory_or_url(
        cls,
        directory_or_url: str,
        device: str
    ) -> 'Predictor':
        """
        Load trained ML model from local directory or URL.
        
        Args:
            directory_or_url: either local directory or URL of the pre-trained model.
            device: Device (e.g. "cuda:0" or "cpu") to map the model to.
        """

        if is_url(directory_or_url):
            predictor = cls.from_url(url=directory_or_url, device=device)
        else:
            predictor = cls.from_directory(folder=directory_or_url, device=device)
        return predictor


class VelocityPredictor(Predictor):
    """
    Model for velocity field prediction in microstructures.
    """
    type: str = 'velocity'

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
        print(f'Initialized {self.type} predictor with {self.trainable_params} parameters.')

    def forward(self, img: torch.Tensor):
        """
        Forward pass for the model.

        Args:
            img: (binary) microstructure images with 1 in fluid areas and 0 in fiber areas. Shape: (batch, 1, height, width).
        """

        mask = img.clone()

        feats = self.pre_process(img)
        out = self.model(feats)

        out = out * mask # multiply by mask

        return out

    def predict(self, img: torch.Tensor):
        """
        Predict velocity field from input microstructure images.

        Args:
            img: (binary) microstructure images with 1 in fluid areas and 0 in fiber areas. Shape: (batch, 1, height, width).
        """

        pred = self(img)
        out = self.normalizer['output'].inverse(pred)
        return out

    def pre_process(self, img: torch.Tensor):
        """
        Pre-process inputs.

        Args:
            img: (binary) microstructure images with 1 in fluid areas and 0 in fiber areas. Shape: (batch, 1, height, width).
        """
        assert img.dim() == 4
        assert img.shape[1] == 1 # only 1 channel

        if self.distance_transform:
            img = apply_distance_transform(img)

        features = self.normalizer['input'](img)

        return features


class PressurePredictor(Predictor):
    """
    Model for pressure field prediction in microstructures.
    """
    type: str = 'pressure'

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
        print(f'Initialized {self.type} predictor with {self.trainable_params} parameters.')

    def forward(
        self,
        img: torch.Tensor,
        x_length: torch.Tensor
    ):
        """
        Forward pass for the model.

        Args:
            img: (binary) microstructure images with 1 in fluid areas and 0 in fiber areas. Shape: (batch, 1, height, width).
            x_length: Microstructure (physical) length. Shape: (batch, 1) or (batch, 1, height, width).
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
        """
        Predict pressure field from input microstructure images.
        
        Args:
            img: (binary) microstructure images with 1 in fluid areas and 0 in fiber areas. Shape: (batch, 1, height, width).
            x_length: Microstructure (physical) length. Shape: (batch, 1) or (batch, 1, height, width).
        """

        pred = self(img, x_length)
        out = self.normalizer['output'].inverse(pred) # ρ-normalized pressure
        return out
    

    def pre_process(
        self,
        img: torch.Tensor,
        x_length: torch.Tensor
    ):
        """
        Pre-process inputs.

        Args:
            img: (binary) microstructure images with 1 in fluid areas and 0 in fiber areas. Shape: (batch, 1, height, width).
            x_length: Microstructure (physical) length. Shape: (batch, 1) or (batch, 1, height, width).
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
        Evaluate whether microstructures' length is passed as scalars or matrices. Returns a 4D tensor.
        
        Args:
            x_length: microstructure (physical) length.
            shape: shape (samples, 1, height, width) of microstructure images. Used to expand `x_length` if needed.
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

        Args:
            img: (binary) microstructure images with 1 in fluid areas and 0 in fiber areas. Shape: (batch, 1, height, width).
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
    Perform distance transform of input images.

    Args:
        imgs: batch of images, with shape (n_img, 1, height, width)
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
