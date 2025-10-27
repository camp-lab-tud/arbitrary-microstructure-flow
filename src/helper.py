from typing import Callable, Literal, Union
import json
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from src.predictor import VelocityPredictor, PressurePredictor
from utils.zenodo import download_data, unzip_data, is_url


def get_norm_params(
    file: str,
    option: Literal['velocity', 'pressure']
):
    """
    Retrieve normalization parameters for dataset.

    Args:
        file: path to file with parameters.
    """

    stats = json.load(open(file))

    if option == 'velocity':
        max_velocity = stats['U']['max']

        out = {
            'input': None,
            'output': (max_velocity, max_velocity)
        }

    elif option == 'pressure':
        max_length = stats['dxyz']['max']
        max_pressure = stats['p']['max']

        out = {
            'input': (1, max_length),
            'output': (max_pressure,)
        }

    return out


def set_model(
    type: Literal['velocity', 'pressure'],
    kwargs: dict,
    norm_file: str
):

    if type=='velocity':
        predictor = VelocityPredictor(**kwargs)
    elif type=='pressure':
        predictor = PressurePredictor(**kwargs)

    norm_params = get_norm_params(
        file=norm_file,
        option=type
    )
    predictor.set_normalizer(norm_params)

    return predictor


def get_model(
    type: Literal['velocity', 'pressure'],
    kwargs: dict,
    model_path: str,
    device: str = None
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if type=='velocity':
        predictor = VelocityPredictor(**kwargs)
    elif type=='pressure':
        predictor = PressurePredictor(**kwargs)

    predictor.to(device)

    predictor.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device(device)
        )
    )
    return predictor


def select_input_output(
    data: dict[str, torch.Tensor],
    option: Literal['velocity', 'pressure'],
    device: str
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Select appropriate input & output for ML models.

    Args:
        data: dictionary returned by data loader.
        option: case into consideration.
        device: device on which to put data.
    """
    
    imgs = data['microstructure'].to(device)

    if option=='velocity':

        input = (imgs,)
        targets = data['velocity'].to(device)
        
    elif option=='pressure':
        
        dxyz = data['dxyz'].to(device)
        x_length = dxyz[:, 0]
        
        input = (imgs, x_length)
        targets = data['pressure'].to(device)

    return input, targets


def run_epoch(
    loaders: tuple[DataLoader, DataLoader],
    predictor: Union[VelocityPredictor, PressurePredictor],
    optimizer: optim.Optimizer,
    criterion: Callable[..., torch.Tensor],
    device: str = 'cuda'
):
    """
    Optimize model for 1 epoch over training set, then evaluate over validation set.
    
    Args:
        loaders: data loaders for training and validation sets.
        predictor: ML model.
        optimizer: optimizer for tuning ML model parameters.
        criterion: cost function.
        device: device on which to train ML model.
    """

    train_loader, val_loader = loaders
    num_train_batch = len(train_loader)
    num_val_batch = len(val_loader)

    if isinstance(predictor, VelocityPredictor):
        option = 'velocity'
    elif isinstance(predictor, PressurePredictor):
        option = 'pressure'

    """1. Training Set"""
    predictor.train()

    running_loss = 0
    for i, data in enumerate(train_loader):
        print(f"Training set: batch [{i+1}/{num_train_batch}]")

        input, targets = select_input_output(data, option, device)

        preds = predictor(*input)

        # normalize targets
        targets = predictor.normalizer['output'](targets)

        loss = criterion(output=preds, target=targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / (i+1)


    """2. Validation Set"""
    predictor.eval()

    with torch.no_grad():
        val_loss = 0
        for j, data in enumerate(val_loader):
            print(f"Validation set: batch [{j+1}/{num_val_batch}]")


            input, targets = select_input_output(data, option, device)

            preds = predictor(*input)

            # normalize targets
            targets = predictor.normalizer['output'](targets)

            loss = criterion(output=preds, target=targets)
            val_loss += loss.item()

        avg_val_loss = val_loss / (j+1)

    return (avg_train_loss, avg_val_loss)


def retrieve_model_path(directory_or_url: str, filename: str = 'model.pt') -> str:
    """
    Retrieve path to pre-trained model.
    
    Args:
        directory_or_url: either local directory or URL of the pre-trained model.
        filename: name of the model file.
    """

    if is_url(directory_or_url):
        # Use pre-trained model in repo

        _folder = 'pretrained'
        if not osp.exists(_folder): os.mkdir(_folder)

        # download pre-trained weights
        zip_path = download_data(url=directory_or_url, save_dir=_folder)

        # unzip data
        folder_path = unzip_data(zip_path=zip_path, save_dir=_folder)

        model_path = osp.join(folder_path, filename)

    else:
        # Use trained model in local directory
        model_path = osp.join(directory_or_url, filename)

    return model_path
