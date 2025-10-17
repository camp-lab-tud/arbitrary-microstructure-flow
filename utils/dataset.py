import os
import os.path as osp
import shutil
import json
import requests
import zipfile

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    SubsetRandomSampler,
    random_split
)
from torchvision.transforms.functional import hflip, vflip




class MicroFlowDataset(Dataset):

    """
    Dataset for steady-state velocity flow field in 2D microstructures.
    """

    def __init__(
        self,
        root_dir: str,
        augment: bool = False,
        download: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            root_dir: directory where data is stored.
            augment: whether to augment the dataset by flipping the arrays.
            download: whether to download the dataset if not present in `root_dir`.
        """

        self.root_dir = root_dir
        self.augment = augment

        self.data: dict[str, torch.Tensor] = {}

        self._download_url = 'https://zenodo.org/records/16940478/files/dataset.zip?download=1'
        if download:
            self.download(url=self._download_url)

        self.process()


    def process(self):
        """Load datset."""

        meta_dict = {
            'microstructure': 'domain.pt',
            'velocity': 'U.pt',
            'pressure': 'p.pt',
            'dxyz': 'dxyz.pt',
            'permeability': 'permeability.pt'
        }

        # Read data
        # images have a shape of (samples, channels, height, width)
        _data_x = {}
        for key, val in meta_dict.items():
            file_path = osp.join(self.root_dir, 'x', val)
            dta = torch.load(file_path)
            _data_x[key] = dta

        try:
            # try if there are simulations with flow in y-direction
            _data_y = {}
            for key, val in meta_dict.items():
                file_path = osp.join(self.root_dir, 'y', val)
                dta = torch.load(file_path)

                if key in ['microstructure', 'velocity', 'pressure']:
                    dta = self._rotate_y_field(dta)

                _data_y[key] = dta

            # Concatenate
            for key in meta_dict.keys():
                self.data[key] = torch.cat(
                    (_data_x[key], _data_y[key]),
                    dim=0
                )
            
            print("Loaded simulations with flow in 'x' and 'y' directions.")

        except:
            self.data = _data_x
            print("Loaded simulations with flow in 'x' direction.")

        if self.augment:
            # Flip
            for key in meta_dict.keys():
                if key in ['microstructure', 'pressure']:
                    self.data[key] = torch.cat(
                        (self.data[key], torch.from_numpy(self.data[key].numpy()[:, :, ::-1, :].copy()))
                    )
                elif key == 'velocity':
                    tmp = torch.from_numpy(self.data[key].numpy()[:, :, ::-1, :].copy()) # flip
                    tmp[:, 1, :, :] = - tmp[:, 1, :, :] # switch sign for y-velocity component

                    self.data[key] = torch.cat(
                        (self.data[key], tmp)
                    )
                else:
                    self.data[key] = torch.cat(
                        (self.data[key], self.data[key])
                    )
        
        # save statistics
        self._save_statistics()

    def download(self, url: str):
        """
        Download dataset.
        
        Args:
            url: URL of the dataset.
        """

        # 1. Download dataset
        zip_path = download_data(url=url, save_dir=self.root_dir)
        
        # 2. Unzip data
        folder_path = unzip_data(zip_path=zip_path, save_dir=self.root_dir)
        
        # 3. Move folder
        dest_path = osp.join(self.root_dir, 'raw')
        try:
            shutil.move(folder_path, dest_path)
            print(f"Moved {folder_path} to {dest_path}.")

        except shutil.Error as e:
            print(f"Error during move operation: {e}")
        except FileNotFoundError:
            print(f"Destination path not found. Make sure the parent directory exists.")
    

    def __len__(self):
        num_data = self.data['microstructure'].shape[0]
        return num_data
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor]:

        sample = {
            'microstructure': self.data['microstructure'][idx, :, :, :].float(),
            'velocity': self.data['velocity'][idx, [0,1], :, :].float(),
            'pressure': self.data['pressure'][idx, :, :, :].float(),
            'dxyz': self.data['dxyz'][idx].float(),
            'permeability': self.data['permeability'][idx]
        }
        return sample

    @staticmethod
    def load_dataset(folder: str):
        """
        Load dataset.
        
        Args:
            folder: dataset folder.
        """

        meta_dict = {
            'microstructure': 'domain.pt',
            'velocity': 'U.pt',
            'pressure': 'p.pt',
            'dxyz': 'dxyz.pt',
            'permeability': 'permeability.pt'
        }
        cases_dict = {'x': None, 'y': None}
        
        def load_flow_results(folder) -> dict[str, torch.Tensor]:
            # load data from given folder
            out = {}
            for key, val in meta_dict.items():
                file_path = osp.join(folder, val)
                data = torch.load(file_path)
                out[key] = data
            return out
    
        # Read data for each case
        for case in cases_dict.keys():

            subfolder = osp.join(folder, case)
            if osp.exists(subfolder):
                cases_dict[case] = load_flow_results(subfolder)
        
        return 
    
    @staticmethod
    def augment_dataset(
        data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Augment dataset by flipping arrays.
        
        ``
        """
        pass

    def _save_statistics(self):
        """
        Save dataset statistics.

        """
        log_file = osp.join(self.root_dir, 'statistics.json')

        stats = {
            'U': {
                'max': self.data['velocity'].abs().max().item()
            },
            'p': {
                'max': self.data['pressure'].abs().max().item()
            },
            'dxyz': {
                'max': self.data['dxyz'].abs().max().item()
            },
        }

        # save
        with open(log_file, 'w') as f:
            json.dump(stats, f, indent=0)

    @staticmethod
    def _rotate_y_field(x: torch.Tensor):
        """
        Rotate microstructure, velocity, and pressure fields
        for simulations with flow in the y-direction.

        
        """
        _, num_channels, _, _ = x.shape

        # rotate
        x = torch.rot90(x, k=1, dims=(-2,-1))

        if num_channels != 1:
            # swap channel order
            x = x[:, [1, 0, 2], :, :]

            # change sign of new y-velocity
            x[:, 1, :, :] = - x[:, 1, :, :]
            
        return x



class MicroFlowDataset3D(MicroFlowDataset):

    """
    Dataset for steady-state velocity flow field in slices of a 3D microstructure.
    """

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:

        sample = {
            'microstructure': self.data['microstructure'][idx, :, :, :].float(),
            'velocity': self.data['velocity'][idx, [0,1], :, :].float(),
            'pressure': self.data['pressure'][idx, :, :, :].float(),
            'dxyz': self.data['dxyz'][idx].float(),
            'permeability': self.data['permeability'][0] # there's a single permeability value
        }
        return sample




def get_loader(
    root_dir,
    augment=False,
    train_ratio=0.8,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    seed=2024,
    k_folds: int = None
) -> list[tuple[DataLoader, DataLoader]]:
    """
    Load dataset.

    `root_dir`: directory where data is stored.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else seed

    # Dataset
    dataset = MicroFlowDataset(root_dir, augment=augment)

    # Split data
    if k_folds is None:
        train_size = int(train_ratio*len(dataset))
        lengths = [train_size, len(dataset)-train_size]

        train_set, test_set = random_split(
            dataset,
            lengths,
            generator=generator
        )

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory
        )

        test_loader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory
        )

        out = [(train_loader, test_loader)]

    else:
        kf = KFold(
            n_splits=k_folds,
            shuffle=True,
            random_state=seed
        )

        out = []
        for i, (train_idx, test_idx) in enumerate(kf.split(dataset)):

            train_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(train_idx, generator=generator),
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=pin_memory
            )

            test_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(test_idx, generator=generator),
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=pin_memory
            )
            out.append(
                (train_loader, test_loader)
            )

    return out


def load_micrograph_data(folder: str) -> dict[str, torch.Tensor]:
    """
    Load micrograph data. The data is the one used in this paper:

    Syerko, Elena, et al.
    "Benchmark exercise on image-based permeability determination of engineering textiles: Microscale predictions."
    Composites Part A: Applied Science and Manufacturing 167 (2023): 107397.

    `folder`: The folder contains (.tif) images representing microstructure cross-sections 
    obtained through an X-ray microscope.
    """
    VOXEL_SIZE = 0.521 * 1e-6 # 0.521 microns/voxel

    img_paths = os.listdir(folder)
    # sort paths
    img_paths = sorted(img_paths)
    # full paths
    img_paths = [osp.join(folder, _pth) for _pth in img_paths]

    """1. Microstructure images"""
    img_list = []
    for path in img_paths:

        im = Image.open(path)

        # convert to binary
        im = im.convert('1')

        im = np.array(im) # 2D array

        # invert so that there is 0 in fiber regions, and 1 elsewhere
        im = np.invert(im)

        # 4D tensor (batch, channels, height, width)
        img_tens = torch.from_numpy(im).unsqueeze(0).unsqueeze(0)
        img_list.append(img_tens)

    # concatenate
    microstructure = torch.cat(img_list, dim=0)


    """2. Microstructure dimensions"""

    dx = microstructure.shape[-1] * VOXEL_SIZE
    dy = microstructure.shape[-2] * VOXEL_SIZE
    dz = VOXEL_SIZE

    num_slices = microstructure.shape[0]
    dxyz = torch.tensor([[dx, dy, dz]]).expand(num_slices, -1)


    out = {
        'microstructure_original': microstructure.float(),
        'dxyz': dxyz
    }
    return out



def download_data(
    url: str,
    save_dir: str
):
    """
    Download data from URL.

    Args:
        url: URL of the data.
        save_dir: directory where data is stored.
    """

    if not osp.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    zip_path = osp.join(save_dir, 'dataset.zip')

    if not osp.exists(zip_path):
        
        print(f'Downloading data from {url} ...')
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print('Download completed.')
    else:
        print(f'Zip file already exists at {zip_path}. Skipping download.')

    return zip_path


def unzip_data(zip_path: str, save_dir: str) -> str:
    """
    Extract data from zip file.

    Args:
        zip_path: path to the zip file.
        save_dir: directory where data is extracted to.
    """
    print(f'Extracting data from {zip_path}...')

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # the zip file contains a single folder (with subfolders/files)
        namelist = zip_ref.namelist()
        folder_name = namelist[0].split('/')[0]
        zip_ref.extractall(save_dir)

    folder_path = osp.join(save_dir, folder_name)
    print(f'Data extracted to {folder_path}.')
    return folder_path