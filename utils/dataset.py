import os
import os.path as osp
import json

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
        augment: bool = False
    ):
        self.root_dir = root_dir
        self.augment = augment

        self.data: dict[str, torch.Tensor] = {}

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
        
        `folder`: dataset folder.
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
        log_file = osp.join(self.root_dir, 'statistics.json')
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

    def __getitem__(self, idx):

        img  = self.input[idx]
        U = self.target_U[idx]
        p = self.target_p[idx]

        dxdydz = self.dxyz[idx]

        # there's a single value,
        # representing the permeability of the 3D microstructure
        k = self.permeab[0]

        sample = {
            'microstructure': img,
            'velocity': U,
            'pressure': p,
            'dxyz': dxdydz,
            'permeability': k

        }
        if self.transform:
            sample = self.transform(sample)

        return sample



class DatasetTransform:
    """
    Normalize velocity, pressure, and dimension values in dataset.
    """

    def __init__(self, input_var: str | dict) -> None:


        if isinstance(input_var, str):
            # the input is directory of dataset,
            # compute statistics
            root_dir = input_var
            
            # velocity, pressure, and dimensions
            target_U: torch.Tensor = torch.load(osp.join(root_dir, 'x', 'U.pt'))
            # target_U_y: torch.Tensor = torch.load(osp.join(root_dir, 'y', 'U.pt'))
            target_p: torch.Tensor = torch.load(osp.join(root_dir, 'x', 'p.pt'))
            dxyz: torch.Tensor = torch.load(osp.join(root_dir, 'x', 'dxyz.pt'))

            # target_U = torch.cat((target_U_x, target_U_y[:, [1,0,2]]))
            self._max_U = target_U.abs().max().item()
            self._max_p = target_p.max().item()
            self._max_d = dxyz.max().item()

            # save statistics
            self._params = {
                'U': {'max': self._max_U},
                'p': {'max': self._max_p},
                'd': {'max': self._max_d},
            }

            # write
            log_file = osp.join(root_dir, 'statistics.json')
            with open(log_file, 'w') as f:
                json.dump(self._params, f, indent=0)

        elif isinstance(input_var, dict):
            
            self._params = input_var

            # Directly use statistics that have already been computed
            self._max_U = self._params['U']['max']
            self._max_p = self._params['p']['max']
            self._max_d = self._params['d']['max']

        print(f'Statistics: {self._params}')

    def __call__(
        self,
        data: dict[str, torch.Tensor]
    ):

        velocity = data['velocity']
        pressure = data['pressure']
        dxyz = data['dxyz']

        # transform
        velocity_new = self.transform_U(velocity)
        pressure_new = self.transform_p(pressure)
        dxyz_new = self.transform_d(dxyz)

        data['velocity'] = velocity_new
        data['pressure'] = pressure_new
        data['dxyz'] = dxyz_new

        return data

    def inverse_transform(
        self,
        data: dict[str, torch.Tensor]
    ):

        velocity = data['velocity']
        pressure = data['pressure']
        dxyz = data['dxyz']

        # inverse-transform
        velocity_new = self.inverse_transform_U(velocity)
        pressure_new = self.inverse_transform_p(pressure)
        dxyz_new = self.inverse_transform_d(dxyz)

        data['velocity'] = velocity_new
        data['pressure'] = pressure_new
        data['dxyz'] = dxyz_new

        return data

    def transform_U(self, data: torch.Tensor):
        """Transform velocity"""

        # transform
        data = data / self._max_U

        return data
    
    def transform_p(self, data: torch.Tensor):
        """Transform pressure"""

        # transform
        data = data / self._max_p

        return data
    
    def transform_d(self, data: torch.Tensor):
        """Transform dimension"""

        # transform
        data = data / self._max_d

        return data
    
    def inverse_transform_U(self, data: torch.Tensor):
        """Inverse-transform velocity"""

        # inverse-transform
        data = data * self._max_U

        return data
    
    def inverse_transform_p(self, data: torch.Tensor):
        """Inverse-transform pressure"""

        # inverse-transform
        data = data * self._max_p

        return data

    def inverse_transform_d(self, data: torch.Tensor):
        """Inverse-transform dimension"""

        # inverse-transform
        data = data * self._max_d

        return data


class LogTransform:

    def __init__(self):
        pass

    def __call__(self, data: torch.Tensor):

        min_val = data.min()

        # shift so that all values are positive
        data_tr = data - min_val

        # transform
        out = torch.log10( 1 + data_tr)

        return out

    def inverse_transform(
        data: torch.Tensor,
        min_val: float
    ):
        # apply log
        data_tr = 10**(data)

        # shift
        out = data_tr - 1 + min_val

        return out 


def get_loader(
    root_dir,
    augment=False,
    transform=None,
    train_ratio=0.8,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    seed=2024,
    k_folds: int = None
) -> list[tuple]:
    """
    Load dataset.

    `root_dir`: directory where data is stored.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else seed

    # Dataset
    dataset = MicroFlowDataset(
        root_dir,
        augment=augment,
        # transform=transform
    )

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
