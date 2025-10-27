import os.path as osp
import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from src.predictor import (
    Predictor,
    VelocityPredictor,
    PressurePredictor
)
from src.apps import SlidingWindow
from utils.dataset import (
    load_VirtualPermeabilityBenchmark,
    resize_image,
    BlindDataset
)
from src.physics import (
    get_flow_rate,
    get_average_pressure
)


parser = argparse.ArgumentParser()
parser.add_argument('--micrograph-dir', type=str, required=True, default='../../data/unet/micrograph/Benchmark package/Stack of segmented images', help='Micrograph data directory.')
parser.add_argument('--velocity-model', type=str, default='https://zenodo.org/records/17306446/files/velocity_model_base.zip?download=1', help='Folder with or URL to velocity model.')
parser.add_argument('--pressure-model', type=str, default='https://zenodo.org/records/17306446/files/pressure_model_base.zip?download=1', help='Folder with or URL to pressure model.')
parser.add_argument('--window-step', type=int, default=256, help='Window step to use with sliding window algorithm.')
parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., cpu, cuda).')
args = parser.parse_args()

if args.device is None:
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_HEIGHT = 256


def make_loader(folder: str):

    data = load_VirtualPermeabilityBenchmark(folder)

    # resize microstructures
    data['microstructure'] = resize_image(
        img=data['microstructure_original'],
        target_height=TARGET_HEIGHT
    )
    
    # dataset
    dataset = BlindDataset(data)

    # data loader
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False
    )
    return loader


@torch.no_grad()
def evaluate_model(
    velocity_predictor: VelocityPredictor,
    pressure_predictor: PressurePredictor,
    data: dict[str, torch.Tensor],
    device: str
) -> dict[str, torch.Tensor]:

    resin_density = 1250

    imgs = data['microstructure'].to(device)
    dxyz = data['dxyz'].to(device)
    dx = dxyz[:, 0]

    slw = SlidingWindow(
        velocity_model=velocity_predictor,
        pressure_model=pressure_predictor,
        step_size=args.window_step
    )
    vel_result = slw.predict_velocity(img=imgs)
    pres_result = slw.predict_pressure(img=imgs, x_length=dx)

    velocity = vel_result['prediction']
    pressure = pres_result['prediction'] * resin_density
    

    # flow rate
    dy = dxyz[:, 1]
    dz = dxyz[:, 2]
    cross_section_area = dy * dz
    flow_rate = get_flow_rate(imgs, velocity.unsqueeze(0), cross_section_area)

    # average pressure
    avg_pressure = get_average_pressure(imgs.squeeze(0), pressure)

    # permeability
    kval = compute_permeability(flow_rate[:, 0], avg_pressure[:, 0], dx, cross_section_area)

    out = {
        'microstructure_original': data['microstructure_original'].to(device),

        'microstructure': imgs, # shape: (1, channels, height, width)
        'velocity': velocity.unsqueeze(0),
        'pressure': pressure.unsqueeze(0),
        'dxyz': dxyz,

        'flow_rate': flow_rate,
        'average_pressure': avg_pressure,
        'permeability': kval
    }
    return out


def compute_permeability(
    flow_rate: torch.Tensor,
    avg_pressure: torch.Tensor,
    domain_length: torch.Tensor,
    cross_section_area: torch.Tensor
):
    for val in (flow_rate, avg_pressure, domain_length, cross_section_area):
        assert val.dim() == 1, 'Inputs should be 1D arrays'

    mu = 0.5
    # permeability (Darcy's law)
    out = flow_rate * mu * domain_length / (cross_section_area * avg_pressure)
    
    return out

def main():

    """1. Load data"""

    loader = make_loader(args.micrograph_dir)


    """2. Load models"""
    
    # velocity predictor
    velocity_predictor = Predictor.from_directory_or_url(
        directory_or_url=args.velocity_model,
        device=args.device
    )
    # pressure predictor
    pressure_predictor = Predictor.from_directory_or_url(
        directory_or_url=args.pressure_model,
        device=args.device
    )
    assert isinstance(velocity_predictor, VelocityPredictor), 'Loaded model is not a VelocityPredictor.'
    assert isinstance(pressure_predictor, PressurePredictor), 'Loaded model is not a PressurePredictor.'
    velocity_predictor.eval()
    pressure_predictor.eval()


    """3. Evaluation"""

    start = time.time()
    result_list: list[dict] = []

    for i, data in enumerate(loader):
        print(f'Batch {i}...')

        result = evaluate_model(
            velocity_predictor=velocity_predictor,
            pressure_predictor=pressure_predictor,
            data=data,
            device=args.device
        )
        result_list.append(result)
    end = time.time()
    duration = end - start; print(f'Time: {duration:.2f} seconds.')

    del velocity_predictor, pressure_predictor # to free up memory

    results: dict[str, list[torch.Tensor]] = {}
    for key in result_list[0].keys():
        results[key] = [val[key] for val in result_list]
    
    for key, val in results.items():
        results[key] = torch.cat(val)
    results['run_time'] = duration
    
    # save
    save_dir = Path(args.micrograph_dir).parent
    torch.save(results, osp.join(save_dir, 'predictions.pt'))
    

if __name__=='__main__':

    main()