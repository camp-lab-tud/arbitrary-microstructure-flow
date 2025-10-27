from typing import Union
import json
import os
import os.path as osp
import argparse

import torch

from src.predictor import VelocityPredictor, PressurePredictor
from src.helper import get_model, retrieve_model_path
from src.unet.metrics import normalized_mae_loss
from utils.dataset import get_loader


parser = argparse.ArgumentParser()
parser.add_argument('--directory-or-url', type=str, required=True, help='Local directory or URL with trained model file.')
parser.add_argument('--root-dir', type=str, default=None, help='Dataset directory.')
parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid'], help='Dataset split to use.')
parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., cpu, cuda).')
parser.add_argument('--save-dir', type=str, default=None, help='Where to save evaluation results.')
args = parser.parse_args()

if args.device is None:
    args.device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate_model(
    predictor: Union[VelocityPredictor, PressurePredictor],
    data: dict[str, torch.Tensor],
    device: str
) -> dict[str, torch.Tensor]:

    imgs = data['microstructure'].to(device)
    dxyz = data['dxyz'].to(device)

    if isinstance(predictor, VelocityPredictor):
        input = (imgs,)
        targets = data['velocity'].to(device)

    elif isinstance(predictor, PressurePredictor):
        x_length = dxyz[:, 0]
        
        input = (imgs, x_length)
        targets = data['pressure'].to(device)

    preds = predictor.predict(*input)

    loss = normalized_mae_loss(preds, targets, reduce=False)
    print('loss: ', loss.mean().item())

    out = {
        'microstructure': imgs,
        'dxyz': dxyz,
        'prediction': preds,
        'target': targets,
        'loss': loss
    }
    return out


def main():

    # get model path from local directory or URL
    model_path = retrieve_model_path(
        directory_or_url=args.directory_or_url,
        filename='model.pt'
    )

    # read log file
    log_file = osp.join(osp.dirname(model_path), 'log.json')
    with open(log_file) as fp:
        log_data = json.load(fp)


    param_dict = log_data['params']

    data_kwargs = param_dict['dataset']
    if args.root_dir is not None:
        data_kwargs['root_dir'] = args.root_dir
    predictor_type = param_dict['training']['predictor_type']
    predictor_kwargs = param_dict['training']['predictor']


    """Dataset"""
    train_loader, val_loader = get_loader(**data_kwargs)[0]


    """Model"""    
    predictor = get_model(
        type=predictor_type,
        kwargs=predictor_kwargs,
        model_path=model_path,
        device=args.device
    )
    predictor.eval()


    """Evaluation"""
    if args.split == 'train':
        loader = train_loader
    elif args.split =='valid':
        loader = val_loader

    all_results = {
        'microstructure': [],
        'dxyz': [],
        'prediction': [],
        'target': [],
        'loss': []
    }
    for i, data in enumerate(loader):

        print(f'Batch {i}...')

        result = evaluate_model(
            predictor=predictor,
            data=data,
            device=args.device
        )
        
        for key in all_results.keys():
            all_results[key].append( result[key].cpu() )

    for key in all_results.keys():
        all_results[key] = torch.cat( all_results[key] )

    # save
    if args.save_dir is None:
        args.save_dir = osp.dirname(model_path)

    subfolder = osp.join(args.save_dir, 'evaluation')
    if not osp.exists(subfolder):
        os.mkdir(subfolder)
    
    torch.save(
        all_results,
        osp.join(subfolder, f'results_{predictor_type}_{args.split}.pt')
    )


if __name__=='__main__':

    main()