import argparse
import os
import os.path as osp
from datetime import datetime

import torch


parser = argparse.ArgumentParser()

parser.add_argument(
    '--name',
    type=str,
    default='unet',
    help='Arbitrary title describing the dataset used or model being trained.'
)
parser.add_argument(
    '--save-dir',
    type=str,
    default='./trained/',
    help='Directory where to save results.'
)
parser.add_argument(
    '--mode',
    type=str,
    default='train',
    choices=['train', 'CV', 'optimize'],
    help='Flag indicating whether to train model, cross-validate model, or perform parameter optimization.'
)


group_dataset = parser.add_argument_group(
    'Dataset Parameters',
    'Parameters for loading dataset.'
)
group_train = parser.add_argument_group(
    'Training Parameters',
    'Parameters related to model training.'
)
group_optim = parser.add_argument_group(
    'Optimization Parameters',
    'Parameters related to model optimization.'
)

"""Dataset"""

group_dataset.add_argument(
    '--root-dir',
    type=str,
    required=True,
    help='Directory for dataset.'
)
group_dataset.add_argument(
    '--batch-size',
    type=int,
    default=10,
    help='Batch size.'
)
group_dataset.add_argument(
    '--augment',
    type=bool,
    default=False,
    help='Whether to augment dataset (e.g., by flipping images).'
)
group_dataset.add_argument(
    '--shuffle',
    type=bool,
    default=False,
    help='Whether to shuffle data during training.'
)
group_dataset.add_argument(
    '--k-folds',
    type=int,
    default=5,
    help='Number of folds when splitting dataset.'
)


"""Training"""

group_train.add_argument(
    '--device',
    type=str,
    default=None,
    help='Device (e.g., cpu, cuda) on which to train neural network.'
)
group_train.add_argument(
    '--learning-rate',
    type=float,
    default=1e-4,
    help='Learning rate.'
)
group_train.add_argument(
    '--scheduler-flag',
    type=bool,
    default=False,
    help='Whether to use learning rate scheduler.'
)
group_train.add_argument(
    '--scheduler-gamma',
    type=float,
    default=0.95499, # 0.977
    help='If `--scheduler-flag` is True, multiplicative factor of learning rate decay (in ExponentialLR).'
)
group_train.add_argument(
    '--num-epochs',
    type=int,
    default=100,
    help='Number of epochs.'
)
group_train.add_argument(
    '--cost-function',
    type=str,
    default='normalized_mae_loss',
    choices=['normalized_mae_loss', 'mae_loss'],
    help='Cost function for training.'
)

group_train.add_argument(
    '--predictor-type',
    type=str,
    default='velocity',
    choices=['velocity', 'pressure'],
    help='Type of ML predictor (for the velocity or pressure field)'
)
group_train.add_argument(
    '--model-name',
    type=str,
    default='UNet',
    help='Neural network model'
)
group_train.add_argument(
    '--in-channels',
    type=int,
    required=True,
    help='Number of channels in input data.'
)
group_train.add_argument(
    '--out-channels',
    type=int,
    required=True,
    help='Number of channels in output data.'
)
group_train.add_argument(
    '--features',
    type=int,
    nargs='+',
    default=[64, 128, 256, 512],
    help='Number of channels at each (depth) level in the U-Net architecture.'
)
group_train.add_argument(
    '--kernel-size',
    type=int,
    default=3,
    help='Kernel size for convolutional layers.'
)
group_train.add_argument(
    '--padding-mode',
    type=str,
    default='reflect',
    help='Type of padding for convolutional layers.'
)
group_train.add_argument(
    '--activation',
    type=str,
    default='silu',
    choices=['silu', 'relu', 'leakyrelu','softplus'],
    help='Activation functions inside neural network.'
)
group_train.add_argument(
    '--final-activation',
    type=str,
    default=None,
    choices=['silu', 'relu', 'leakyrelu','softplus'],
    help='Activation function before ouput.'
)
group_train.add_argument(
    '--attention',
    type=str,
    default='',
    help='Expression determining the use of attention in U-Net model (e.g., "4..1"). For details, see model documentation.'
)
group_train.add_argument(
    '--distance-transform',
    type=bool,
    default=True,
    help='Whether to use distance transform for input image.'
)


"""Optimization Parameters"""

group_optim.add_argument(
    '--n-trials',
    type=int,
    default=100,
    help='Number of trials for optimization algorithm.'
)
group_optim.add_argument(
    '--range-batch-size',
    type=int,
    default=[10, 40],
    nargs=2,
    help='Range for batch size.'
)
group_optim.add_argument(
    '--range-kernel-size',
    type=int,
    default=[3, 7],
    nargs=2,
    help='Range for kernel size.'
)
group_optim.add_argument(
    '--range-level',
    type=int,
    default=[1, 7],
    nargs=2,
    help='Number of levels in U-Net.'
)
group_optim.add_argument(
    '--top-bottom',
    type=bool,
    default=True,
    nargs=2,
    help='If "True", define channel sizes from top-to-bottom. If "False", then proceed from bottom-to-top.'
)
group_optim.add_argument(
    '--top-feature-channels',
    type=int,
    default=32,
    help='Number of feature channels at top-level of U-Net.'
)
group_optim.add_argument(
    '--bottom-feature-channels',
    type=int,
    default=2048,
    help='Number of feature channels at bottom-level of U-Net.'
)
group_optim.add_argument(
    '--range-learning-rate',
    type=float,
    default=[1e-7, 1e-3],
    nargs=2,
    help='Range for learning rate.'
)



def process_args(args: argparse.Namespace):
    """
    Process command line arguments into a dictionary.

    `args`: command line arguments.
    """

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    param_dict = {
        'name': args.name,
        'mode': args.mode,
        'save_dir': args.save_dir,

        'dataset': {
            'root_dir': args.root_dir,
            'batch_size': args.batch_size,
            'augment': args.augment,
            'shuffle': args.shuffle,
            'k_folds': args.k_folds
        },
        'training': {
            'device': args.device,
            'learning_rate': args.learning_rate,
            'scheduler': {
                'flag': args.scheduler_flag,
                'gamma': args.scheduler_gamma,
            },
            'num_epochs': args.num_epochs,
            'cost_function': args.cost_function,
            'predictor_type': args.predictor_type,
            'predictor': {
                'model_name':args.model_name,
                'model_kwargs': {
                    'in_channels': args.in_channels,
                    'out_channels': args.out_channels,
                    'features': args.features,
                    'kernel_size': args.kernel_size,
                    'padding_mode': args.padding_mode,
                    'activation': args.activation,
                    'final_activation': args.final_activation,
                    'attention': args.attention
                },
                'distance_transform': args.distance_transform
            }
        },
        'optimization': {
            'n_trials': args.n_trials,
            'range_batch_size': args.range_batch_size,
            'range_kernel_size': args.range_kernel_size,
            'range_level': args.range_level,
            'range_learning_rate': args.range_learning_rate,
            'top_bottom': args.top_bottom,
            'top_feature_channels': args.top_feature_channels,
            'bottom_feature_channels': args.bottom_feature_channels
        }
    }
    return param_dict


def make_log_folder(param_dict: dict):
    """
    Create folder where to results.

    `param_dict`: dictionary with parameters.
    """

    name = param_dict['name']
    save_dir = param_dict['save_dir']

    dataset_kwargs = param_dict['dataset']
    train_kwargs = param_dict['training']

    batch_size = dataset_kwargs['batch_size']

    learning_rate = train_kwargs['learning_rate']
    num_epochs = train_kwargs['num_epochs']

    predictor_type = train_kwargs['predictor_type']
    predictor_kwargs = train_kwargs['predictor']
    in_channels = predictor_kwargs['model_kwargs']['in_channels']
    out_channels = predictor_kwargs['model_kwargs']['out_channels']
    features = predictor_kwargs['model_kwargs']['features']
    kernel_size = predictor_kwargs['model_kwargs']['kernel_size']
    padding_mode = predictor_kwargs['model_kwargs']['padding_mode']
    attention = predictor_kwargs['model_kwargs']['attention']


    # Create log folder
    time_stamp = datetime.now().strftime("%Y%m%d")
    
    descr_str = f'in-{in_channels}-out-{out_channels}-' \
        f'f-{len(features)}-k-{kernel_size}-p-{padding_mode}-a-{attention}-' \
        f'b-{batch_size}-lr-{learning_rate:.2e}-ep-{num_epochs}'
    
    sub_dir = time_stamp + f'_{name}_{predictor_type}_' + descr_str
    log_folder = osp.join(save_dir, sub_dir)

    if not osp.exists(log_folder):
        os.makedirs(log_folder)

    return log_folder