import time
import json
import os.path as osp
from sqlalchemy import create_engine

import torch
import torch.optim as optim
import optuna
from optuna.trial import Trial

# Local
from utils.dataset import get_loader
from src.helper import set_model, run_epoch
from src.unet.metrics import cost_function

from config import parser, process_args, make_log_folder


args = parser.parse_args()


def train(
    train_loader,
    test_loader,
    trial: Trial = None
):
    
    param_dict = process_args(args)
    log_dict = {
        'params': param_dict,
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'time': [],
        'learning_rate_history': []
    }
    log_folder = make_log_folder(param_dict)

    root_dir = param_dict['dataset']['root_dir']
    train_dict = param_dict['training']

    device = train_dict['device']
    learning_rate = train_dict['learning_rate']
    scheduler_kwargs = train_dict['scheduler']
    num_epochs = train_dict['num_epochs']
    cost_function_name = train_dict['cost_function']
    predictor_type = train_dict['predictor_type']
    predictor_kwargs = train_dict['predictor']
    

    # Model
    predictor = set_model(
        type=predictor_type,
        kwargs=predictor_kwargs,
        norm_file=osp.join(root_dir, 'statistics.json')
    )
    predictor.to(device)

    optimizer = optim.Adam(
        predictor.parameters(),
        lr=learning_rate
    )

    scheduler = None
    if scheduler_kwargs['flag']:
        gamma = scheduler_kwargs['gamma']
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    criterion = cost_function(cost_function_name)

    best_loss = float('inf')
    for epoch in range(num_epochs):

        current_lr = optimizer.param_groups[0]['lr']

        # run epoch
        start_time = time.time()
        avg_train_loss, avg_val_loss = run_epoch(
            loaders=(train_loader, test_loader),
            predictor=predictor,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        dtime = time.time() - start_time

        # log
        log_dict['epoch'].append(epoch)
        log_dict['time'].append(dtime)
        log_dict['train_loss'].append(avg_train_loss)
        log_dict['val_loss'].append(avg_val_loss)
        log_dict['learning_rate_history'].append(current_lr)

        # Save
        model_path = osp.join(log_folder, 'model.pt')
        best_model_path = osp.join(log_folder, 'best_model.pt')
        log_path = osp.join(log_folder, 'log.json')

        torch.save(predictor.state_dict(), model_path)
        if avg_val_loss < best_loss:
            torch.save(predictor.state_dict(), best_model_path)
            best_loss = avg_val_loss

        with open(log_path, 'w') as f:
            json.dump(log_dict, f, indent=4)

        print(f"Epoch {epoch}: train_loss={avg_train_loss} | val_loss={avg_val_loss} | time={dtime:.2f} s")
        if scheduler is not None: scheduler.step()


        if trial is not None:
            trial.report(avg_val_loss, epoch)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
    return avg_train_loss, avg_val_loss


def objective(trial: Trial):
    """Objective function for hyper-parameter tuning."""

    # sample hyper-parameters
    args.batch_size = trial.suggest_int(
        "batch_size",
        args.range_batch_size[0],
        args.range_batch_size[1]
    )
    args.kernel_size = trial.suggest_int(
        "kernel_size",
        args.range_kernel_size[0],
        args.range_kernel_size[1],
        step=2
    )
    levels = trial.suggest_int(
        "levels",
        args.range_level[0],
        args.range_level[1]
    )
    factors = [2**val for val in range(levels)]
    if args.top_bottom:
        args.features = [args.top_feature_channels * val for val in factors]
    else:
        args.features = [int(args.bottom_feature_channels / val) for val in reversed(factors)]

    args.learning_rate = trial.suggest_float(
        "learning_rate",
        args.range_learning_rate[0],
        args.range_learning_rate[1],
        log=True
    )
    
    # load data
    train_loader, test_loader = get_loader(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        augment=args.augment,
        k_folds=args.k_folds,
        num_workers=1
    )[0]

    # train
    _, val_loss = train(train_loader, test_loader, trial)

    return val_loss


if __name__=='__main__':

    if args.mode == 'train':

        # load data
        train_loader, test_loader = get_loader(
            root_dir=args.root_dir,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            augment=args.augment,
            k_folds=args.k_folds,
            num_workers=1
        )[0]

        # train
        train(train_loader, test_loader)

    elif args.mode == 'CV':
        # Cross-Validation

        # load data
        data_folds = get_loader(
            root_dir=args.root_dir,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            augment=args.augment,
            k_folds=args.k_folds,
            num_workers=1
        )

        # train
        for i, (train_loader, test_loader) in enumerate(data_folds):
            print(f'Cross-Validation [{i+1}/{args.k_folds}]')

            args.name = f'kfold-{i+1}.{args.k_folds}'

            train(train_loader, test_loader)


    elif args.mode == 'optimize':

        # Create SQL engine
        db_path = osp.abspath(
            osp.join(args.save_dir, f'study.db')
        )
        url = f"sqlite:////{db_path}"
        engine = create_engine(url)

        # Set up study
        study = optuna.create_study(
            direction='minimize',
            study_name=args.name,
            storage=url
        )
        study.optimize(objective, n_trials=args.n_trials)

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print("Study statistics:")
        print("\t Number of finished trials: ", len(study.trials))
        print("\t Number of pruned trials: ", len(pruned_trials))
        print("\t Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial
        print("\t Value: ", trial.value)

        print("\t Params:")
        for key, value in trial.params.items():
            print(f"\t {key}: {value}")
