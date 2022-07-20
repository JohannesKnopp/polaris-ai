import gcd_single_job_multivariate_prediction
from shared_workspace_module import SharedWorkspaceModule
from gcd_data_manipulation import ClusterDataset
from gcd_data_manipulation import prepare_data

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import numpy as np

from ray import tune
from ray.tune import CLIReporter
from ray.tune import ASHAScheduler
from functools import partial


def train_gcd(config, checkpoint_dir=None):
    model = SharedWorkspaceModule(
        config['h_dim'],
        config['ffn_dim'],
        config['num_layers'],
        config['num_heads'],
        config['dropout'],
        config['shared_memory_attention'],
        config['share_vanilla_parameters'],
        config['use_topk'],
        config['topk'],
        config['mem_slots'],
        config['num_targets']
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, 'checkpoint'))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    with open('columns_selection.json') as f:
        columns_to_consider = json.load(f)['GWT_efficiency_1']

    data = prepare_data('../data/task-usage_job-ID-3418339_total.csv', columns_to_consider, targets=[0], sliding_window=1)

    train_data = ClusterDataset(data, num_targets=1, training=True, split_percentage=0.7)
    val_data = ClusterDataset(data, num_targets=1, training=False, split_percentage=0.7)

    train_data.values.cuda()
    val_data.values.cuda()

    train_loader = DataLoader(train_data, batch_size=config['batch_size'])
    val_loader = DataLoader(val_data, batch_size=config['batch_size'])

    model.cuda()

    criterion = nn.L1Loss(reduction='sum').cuda()

    for epoch in range(100):
        print(f'Epoch {epoch}/100:')
        model.train()
        total_loss = 0.

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_data)
        print(f'Average Training Loss: {average_loss: .5f}')

        model.eval()

        validation_loss = 0.

        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            validation_loss += loss.item()

        average_validation_loss = validation_loss / len(val_data)
        print(f'Validation Loss: {average_validation_loss: .5f}')

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'checkpoint')
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=average_validation_loss)  # TODO add rmse_percentage


if __name__ == '__main__':
    MAX_NUM_EPOCHS = 100

    config = {
        'h_dim': tune.sample_from(lambda _: 2**np.random.randint(3, 7)),
        'ffn_dim': tune.sample_from(lambda _: 2**np.random.randint(3, 9)),
        'num_layers': tune.sample_from(lambda _: np.random.randint(3, 20)),
        'num_heads': tune.sample_from(lambda _: np.random.randint(3, 8)),
        'dropout': 0,
        'shared_memory_attention': True,
        'share_vanilla_parameters': True,
        'use_topk': True,
        'topk': tune.sample_from(lambda _: np.random.randint(3, 8)),
        'mem_slots': tune.sample_from(lambda _: np.random.randint(config['topk'], 10)),
        'num_targets': 1,
        'batch_size': tune.sample_from(lambda _: 8*np.random.randint(2, 16)), # values between 16 and 120 in increments of 8
        'lr': tune.loguniform(1e-4, 1e-1)
    }

    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=MAX_NUM_EPOCHS,
        grace_period=5,
    )

    reporter = CLIReporter(
        parameter_columns=['h_dim', 'ffn_dim', 'num_layers', 'num_heads', 'topk', 'mem_slots', 'batch_size', 'lr'],
        metric_columns=['loss']
    )

    result = tune.run(
        partial(train_gcd, checkpoint_dir='tuning_checkpoint'),
        config=config,
        num_samples=1000,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f'Best trial config: {best_trial.config}')
    print(f'Best trial final validation loss: {best_trial.last_result["loss"]}')

    # TODO test set run