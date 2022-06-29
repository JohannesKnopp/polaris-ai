import argparse
import sys
import json
import time
import math
import os

import numpy as np
from collections import defaultdict
from matplotlib import pyplot

from gcd_data_manipulation import ClusterDataset
from gcd_data_manipulation import prepare_data
from shared_workspace_module import SharedWorkspaceModule

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error


def str2bool(v):
    """Method to map string to bool for argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_val_history(history, output_path, name):
    # plot history
    pyplot.clf()
    pyplot.plot(history['avg_loss'], label='train')
    pyplot.plot(history['avg_val_loss'], label='test')
    pyplot.legend()
    # pyplot.ylim((0.00, 0.27))
    pyplot.savefig(os.path.join(output_path, 'gcd_%s_val-loss.png' % name))


def rmse_percentage(outputs: torch.Tensor, targets: torch.Tensor):
    return math.sqrt(mean_squared_error(targets, outputs)) / targets.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer + Shared Workspace predictor for job efficiency')

    parser.add_argument('--job_id', default=3418339, type=int, help='ID of the job considered for the model generation')
    parser.add_argument('--epochs', default=100, type=int, help='num of epochs to train')
    parser.add_argument('--ffn_dim', type=int, default=512)  # neurons in one feed forward layer
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size to use')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')

    parser.add_argument('--h_dim', type=int, default=64) # this is shape of the input data TODO doublecheck
    parser.add_argument('--num_layers', default=12, type=int, help='num of layers')
    parser.add_argument('--num_heads', default=4, type=int, help='num of heads in Multi Head attention layer')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout')
    parser.add_argument('--shared_memory_attention', type=str2bool, default=False)
    parser.add_argument('--share_vanilla_parameters', type=str2bool, default=False)
    parser.add_argument('--use_topk', type=str2bool, default=False)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--mem_slots', type=int, default=4)
    # TODO currently omitted null_attention and num_steps -> not necessary afaik

    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment. Necessary to save results')

    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    # parser.add_argument('--model', default="default", type=str, choices=('default', 'functional'), help='type of transformer to use')
    # parser.add_argument('--version', default=0, type=int, help='version for shared transformer-- 0 or 1')
    # parser.add_argument('--num_templates', default=12, type=int, help='num of templates for shared transformer')
    # parser.add_argument('--patch_size', default=4, type=int, help='patch_size for transformer')
    # parser.add_argument('--null_attention', type=str2bool, default=False)
    # parser.add_argument('--num_gru_schemas', type=int, default=1)
    # parser.add_argument('--num_attention_schemas', type=int, default=1)
    # parser.add_argument('--schema_specific', type=str2bool, default=False)
    # parser.add_argument('--num_eval_layers', type=int, default=1)
    # parser.add_argument('--num_digits_for_mnist', type=int, default=3)
    # parser.add_argument('--seed', type=int, default=0)

    with open('columns_selection.json') as f:
        columns_selection = json.load(f)

    args = parser.parse_args(sys.argv[1:])
    epochs = args.epochs
    ffn_dim = args.ffn_dim
    batch_size = args.batch_size
    exp_name = args.exp_name
    JOB_ID = args.job_id

    input_path = f'../data/task-usage_job-ID-{JOB_ID}_total.csv'
    figures_path = '../experiments_result/figures_GWT'
    results_path = 'results'
    model_path = '../models'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    columns_to_consider = columns_selection['GWT_efficiency_1']  # TODO add argparse

    # transform = torchvision.transforms.ToTensor()

    data = prepare_data(input_path, columns_to_consider, aggr_type='mean')

    train_data = ClusterDataset(data, training=True, split_percentage=0.7)
    val_data = ClusterDataset(data, training=False, split_percentage=0.7)

    train_data.values.to(device)
    val_data.values.to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # results = defaultdict(list)  # TODO

    model = SharedWorkspaceModule(
        h_dim=args.h_dim,
        ffn_dim=args.ffn_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        shared_memory_attention=args.shared_memory_attention,
        share_vanilla_parameters=args.share_vanilla_parameters,
        use_topk=args.use_topk,
        topk=args.topk,
        mem_slots=args.mem_slots
    )  # TODO
    model.cuda()

    criterion = nn.L1Loss(reduction='sum').cuda()  # TODO different loss function ?
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) # relevant ?
    # scheduler = torch.optim.lr_scheduler TODO add scheduler
    history_loss = defaultdict(list)

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1} / {epochs}')
        start_time = time.time()
        model.train()
        total_loss = 0.
        # train_predictions =

        # mini_batch_size = 10
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # reshape target to torch.Size([batch_size, 1]) to match the output of the model
            targets = targets.reshape((targets.shape[0], 1))
            # move tensors to cuda
            inputs, targets = inputs.to(device), targets.to(device)

            # zero the gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model(inputs)
            # TODO pre_loss_fn ?
            # compute loss
            loss = criterion(outputs, targets)  # TODO double check squeeze is ok
            # back propagation
            loss.backward()
            # gradient descent
            optimizer.step()

            total_loss += loss.item()
            # if batch_idx % mini_batch_size == 9:
            #     last_loss = current_loss / mini_batch_size
            #     print(f'Loss after mini-batch {batch_idx + 1 : 3d}: {last_loss : .5f}')
            #     current_loss = 0.

        average_loss = total_loss / len(train_data)
        history_loss['avg_loss'].append(average_loss)

        print(f'Average Training Loss: {average_loss: .5f}')

        model.eval()

        validation_loss = 0.
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            targets = targets.reshape((targets.shape[0], 1))
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            validation_loss += loss.item()

        average_validation_loss = validation_loss / len(val_data)
        history_loss['avg_val_loss'].append(average_validation_loss)
        print(f'Validation loss: {average_validation_loss: .5f}')

    plot_val_history(history_loss, figures_path, exp_name)

    # TODO add RMSE / RMSE% / EPOCH_NUM / ? ... to state_dict()
    # TODO better naming of model save
    # torch.save(model.state_dict(), f'../models/gwt_models/gwt_model_{exp_name}.pth')

    model_state_dict = {
        'state_dict': model.state_dict(),
        'model_args': args,
        'epoch': epochs,
        'loss': history_loss['avg_loss'][-1]
    }

    torch.save(model_state_dict, f'../models/gwt_models/gwt_model_{exp_name}.pth')
