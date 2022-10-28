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
from autoencoder import Autoencoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from sklearn.metrics import mean_squared_error
# from torchmetrics import MeanSquaredError

USE_AE = False

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
    pyplot.plot(history['avg_val_loss'], label='validation')
    pyplot.legend()
    # pyplot.ylim((0.00, 0.27))
    pyplot.savefig(os.path.join(output_path, 'gcd_%s_val-loss.png' % name))


def rmse(yhat, y):
    # yhat = outputs.numpy()
    # y = targets.numpy()
    return torch.sqrt(torch.mean(torch.square(y - yhat)))

def rmspe(yhat, y):
    return rmse(yhat, y) / torch.mean(y)

def rmsse(yhat, y):
    e = y - yhat
    m = 1 / (len(y) - 1)
    s = (y - torch.roll(y, 1))[1:]
    t = torch.sum(abs(s))
    #print(t, m, e)
    return torch.sqrt(torch.mean((e / (m * t))**2))

def lag(y):
    yhat = torch.roll(y, 1)[1:]
    y = y[1:]
    return rmse(yhat, y)


def msse_loss(prediction, target):
    e = target - prediction
    m = 1 / (len(target) - 1)
    s = (target - torch.roll(target, 1))[1:]
    t = torch.sum(abs(s))
    return torch.mean(torch.sum((e / (m * t))**2))


def train_one_epoch():
    #start_time = time.time()
    model.train()
    total_loss = 0.
    # train_predictions =

    # optimizer.zero_grad()
    # mini_batch_size = 10
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # reshape target to torch.Size([batch_size, 1]) to match the output of the model
        # move tensors to cuda
        inputs, targets = inputs.to(device), targets.to(device)

        # zero the gradients
        optimizer.zero_grad()
        # forward pass

        if USE_AE:
            inputs = autoencoder(inputs)

        outputs = model(inputs)
        # TODO pre_loss_fn ?
        # compute loss
        # loss = criterion(outputs, targets)

        # print(outputs.shape, targets.shape)

        # loss_list = list()
        #
        # for i in range(num_targets):
        #     loss_list.append(criterion(outputs[:, i], targets[:, i]))


        # back propagation
        # loss = sum(loss_list)
        loss = criterion(outputs, targets)
        # print(outputs.shape, targets.shape)
        # print(outputs, targets)

        loss.backward()
        # gradient descent
        optimizer.step()

        total_loss += loss.item()
        # if batch_idx % mini_batch_size == 9:
        #     last_loss = current_loss / mini_batch_size
        #     print(f'Loss after mini-batch {batch_idx + 1 : 3d}: {last_loss : .5f}')
        #     current_loss = 0.
        # exit(0)

    average_loss = total_loss / len(train_data)
    history_loss['avg_loss'].append(average_loss)

    print(f'Average Training Loss: {average_loss: .5f}')


def validate_one_epoch():
    model.eval()

    validation_loss = 0.

    full_pred = torch.Tensor().to(device)
    full_target = torch.Tensor().to(device)


    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # targets = targets.reshape((targets.shape[0], 1))
        inputs, targets = inputs.to(device), targets.to(device)
        if USE_AE:
            inputs = autoencoder(inputs)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        full_pred = torch.cat((full_pred, outputs))
        full_target = torch.cat((full_target, targets))

        validation_loss += loss.item()

    average_validation_loss = validation_loss / len(val_data)
    history_loss['avg_val_loss'].append(average_validation_loss)
    print(f'Validation loss: {average_validation_loss: .5f}')
    print(f'Validation RMSE: {rmse(full_pred, full_target)}')
    print(f'Validation RMSPE: {rmspe(full_pred, full_target)}')
    print(f'Validation RMSSE: {rmsse(full_pred, full_target)}')
    print(f'lag: {lag(full_target)}')
    print(f'lag rmspe: {lag(full_target) / torch.mean(full_target)}')

    # print(f'{full_pred =}')
    # print(f'{full_target =}')


# def hyperparameter_tuning(config, checkpoint_dir=None):
#     model = SharedWorkspaceModule(
#         h_dim=config['h_dim'],
#         ffn_dim=config['ffn_dim'],
#         num_layers=config['num_layers'],
#         num_heads=config['num_heads'],
#         dropout=config['dropout'],
#         shared_memory_attention=config['shared_memory_attention'],
#         share_vanilla_parameters=config['share_vanilla_parameters'],
#         use_topk=config['use_topk'],
#         topk=config['topk'],
#         mem_slots=config['mem_slots'],
#         num_targets=config['num_targets']
#     ).cuda()
#
#     criterion = torch.utils.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer + Shared Workspace predictor for job efficiency')

    parser.add_argument('--job_id', default=3418339, type=int, help='ID of the job considered for the model generation')
    parser.add_argument('--epochs', default=100, type=int, help='num of epochs to train')
    parser.add_argument('--ffn_dim', type=int, default=512)  # neurons in one feed forward layer
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size to use')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')

    parser.add_argument('--h_dim', type=int, default=64)  # this is shape of the input data TODO doublecheck
    parser.add_argument('--num_layers', default=12, type=int, help='num of layers')
    parser.add_argument('--num_heads', default=4, type=int, help='num of heads in Multi Head attention layer')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout')
    parser.add_argument('--shared_memory_attention', type=str2bool, default=False)
    parser.add_argument('--share_vanilla_parameters', type=str2bool, default=False)
    parser.add_argument('--use_topk', type=str2bool, default=False)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--mem_slots', type=int, default=4)
    # TODO currently omitted null_attention and num_steps -> not necessary afaik

    parser.add_argument('--prediction_targets', type=int, nargs='+', default=[0])
    parser.add_argument('--sliding_window', type=int, default=1)

    parser.add_argument('--columns_to_consider', type=str, default='GWT_efficiency_1')
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
    num_targets = len(args.prediction_targets)

    input_path = f'../data/task-usage_job-ID-{JOB_ID}_total.csv'
    figures_path = '../experiments_result/figures_GWT'
    results_path = 'results'
    model_path = '../models'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    columns_to_consider = columns_selection[args.columns_to_consider]  # TODO add argparse

    # transform = torchvision.transforms.ToTensor()

    data = prepare_data(input_path, columns_to_consider, targets=args.prediction_targets,
                        sliding_window=args.sliding_window, aggr_type='mean')

    train_data = ClusterDataset(data, num_targets=num_targets, training=True, split_percentage=0.7)
    val_data = ClusterDataset(data, num_targets=num_targets, training=False, split_percentage=0.7)

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
        mem_slots=args.mem_slots,
        num_targets=num_targets
    )  # TODO
    model.cuda()

    autoencoder = Autoencoder()
    autoencoder.cuda()

    criterion = nn.L1Loss().cuda()  # TODO different loss function ?
    # criterion = msse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)  # relevant ?
    # scheduler = torch.optim.lr_scheduler TODO add scheduler
    history_loss = defaultdict(list)

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1} / {epochs}')
        train_one_epoch()
        validate_one_epoch()

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
