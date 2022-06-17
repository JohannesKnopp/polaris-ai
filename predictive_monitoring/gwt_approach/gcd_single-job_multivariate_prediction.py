import argparse
import sys
import json
import time
import numpy as np
from collections import defaultdict

from gcd_data_manipulation import ClusterDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TransformerEncoder





class SharedWorkspaceModel(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.train_x = train_x
        self.transformer = TransformerEncoder(
            # embed_dim=args.embed_dim, TODO
            embed_dim=input_shape,
            ffn_dim=neurons,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            shared_memory_attention=args.shared_memory_attention,
            share_parameters=args.share_vanilla_parameters,
            use_topk=args.use_topk,
            topk=args.topk,
            mem_slots=args.mem_slots
        )
        self.output = nn.Linear(neurons, output_shape)

    def forward(self, features):
        x = self.transformer(features)
        x = self.output(x)
        return x


def str2bool(v):
    """Method to map string to bool for argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def train_model(epoch):
    print('\nEpoch: %d', epoch)
    start_time = time.time()
    model.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer + Shared Workspace predictor for job efficiency')

    parser.add_argument('--job_id', default=3418339, type=int, help='ID of the job considered for the model generation')
    parser.add_argument('--epochs', default=100, type=int, help='num of epochs to train')
    parser.add_argument('--neurons', type=int, default=512)  # neurons in one feed forward layer
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size to use')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')

    # parser.add_argument('--embed_dim', type=int, default=256) this is shape of the input data TODO doublecheck
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
    neurons = args.neurons
    batch_size = args.batch_size
    exp_name = args.exp_name
    JOB_ID = args.job_id

    input_path = f'../data/task-usage_job-ID-{JOB_ID}_total.csv'
    figures_path = '../experiments_result/figures_GWT'
    results_path = 'results'
    model_path = '../models'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    columns_to_consider = columns_selection['GWT_efficiency_1']  # TODO add argparse

    cluster_data = ClusterDataset(input_path, columns_to_consider, aggr_type='mean') # TODO vary aggr_type

    train_loader = DataLoader(cluster_data, )


    results = defaultdict(list)

    # >>> print(train_x.shape, train_y.shape)
    # (4032, 1, 16) (4032,)

    model = SharedWorkspaceModel(input_shape=train_x.shape[2], output_shape=train_x.shape[1])
    model.to(device)

    criterion = nn.L1Loss() # TODO try MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler TODO add scheduler

    for epoch in range(1, epochs):
        train_model(epoch)

    # history_loss = fit_model(train_x, train_y, test_x, test_y, model, epochs, batch_size)

