import argparse
import sys
import json
import time

import einops
import numpy as np
from collections import defaultdict

from gcd_data_manipulation import ClusterDataset

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TransformerEncoder





class SharedWorkspaceModel(nn.Module):
    def __init__(self, h_dim, output_shape):
        super().__init__()
        self.transformer = TransformerEncoder(
            # embed_dim=args.embed_dim, TODO
            embed_dim=h_dim,
            ffn_dim=ffn_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            shared_memory_attention=args.shared_memory_attention,
            share_parameters=args.share_vanilla_parameters,
            use_topk=args.use_topk,
            topk=args.topk,
            mem_slots=args.mem_slots
        )
        self.h_dim = h_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, h_dim))
        self.output = nn.Linear(ffn_dim, output_shape)

    def forward(self, inputs):
        print(inputs.shape)
        x = inputs.cuda()

        # x = einops.repeat(x, 'b f -> b (a f) d', a=16, d=self.h_dim)
        #
        # b, _, _ = x.shape
        #
        # cls_tokens = einops.repeat(self.cls_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_tokens, x), dim=1)
        #
        # # x = einops.rearrange(x, 'b f -> b f f')
        # print(x.shape)
        #
        # x = self.transformer(x)
        # # x = self.mlp_head(x[:,0])
        #
        # print('SURVIVED THE TRANSFORMER :OOO')

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

    transform = torchvision.transforms.ToTensor()

    train_data = ClusterDataset(input_path, columns_to_consider, aggr_type='mean', training=True, split_percentage=0.7)
    test_data = ClusterDataset(input_path, columns_to_consider, aggr_type='mean', training=False, split_percentage=0.7)

    train_data.values.to(device)
    test_data.values.to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)


    results = defaultdict(list)

    model = SharedWorkspaceModel(h_dim=args.h_dim, output_shape=1)  # TODO
    model.to(device)

    criterion = nn.L1Loss()  # TODO try MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler TODO add scheduler

    for epoch in range(1, epochs):
        print(f'\nEpoch: {epoch}')
        start_time = time.time()
        model.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # TODO remove encoding step from transformer
            # inputs = torch.unsqueeze(inputs, dim=1)
            # inputs = torch.transpose(inputs, 0, 2)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 8 == 0:
                print(f'loss after {batch_idx} batches: {train_loss}')
                running_loss = 0

        print(f'Loss for epoch {epoch}: {train_loss}')
