import torch
import torch.nn as nn
from torch._C._return_types import min

from transformers import TransformerEncoder

import einops


class SharedWorkspaceModule(nn.Module):

    def __init__(self, h_dim, ffn_dim, num_layers, num_heads, dropout, shared_memory_attention,
                 share_vanilla_parameters, use_topk, topk, mem_slots, num_targets):
        super().__init__()
        self.transformer = TransformerEncoder(
            embed_dim=h_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            shared_memory_attention=shared_memory_attention,
            share_parameters=share_vanilla_parameters,
            use_topk=use_topk,
            topk=topk,
            mem_slots=mem_slots
        )

        self.h_dim = h_dim

        # self.sigmoid = nn.Sigmoid()

        self.cls_token = nn.Parameter(torch.randn(1, 1, h_dim))
        self.output = nn.Linear(h_dim, num_targets)  # TODO

    def forward(self, inputs):
        # print(inputs.shape)
        x = inputs.cuda()
        x = einops.repeat(x, 'b f -> b f h', h=self.h_dim)

        # TODO maybe only makes sense for sliding window
        # cls_tokens = einops.repeat(self.cls_token, '() n d -> b n d', b=x.shape[0])
        # x = torch.cat((cls_tokens, x), dim=1)
        #
        # print(x.shape)
        x = self.transformer(x)
        # TODO maybe insert here cls token

        # print(x.shape)
        # print(x[:, 0], x[:, 0].shape)

        x = self.output(x[:, 0])
        # x = self.sigmoid(x)  # TODO check validity
        # x = x.clamp(min=0, max=1)
        # print(x)
        return x
