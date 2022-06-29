import torch
import torch.nn as nn
from transformers import TransformerEncoder

import einops


class SharedWorkspaceModule(nn.Module):

    def __init__(self, h_dim, ffn_dim, num_layers, num_heads, dropout, shared_memory_attention,
                 share_vanilla_parameters, use_topk, topk, mem_slots):
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

        self.cls_token = nn.Parameter(torch.randn(1, 1, h_dim))
        self.output = nn.Linear(h_dim, 1)  # TODO

    def forward(self, inputs):
        # print(inputs.shape)
        x = inputs.cuda()
        x = einops.repeat(x, 'b f -> b f h', h=self.h_dim)

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

        # TODO maybe insert here cls token

        x = self.output(x[:, 0])
        return x
