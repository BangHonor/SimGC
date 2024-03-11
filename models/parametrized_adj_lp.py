import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from itertools import product
import numpy as np

class PGE_Edge(nn.Module):

    def __init__(self, nfeat, nhid=256, nlayers=3, device=None, args=None):
        super(PGE_Edge, self).__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat*2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers-2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))

        self.device = device
        self.reset_parameters()
        self.cnt = 0
        self.args = args

    def forward(self, edge_embed, inference=False):
        for ix, layer in enumerate(self.layers):
            edge_embed = layer(edge_embed)
            if ix != len(self.layers) - 1:
                edge_embed = self.bns[ix](edge_embed)
                edge_embed = F.relu(edge_embed)

        pred = torch.sigmoid(edge_embed).reshape(-1)

        return pred

    @torch.no_grad()
    def inference(self, edge_embed):
        self.eval()
        pred = self.forward(edge_embed, inference=True)
        return pred

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)