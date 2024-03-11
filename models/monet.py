import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import numpy as np
from torch_geometric.nn import GMMConv
"""
    GMM: Gaussian Mixture Model Convolution layer
    Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs (Federico Monti et al., CVPR 2017)
    https://arxiv.org/pdf/1611.08402.pdf
"""

class MoNet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, kernel=1, dim=1, with_bias=True, norm=False, device=None):
        super().__init__()
        self.name = 'MoNet'
        in_dim_node = nfeat
        hidden_dim = nhid
        out_dim = nhid
        kernel = kernel#uj的维度 dim*kernel
        dim = dim#是A矩阵的维度 dim*2
        n_classes = nclass
        self.dropout = dropout
        self.n_layers = nlayers
        self.batch_norm = norm
        self.residual = with_bias
        self.device = device
        self.n_classes = n_classes
        self.dim = dim
        # aggr_type = "sum"  # default for MoNet
        aggr_type = "sum"

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)       # node feat is an integer
        # self.embedding_e = nn.Linear(1, dim)  # edge feat is a float
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()
        self.batchnorm_h = nn.ModuleList()

        # Hidden layer
        for _ in range(self.n_layers - 1):
            self.layers.append(GMMConv(hidden_dim, hidden_dim, dim, kernel, separate_gaussians = False ,aggr = aggr_type,
                                        root_weight = True, bias = True))
            if self.batch_norm:
                self.batchnorm_h.append(nn.BatchNorm1d(hidden_dim))
            self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
            
        # Output layer
        self.layers.append(GMMConv(hidden_dim, out_dim, dim, kernel, separate_gaussians = False ,aggr = aggr_type,
                                        root_weight = True, bias = True))
        if self.batch_norm:
            self.batchnorm_h.append(nn.BatchNorm1d(out_dim))
        self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        # self.MLP_layer = MLPReadout(out_dim, n_classes)
        self.MLP_layer = nn.Linear(out_dim, n_classes, bias=True)
        # to do

    def forward(self, h, edge_index, edge_weight):
        h = self.embedding_h(h)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=h.size(0))
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        pseudo = torch.cat((deg_inv_sqrt[row].unsqueeze(-1), deg_inv_sqrt[col].unsqueeze(-1)), dim=1)

        for i in range(self.n_layers):
            h_in = h
            h = self.layers[i](h, edge_index, self.pseudo_proj[i](pseudo))
            if self.batch_norm:
                h = self.batchnorm_h[i](h)  # batch normalization
            h = F.relu(h)  # non-linear activation
            if self.residual:
                h = h_in + h  # residual connection
            h = F.dropout(h, self.dropout, training=self.training)

        return self.MLP_layer(h)
    
    @torch.no_grad()
    def predict(self, h, edge_index, edge_weight, syn=False):
        h = self.embedding_h(h)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=h.size(0))#计算每一个点的degree
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        pseudo = torch.cat((deg_inv_sqrt[row].unsqueeze(-1), deg_inv_sqrt[col].unsqueeze(-1)), dim=1)#伪坐标

        for i in range(self.n_layers):
            h_in = h
            h = self.layers[i](h, edge_index, self.pseudo_proj[i](pseudo))
            if self.batch_norm:
                h = self.batchnorm_h[i](h)  # batch normalization
            h = F.relu(h)  # non-linear activation
            if self.residual:
                h = h_in + h  # residual connection
            h = F.dropout(h, self.dropout, training=self.training)

        return self.MLP_layer(h)
    
    def initialize(self):
        self.embedding_h.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.pseudo_proj:
            layer[0].reset_parameters()
        if self.batch_norm:
            for bn in self.batchnorm_h:
                bn.reset_parameters()
        self.MLP_layer.reset_parameters()