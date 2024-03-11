import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter, scatter_add
from torch_geometric.nn.conv import MessagePassing

"""
    ResGatedGCN: Residual Gated Graph ConvNets for pyg implement, is made by myself
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
class ResGatedGCNLayer(MessagePassing):
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: Tensor, Ah: Tensor ,edge_weight: OptTensor):
        e_ij = edge_weight + alpha_j + alpha_i# e_ij = Ce_ij + Dhi + Ehj
        return [x_j, e_ij, Ah]

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        Ah_i = inputs[2]
        Bh_j = inputs[0]
        sigma_ij = torch.sigmoid(inputs[1])
        e = inputs[1]
        # aa=scatter(sigma_ij * Bh_j, index, dim=self.node_dim, dim_size=dim_size,
        #         reduce='add')
        h = Ah_i + scatter(sigma_ij*Bh_j, index, dim= self.node_dim, dim_size=dim_size,
                reduce='add') / (scatter(sigma_ij, index, dim=self.node_dim, dim_size=dim_size, reduce='sum') + 1e-6)
        return [h, e]
        # hi = Ahi + sum_j eta_ij/sum_j' eta_ij' * Bhj <= dense attention

    def forward(self, h, edge_index, edge_weight):
        # h = conv(h, edge_index, e)g, h, e
        Ah = self.A(h)
        Bh = self.B(h)
        Dh = self.D(h)
        Eh = self.E(h)
        Ce = self.C(edge_weight)
        # g.update_all(self.message_func, self.reduce_func)
        m = self.propagate(edge_index, x=(Bh,Bh), alpha=(Dh,Eh), Ah=Ah, edge_weight=Ce,
                           size=None)
        h = m[0]  # result of graph convolution
        e = m[1]  # result of graph convolution

        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                                            self.in_channels,
                                                            self.out_channels)
