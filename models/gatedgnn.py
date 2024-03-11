import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv import (
    MessagePassing,
    GatedGraphConv
)
from torch_geometric.nn.models import MLP
from models.gated_gcn_layer import ResGatedGCNLayer
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)

from torch_scatter import scatter, scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor


"""
    Gated Graph Sequence Neural Networks
    An Experimental Study of Neural Networks for Variable Graphs 
    Li Y, Tarlow D, Brockschmidt M, et al. Gated graph sequence neural networks[J]. arXiv preprint arXiv:1511.05493, 2015.
    https://arxiv.org/abs/1511.05493
    Note that the pyg and dgl of the gatedGCN are different models.
"""
class GatedGNN(nn.Module):

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nlayers: int,
        nclass: Optional[int] = None,
        dropout: float = 0.0,
        norm: Union[str, Callable, None] = None,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()

        in_dim_node = nfeat
        hidden_dim = nhid
        self.dropout = dropout
        self.nlayers = nlayers
        self.batch_norm = norm
        self.residual = True
        self.nclass = nclass

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)      
        self.layers = nn.ModuleList()
        for i in range(nlayers):
            self.layers.append(GatedGraphConv(hidden_dim, aggr = 'add'))
        if self.batch_norm:
            self.normlayers = nn.ModuleList()
            self.normlayers.append(nn.BatchNorm1d(hidden_dim))
        self.rnn = torch.nn.GRUCell(nhid, nhid, bias=bias)
        self.MLP_layer = nn.Linear(hidden_dim, nclass, bias=True)

    def initialize(self):
        r"""Resets all learnable parameters of the module."""
        if self.batch_norm:
            for norm in self.normlayers:
                norm.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.embedding_h.reset_parameters()
        self.MLP_layer.reset_parameters()
        self.rnn.reset_parameters()

    def forward(self, h, edge_index, edge_weight: OptTensor = None,):
        h = self.embedding_h(h)
        h_in = h
        for i in range(self.nlayers):
            m = self.layers[i](h, edge_index, edge_weight)
            h = self.rnn(m, h)
        if self.batch_norm:
            h = self.normlayers[0](h)
        if self.residual:
            h = h_in + h  
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.MLP_layer(h)
        return F.log_softmax(h,dim=1)

    @torch.no_grad()
    def predict(self, h, edge_index, edge_weight: OptTensor = None,):
        self.eval()
        return self.forward(h, edge_index, edge_weight)
    
    @torch.no_grad()
    def inference(self, loader: NeighborLoader,
                  device: Optional[torch.device] = None,
                  progress_bar: bool = False) -> Tensor:

        self.eval()
        assert isinstance(loader, NeighborLoader)
        assert len(loader.dataset) == loader.data.num_nodes
        assert not self.training

        self.embedding_h.to('cpu')
        self.MLP_layer.to('cpu')
        self.normlayers.to('cpu')
        x_all = loader.data.x.cpu()
        x_all = self.embedding_h(x_all)
        loader.data.n_id = torch.arange(x_all.size(0))

        temp=x_all
        for conv in self.layers:
            xs: List[Tensor] = []
            for batch in loader:
                h = x_all[batch.n_id].to(device)
                edge_index = batch.edge_index.to(device)
                edge_weight = batch.edge_weight.to(device)

                h_in = h[:batch.batch_size]
                h = conv(h, edge_index, edge_weight)[:batch.batch_size]
                h = self.rnn(h, h_in)
                xs.append(h.cpu())
            x_all = torch.cat(xs, dim=0)
        if self.batch_norm:
            h = self.normlayers[0](x_all)
        if self.residual:
            h = temp + h  # residual connection
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.MLP_layer(h)#第一次tensor([[-0.0899,  0.6363,  0.0926,  ...,  0.8083, -0.6894,  0.1474],

        del loader.data.n_id
        self.embedding_h.to(device)
        self.MLP_layer.to(device)
        self.normlayers.to(device)

        return F.log_softmax(h,dim=1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.nfeat}, '
                f'{self.nclass}, nlayers={self.nlayers})')
    
"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""

class ResGatedGCN(nn.Module):

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nlayers: int,
        nclass: Optional[int] = None,
        dropout: float = 0.0,
        norm: Union[str, Callable, None] = None,
        **kwargs,
    ):
        super().__init__()

        in_dim_node = nfeat
        in_dim_edge= 1
        hidden_dim = nhid
        n_classes = nclass
        num_bond_type=3
        self.dropout = dropout
        self.nlayers = nlayers
        self.batch_norm = norm
        self.residual = True
        self.n_classes = n_classes
        self.pos_enc = False
        self.edge_feat = False

        if self.pos_enc:
            pos_enc_dim = 1
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)       # node feat is an integer
        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)  # edge feat is a float

        self.layers = nn.ModuleList([ResGatedGCNLayer(hidden_dim, hidden_dim, self.dropout,
                                                   self.batch_norm, self.residual) for _ in range(self.nlayers)])
        if self.batch_norm:
            self.normlayers_h = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(self.nlayers)])
            self.normlayers_e = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(self.nlayers)])
        self.MLP_layer = nn.Linear(hidden_dim, n_classes, bias=True)

    def initialize(self):
        r"""Resets all learnable parameters of the module."""
        if self.batch_norm:
            for norm in self.normlayers_h:
                norm.reset_parameters()
            for norm in self.normlayers_e:
                norm.reset_parameters()
        self.embedding_h.reset_parameters()
        self.embedding_e.reset_parameters()
        self.MLP_layer.reset_parameters()

    def forward(self, h, edge_index, edge_weight: OptTensor = None, h_pos_enc=None):
        if edge_weight==None:
            edge_weight=edge_index.storage._value
            edge_index=torch.stack([edge_index.storage._row, edge_index.storage._col], dim=0)
        e=edge_weight.reshape(-1,1)

        # input embedding
        h = self.embedding_h(h)
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float())
            h = h + h_pos_enc
        e = self.embedding_e(e)
        # res gated convnets
        for i in range(self.nlayers):
            h_in = h
            e_in = e
            h, e = self.layers[i](h, edge_index, e)
            if self.batch_norm:
                h = self.normlayers_h[i](h)
                e = self.normlayers_e[i](e)  # batch normalization
            if self.residual:
                h = h_in + h  # residual connection
                e = e_in + e
            h = F.dropout(h, self.dropout, training=self.training)
            e = F.dropout(e, self.dropout, training=self.training)
        # output
        h_out = self.MLP_layer(h)
        return F.log_softmax(h_out,dim=1)

    @torch.no_grad()
    def predict(self, h, edge_index, edge_weight: OptTensor = None, h_pos_enc=None):
        self.eval()
        return self.forward(h, edge_index, edge_weight, h_pos_enc)
    
    @torch.no_grad()
    def inference(self, loader: NeighborLoader,
                  device: Optional[torch.device] = None,
                  progress_bar: bool = False) -> Tensor:

        self.eval()
        assert isinstance(loader, NeighborLoader)
        assert len(loader.dataset) == loader.data.num_nodes
        assert not self.training

        self.embedding_h.to('cpu')
        self.embedding_e.to('cpu')
        self.MLP_layer.to('cpu')
        x_all = self.embedding_h(loader.data.x.cpu())
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float())
            x_all = x_all + h_pos_enc
        initial_edge_weight=loader.data.edge_weight
        loader.data.edge_weight = self.embedding_e(loader.data.edge_weight.reshape(-1,1).cpu())
        loader.data.n_id = torch.arange(x_all.size(0))

        for i in range(self.nlayers):
            hs: List[Tensor] = []
            es: List[Tensor] = []
            for batch in loader:
                h = x_all[batch.n_id].to(device)
                h_in=h[:batch.batch_size]
                edge_index = batch.edge_index.to(device)
                e = batch.edge_weight.to(device)#这里的边全都是该batch的节点的边，所以下面es直接append就好。问题：这里的batch会不会随着data.edge_index变？会的
                e_in = e

                h, e = self.layers[i](h, edge_index, e)
                h = h[:batch.batch_size]
                if self.batch_norm:
                    h = self.normlayers_h[i](h)
                    e = self.normlayers_h[i](e)
                if self.residual:
                    h = h_in + h  # residual connection
                    e = e_in + e  # re
                h = F.dropout(h, self.dropout, training=self.training)
                e = F.dropout(e, self.dropout, training=self.training)
                hs.append(h.cpu())
                es.append(e.cpu())

            x_all = torch.cat(hs, dim=0)
            loader.data.edge_weight = torch.cat(es, dim=0)

        h = self.MLP_layer(x_all)

        del loader.data.n_id
        self.embedding_h.to(device)
        self.embedding_e.to(device)
        self.MLP_layer.to(device)
        loader.data.edge_weight=initial_edge_weight
        return F.log_softmax(h,dim=1)
    