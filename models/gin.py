from numpy import float32
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics import f1_score
from torch.nn import init
import torch_sparse
from torch_geometric.nn.inits import *
from torch_geometric.loader import NeighborSampler
from torch_sparse import SparseTensor
import gc
import os

class GIN(nn.Module):
    
    def __init__(self, nfeat, nhid, nclass, nlayers=2, mlp_layers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, with_bn=False, device=None):

        super(GIN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.layers = nn.ModuleList([])
        self.nlayers=nlayers
        self.mlp_layers=mlp_layers

        if nlayers == 1:
            if with_bn:
                self.bns = nn.ModuleList()#储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器
                for i in range(mlp_layers-1):
                    self.bns.append(nn.BatchNorm1d(nhid))

            for i in range(mlp_layers):
                if i==0:
                    self.layers.append(MyLinear(nfeat,nhid))
                elif i!=mlp_layers-1:
                    self.layers.append(MyLinear(nhid,nhid))
                else:
                    self.layers.append(MyLinear(nhid,nclass))                
        else:
            if with_bn:
                self.bns = nn.ModuleList()#储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器
                for i in range(nlayers*mlp_layers-1):
                    self.bns.append(nn.BatchNorm1d(nhid))
            #第一层linear
            self.layers.append(MyLinear(nfeat,nhid))
            #中间层linear
            for i in range(nlayers*mlp_layers-2):
                self.layers.append(MyLinear(nhid,nhid))
            #输出层linear
            self.layers.append(MyLinear(nhid,nclass))

        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None#syn
        self.features = None#syn
        self.multi_label = None

    def forward(self, x, adj):#tensor.sparse_coo
        for ix, layer in enumerate(self.layers):
            if(ix%self.mlp_layers==0):
                if isinstance(adj, torch_sparse.SparseTensor):#一样可以用SparseTensor或者EdgeIndex！！！！！
                    x = x+torch_sparse.matmul(adj, x)
                else:
                    x = x+torch.spmm(adj, x)

            x = layer.forward(x)#layer是一个GraphConvolution layer

            if ix != len(self.layers) - 1:#最后一层不dropout和BN
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    @torch.no_grad()
    def inference(self, x_all, adj, device):#subgraph
        #x_all:tensor adj:SparseTensor all on cpu
        self.eval()
        subgraph_loader = NeighborSampler(adj, node_idx=None, sizes=[-1],#要求adj是SparseTensor或者EdgeIndex
                                  batch_size=100000, shuffle=False,
                                  num_workers=12)
        for ix, layer in enumerate(self.layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(self.device)#edge_index还是SparseTensor或者EdgeIndex
                x = x_all[n_id].to(self.device)
                
                if(ix%self.mlp_layers==0):
                    if isinstance(edge_index, torch_sparse.SparseTensor):#一样可以用SparseTensor或者EdgeIndex！！！！！
                        x = x_all[n_id[:batch_size]].to(self.device)+torch_sparse.matmul(edge_index, x)
                    else:
                        x = x_all[n_id[:batch_size]].to(self.device)+torch.spmm(edge_index, x)

                x = layer(x)
                if ix != self.nlayers*self.mlp_layers - 1:
                    x = self.bns[ix](x) if self.with_bn else x
                    if self.with_relu:
                        x = F.relu(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        if self.multi_label:
            return torch.sigmoid(x_all.to(device))
        else:
            return F.log_softmax(x_all.to(device), dim=1)


class MyLinear(Module):
    """Simple Linear layer, modified from https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        if self.bias is not None:
            return support + self.bias
        else:
            return support

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
