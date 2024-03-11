import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)
sys.path.insert(0,'/home/disk3/xzb/GCond/models')
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_sparse
import scipy.sparse as sp
import numpy as np

from copy import deepcopy
from deeprobust.graph import utils
from models.gcn_lib.sparse.torch_nn import norm_layer
from models.gcn_lib.sparse.torch_vertex import GENConv
from numpy import float32
from sklearn.metrics import f1_score
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.utils.checkpoint import checkpoint
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn.inits import *
from torch_sparse import SparseTensor

class DeeperGCN(torch.nn.Module):
    def __init__(self, args, nfeat, nclass, device):
        super(DeeperGCN, self).__init__()
        self.device = device

        self.num_layers = args.deep_layers
        self.dropout = args.dropout
        self.block = args.block

        self.checkpoint_grad = False

        in_channels = nfeat
        hidden_channels = args.deep_hidden
        num_tasks = nclass
        conv = args.conv
        aggr = args.gcn_aggr

        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        y = args.y
        self.learn_y = args.learn_y

        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale

        norm = args.norm
        mlp_layers = args.mlp_layers

        if aggr in ['softmax_sg', 'softmax', 'power'] and self.num_layers > 7:
            self.checkpoint_grad = True
            self.ckp_k = self.num_layers // 2

        print('The number of layers {}'.format(self.num_layers),
              'Aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))

        if self.block == 'res+':
            print('LN/BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->LN/BN->ReLU->Res')
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU')
        else:
            raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.node_features_encoder = torch.nn.Linear(in_channels, hidden_channels)
        self.node_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

        for layer in range(self.num_layers):

            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              y=y, learn_y=self.learn_y,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')

            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

    def forward(self,  x, edge_index):
        #x是tensor,edge_index为2*(longtensor)
        h = self.node_features_encoder(x)

        if self.block == 'res+':

            h = self.gcns[0](h, edge_index)

            if self.checkpoint_grad:

                for layer in range(1, self.num_layers):
                    h1 = self.norms[layer - 1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)

                    if layer % self.ckp_k != 0:
                        res = checkpoint(self.gcns[layer], h2, edge_index)
                        h = res + h
                    else:
                        h = self.gcns[layer](h2, edge_index) + h

            else:
                for layer in range(1, self.num_layers):
                    h1 = self.norms[layer - 1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)
                    h = self.gcns[layer](h2, edge_index) + h

            h = F.relu(self.norms[self.num_layers - 1](h))
            h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'res':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index)
                h2 = self.norms[layer](h1)
                h = F.relu(h2)
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception('Unknown block Type')

        h = self.node_pred_linear(h)

        return torch.log_softmax(h, dim=-1)

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))

        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))

        if self.learn_y:
            ys = []
            for gcn in self.gcns:
                ys.append(gcn.sigmoid_y.item())
            if final:
                print('Final sigmoid(y) {}'.format(ys))
            else:
                logging.info('Epoch {}, sigmoid(y) {}'.format(epoch, ys))

        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.node_features_encoder.reset_parameters()
        self.node_pred_linear.reset_parameters()

    def fit_with_val(self, args, features, adj, labels, data, train_iters=200, initialize=True, large=False, verbose=False, normalize=True, patience=None, noval=False, **kwargs):
        #features, adj, labels are from syn!!!!!!!!!!!!!!!!!
        '''data: full data class'''
        if initialize:
            self.initialize()#renew gcn

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)#from syn
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils1 import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm
        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels

        self._train_with_val(labels, data, train_iters, verbose, adj_val=noval)#pass syn labels

    def _train_with_val(self, labels, data, train_iters, verbose, adj_val=False):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val#inductive
        else:
            feat_full, adj_full = data.feat_full, data.adj_full#transductive 

        adj_full = utils.to_tensor1(adj_full, device=self.device)#trans: whole big graph
        adj_full = utils.normalize_adj_tensor(adj_full, sparse=True)
        adj_full = SparseTensor(row=adj_full._indices()[0], col=adj_full._indices()[1],
                value=adj_full._values(), sparse_sizes=adj_full.size()).t()#所有东西丢入深度学习网络之前都需要转变为tensor
        row, col, _ = adj_full.coo()  #data is torch_geometric.data.data.Data
        adj_full = torch.stack([row, col], axis=0)

        adj_syn = sp.coo_matrix(self.adj_norm.cpu().detach().numpy())
        indices = np.vstack((adj_syn.row, adj_syn.col))  # 我们真正需要的coo形式
        adj_syn = torch.LongTensor(indices).to(self.device)  # PyG框架需要的coo形式

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=0e-4)

        best_acc_val = 0
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = 0.001
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0e-4)#gcn parameters!!!!!!!!!!!

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, adj_syn)#use condensed graph and gcn to fit gcn
            loss_train = self.loss(output, labels)
            loss_train.backward()
            optimizer.step()

            if i % 20 == 0:
                with torch.no_grad():
                    self.eval()
                    # output = self.inference(feat_full, adj_full)#numpy+SparseTensor
                    output = self.forward(torch.FloatTensor(feat_full).to(self.device), adj_full)#numpy+SparseTensor

                    if adj_val:#induct
                        loss_val = F.nll_loss(output, torch.LongTensor(data.labels_val).to(self.device))
                        acc_val = utils.accuracy(output, torch.LongTensor(data.labels_val).to(self.device))
                    else:#transduct
                        loss_val = F.nll_loss(output[data.idx_val], torch.LongTensor(data.labels_val).to(self.device))
                        acc_val = utils.accuracy(output[data.idx_val], torch.LongTensor(data.labels_val).to(self.device))

                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        self.output = output
                        weights = deepcopy(self.state_dict())#state_dict是在定义了model或optimizer之后pytorch自动生成的,即model此时的参数

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        
    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    @torch.no_grad()
    def predict(self, features=None, adj=None):#align format and forward
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                adj,features = utils.to_tensor(adj,features, device=self.device)

            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
        adj_full = SparseTensor(row=self.adj_norm._indices()[0], col=self.adj_norm._indices()[1],
                value=self.adj_norm._values(), sparse_sizes=self.adj_norm.size()).t()#所有东西丢入深度学习网络之前都需要转变为tensor
        row, col, _ = adj_full.coo()  #data is torch_geometric.data.data.Data
        adj_full = torch.stack([row, col], axis=0)
        adj_full=adj_full.type(torch.LongTensor).to(self.device)

        return self.forward(features, adj_full)

    @torch.no_grad()
    def inference(self, x_all, adj):#subgraph predict
        #x_all:numpy adj:SparseTensor all on cpu
        subgraph_loader = NeighborSampler(adj, node_idx=None, sizes=[-1],#要求adj是SparseTensor或者EdgeIndex
                                  batch_size=4096, shuffle=False,
                                  num_workers=12)
        for ix, layer in enumerate(self.layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(self.device)#edge_index还是SparseTensor或者EdgeIndex
                if torch.is_tensor(x_all):
                    x = x_all[n_id].to(self.device)
                else:
                    x = torch.FloatTensor(x_all[n_id]).to(self.device)  #使用上一层卷积跟新过的embedding, x_all
                x_target = x[:size[1]]
                x = layer(x, edge_index)
                if ix != self.nlayers - 1:
                    x = self.bns[ix](x) if self.with_bn else x
                    if self.with_relu:
                        x = F.relu(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        if self.multi_label:
            return torch.sigmoid(x_all)
        else:
            return F.log_softmax(x_all, dim=1)#return cpu data



