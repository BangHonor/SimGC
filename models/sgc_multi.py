"""multiple transformaiton and multiple propagation"""
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
from torch_geometric.loader import NeighborSampler
from torch_sparse import SparseTensor

class SGC(nn.Module):

    def __init__(self, nfeat, nhid, nclass, naggr=3, ntrans=3, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_bias=True, with_bn=False, device=None):

        """naggr indicates the number of aggre propagations"""
        super(SGC, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass


        self.layers = nn.ModuleList([])
        if ntrans == 1:
            self.layers.append(MyLinear(nfeat, nclass))
        else:
            self.layers.append(MyLinear(nfeat, nhid))
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            for i in range(ntrans-2):
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
                self.layers.append(MyLinear(nhid, nhid))
            self.layers.append(MyLinear(nhid, nclass))

        self.naggr = naggr
        self.dropout = dropout
        self.lr = lr
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.weight_decay = weight_decay
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None

    def forward(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        for i in range(self.naggr):
            if isinstance(adj, torch_sparse.SparseTensor):#一样可以用SparseTensor或者EdgeIndex！！！！！
                x = torch_sparse.matmul(adj, x)
            else:
                x = torch.spmm(adj, x)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler(self, x, adjs):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        for ix, (adj, _, size) in enumerate(adjs):
            x = torch_sparse.matmul(adj, x)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler_large(self, x, adjs):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        for ix, adj in enumerate(adjs):
            adj=adj[0]
            x = torch_sparse.matmul(adj, x)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler_syn(self, x, adjs):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        for ix, (adj) in enumerate(adjs):
            if type(adj) == torch.Tensor:
                x = adj @ x
            else:
                x = torch_sparse.matmul(adj, x)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)


    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def fit_with_val(self, features, adj, labels, data, train_iters=200, initialize=True, verbose=False, normalize=True, patience=None, noval=False, **kwargs):
        '''data: full data class'''
        if initialize:
            self.initialize()

        # features, adj, labels = data.feat_train, data.adj_train, data.labels_train
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
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

        if noval:
            self._train_with_val(labels, data, train_iters, verbose, adj_val=True)
        else:
            self._train_with_val(labels, data, train_iters, verbose)

    def _train_with_val(self, labels, data, train_iters, verbose, adj_val=False):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full

        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_val = torch.LongTensor(data.labels_val).to(self.device)

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0

        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = self.loss(output, labels)
            loss_train.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)
                if adj_val:
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

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
    def predict(self, features=None, adj=None):
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
        elif type(adj) is not SparseTensor:
            if type(adj) is not torch.Tensor:
                adj,features = utils.to_tensor(adj,features, device=self.device)

            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(features, adj_norm)
        else:
            return self.forward(features, adj)

    def inference(self, x_all, adj, device):#subgraph
        #x_all:tensor adj:SparseTensor all on cpu
        self.eval()
        subgraph_loader = NeighborSampler(adj, node_idx=None, sizes=[-1],#要求adj是SparseTensor或者EdgeIndex
                                  batch_size=100000, shuffle=False,
                                  num_workers=12)
        for ix, layer in enumerate(self.layers):
            x_all = layer(x_all)
            if ix != len(self.layers) - 1:
                x_all = self.bns[ix](x_all) if self.with_bn else x_all
                x_all = F.relu(x_all)
                x_all = F.dropout(x_all, self.dropout, training=self.training)

        for i in range(self.naggr):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(self.device)#edge_index还是SparseTensor或者EdgeIndex
                if torch.is_tensor(x_all):
                    x = x_all[n_id].to(self.device)
                else:
                    x = torch.FloatTensor(x_all[n_id]).to(self.device)  #使用上一层卷积跟新过的embedding, x_all

                if isinstance(edge_index, torch_sparse.SparseTensor):#一样可以用SparseTensor或者EdgeIndex！！！！！
                    x = torch_sparse.matmul(edge_index, x)
                else:
                    x = torch.spmm(edge_index, x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)

        if self.multi_label:
            return torch.sigmoid(x_all.to(device))
        else:
            return F.log_softmax(x_all.to(device), dim=1)

    @torch.no_grad()
    def predict_unnorm(self, features=None, adj=None):
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            self.adj_norm = adj
            return self.forward(self.features, self.adj_norm)



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
        output = support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


