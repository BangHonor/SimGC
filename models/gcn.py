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

class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))#确定了这些是GCN的参数，在back pro的时候会对齐求导
        self.bias = Parameter(torch.Tensor(out_features))
        self.weight.requires_grad=True
        self.bias.requires_grad=True

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, input, adj):#整体的表达式为：A(HW) out grad=true
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        if isinstance(adj, torch_sparse.SparseTensor):#一样可以用SparseTensor或者EdgeIndex！！！！！
            output = torch_sparse.matmul(adj, support)
        else:
            output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, with_bn=False, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass

        self.layers = nn.ModuleList([])
        self.nlayers=nlayers
        if nlayers == 1:
            self.layers.append(GraphConvolution(nfeat, nclass, with_bias=with_bias))
        else:#一个GCN有多层卷积
            if with_bn:
                self.bns = nn.ModuleList()#储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器
                self.bns.append(nn.BatchNorm1d(nhid))#对小批量(mini-batch)的2d或3d输入进行批标准化(Batch Normalization)操作
            self.layers.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(nlayers-2):
                self.layers.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nhid, nclass, with_bias=with_bias))#GCN下面所有的层

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
            x = layer.forward(x, adj)#layer是一个GraphConvolution layer
            if ix != len(self.layers) - 1:#最后一层不dropout和BN
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler(self, x, adjs):#SparseTensor
        for ix, (adj, _, size) in enumerate(adjs):
            x = self.layers[ix](x, adj)#每一层采样的节点不一样，邻接矩阵也不一样
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler_syn(self, x, adjs):
        for ix, (adj, _, size) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

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

        if large==False:
            self._train_with_val_perturbation(labels, data, train_iters, verbose, adj_val=noval)#pass syn labels

    def _train_with_val(self, labels, data, train_iters, verbose, adj_val=False):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val#inductive
        else:
            feat_full, adj_full = data.feat_full, data.adj_full#transductive 

        adj_full = utils.to_tensor1(adj_full, device=self.device)#trans: whole big graph
        adj_full = utils.normalize_adj_tensor(adj_full, sparse=True)
        adj_full = SparseTensor(row=adj_full._indices()[0], col=adj_full._indices()[1],
                value=adj_full._values(), sparse_sizes=adj_full.size()).t()#所有东西丢入深度学习网络之前都需要转变为tensor

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)#gcn parameters!!!!!!!!!!!

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)#use condensed graph and gcn to fit gcn
            loss_train = self.loss(output, labels)
            loss_train.backward()
            optimizer.step()

            if i % 20 == 0:
                with torch.no_grad():
                    self.eval()
                    output = self.inference(feat_full, adj_full)#numpy+SparseTensor

                    if adj_val:#induct
                        loss_val = F.nll_loss(output, torch.LongTensor(data.labels_val))
                        acc_val = utils.accuracy(output, torch.LongTensor(data.labels_val))
                    else:#transduct
                        loss_val = F.nll_loss(output[data.idx_val], torch.LongTensor(data.labels_val))
                        acc_val = utils.accuracy(output[data.idx_val], torch.LongTensor(data.labels_val))

                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        self.output = output
                        weights = deepcopy(self.state_dict())#state_dict是在定义了model或optimizer之后pytorch自动生成的,即model此时的参数

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
    
    def _train_with_val_perturbation(self, labels, data, train_iters, verbose, adj_val=False):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val#inductive
        else:
            feat_full, adj_full = data.feat_full, data.adj_full#transductive 

        adj_full = utils.to_tensor1(adj_full, device=self.device)#trans: whole big graph
        adj_full = utils.normalize_adj_tensor(adj_full, sparse=True)
        adj_full = SparseTensor(row=adj_full._indices()[0], col=adj_full._indices()[1],
                value=adj_full._values(), sparse_sizes=adj_full.size()).t()#所有东西丢入深度学习网络之前都需要转变为tensor

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)#gcn parameters!!!!!!!!!!!

            self.train()
            perturb = torch.FloatTensor(self.features.shape[0], self.features.shape[1]).uniform_(8e-3, 8e-3).to(self.device)
            perturb.requires_grad_()
            # perturb_adj = torch.FloatTensor(self.adj_norm.shape[0], self.adj_norm.shape[1]).uniform_(8e-3, 8e-3).to(self.device)
            # perturb_adj.requires_grad_()

            optimizer.zero_grad()
            output = self.forward(self.features+perturb, self.adj_norm)#use condensed graph and gcn to fit gcn
            loss_train = self.loss(output, labels)/3
            for _ in range(2):
                loss_train.backward()
                perturb_data = perturb.detach() + 8e-3 * torch.sign(perturb.grad.detach())#grad不会叠加，而optimizer会
                perturb.data = perturb_data.data
                perturb.grad[:] = 0

                # perturb_data_adj = perturb_adj.detach() + 8e-3 * torch.sign(perturb_adj.grad.detach())#grad不会叠加，而optimizer会
                # perturb_adj.data = perturb_data_adj.data
                # perturb_adj.grad[:] = 0

                output = self.forward(self.features+perturb, self.adj_norm)#use condensed graph and gcn to fit gcn
                loss_train = self.loss(output, labels)/3

            loss_train.backward()
            optimizer.step()#相当于一次性进行了多次梯度下降

            if i % 20 == 0:
                with torch.no_grad():
                    self.eval()
                    output = self.inference(feat_full, adj_full)#numpy+SparseTensor

                    if adj_val:#induct
                        loss_val = F.nll_loss(output, torch.LongTensor(data.labels_val))
                        acc_val = utils.accuracy(output, torch.LongTensor(data.labels_val))
                    else:#transduct
                        loss_val = F.nll_loss(output[data.idx_val], torch.LongTensor(data.labels_val))
                        acc_val = utils.accuracy(output[data.idx_val], torch.LongTensor(data.labels_val))

                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        self.output = output
                        weights = deepcopy(self.state_dict())#state_dict是在定义了model或optimizer之后pytorch自动生成的,即model此时的参数

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    # def _train_with_val_large(self, args, labels, data, train_iters, verbose):
    #     if verbose:
    #         print('=== training gcn model ===')
    #     feat_full, adj_full = data.feat_val, data.adj_val#transductive 
    #     labels_full = torch.LongTensor(data.labels_val).to(self.device)
    #     if os.path.exists('/home/xzb/GCond/saved_ours/adj_val_'+str(args.dataset)+'_'+str(args.seed)):
    #         adj_full=torch.load(f'/home/xzb/GCond/saved_ours/adj_val_{args.dataset}_{args.seed}.pt')
    #     else:
    #         adj_full=utils.to_tensor1(adj=adj_full)
    #         if utils.is_sparse_tensor(adj_full):
    #             adj_norm = utils.normalize_adj_tensor(adj_full, sparse=True)#D^(-0.5)*(A+I)*D^0.5
    #         else:
    #             adj_norm = utils.normalize_adj_tensor(adj_full)
    #         adj_full = adj_norm#正则化之后的邻接矩阵A 2708*2708
    #         adj_full = SparseTensor(row=adj_full._indices()[0], col=adj_full._indices()[1],
    #                 value=adj_full._values(), sparse_sizes=adj_full.size()).t()
    #         torch.save(adj_full, f'/home/xzb/GCond/saved_ours/adj_val_{args.dataset}_{args.seed}.pt')

    #     optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     best_acc_val = 0

    #     for i in range(train_iters):
    #         if i == train_iters // 2:
    #             lr = self.lr*0.1
    #             optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)#gcn parameters!!!!!!!!!!!

    #         self.train()
    #         optimizer.zero_grad()
    #         output = self.forward(self.features, self.adj_norm)#use condensed graph and gcn to fit gcn
    #         loss_train = self.loss(output, labels)
    #         loss_train.backward()
    #         optimizer.step()#fit gcn para

    #         if  i % 20 == 0:
    #             with torch.no_grad():
    #                 self.eval()
    #                 output = self.inference(feat_full, adj_full).to(self.device)#features、adj:numpy
    #                 acc_full = utils.accuracy(output, labels_full)

    #                 if acc_full > best_acc_val:
    #                     best_acc_val = acc_full
    #                     self.output = output
    #                     weights = deepcopy(self.state_dict())#state_dict是在定义了model或optimizer之后pytorch自动生成的,即model此时的参数
    #     del feat_full, labels_full, adj_full
    #     gc.collect()

    #     if verbose:
    #         print('=== picking the best model according to the performance on validation ===')
    #     self.load_state_dict(weights)
        
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
    def predict(self, x, adj):#align format and forward
        model.eval()
        for ix, layer in enumerate(self.layers):
            x = layer.forward(x, adj)#layer是一个GraphConvolution layer
            if ix != len(self.layers) - 1:#最后一层不dropout和BN
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)
    
    @torch.no_grad()
    def inference(self, x_all, adj, device):#subgraph
        #x_all:numpy/tensor adj:SparseTensor all on cpu
        self.eval()
        subgraph_loader = NeighborSampler(adj, node_idx=None, sizes=[-1],#要求adj是SparseTensor或者EdgeIndex
                                  batch_size=100000, shuffle=False,
                                  num_workers=12)
        for ix, layer in enumerate(self.layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(self.device)#edge_index还是SparseTensor或者EdgeIndex
                if torch.is_tensor(x_all):
                    x = x_all[n_id].to(self.device)
                else:
                    x = torch.FloatTensor(x_all[n_id]).to(self.device)  #使用上一层卷积跟新过的embedding, x_all
                x = layer(x, edge_index)
                if ix != self.nlayers - 1:
                    x = self.bns[ix](x) if self.with_bn else x
                    if self.with_relu:
                        x = F.relu(x)
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

