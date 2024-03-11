import numpy as np
import random
import time
import argparse
import time
import deeprobust.graph.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
import torch_sparse
import os 
import gc
import math

from utils import *
from torch_sparse import SparseTensor
from copy import deepcopy
from torch_geometric.utils import coalesce

from models.basicgnn import GCN as GCN_PYG, GIN as GIN_PYG, SGC as SGC_PYG, GraphSAGE as SAGE_PYG, JKNet as JKNet_PYG
from models.mlp import MLP as MLP_PYG
from models.parametrized_adj import PGE


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=1, help='gpu id')
parser.add_argument('--parallel_gpu_ids', type=list, default=[0,1], help='gpu id')#用于batch训练，每个batch加载在不同gpu跑
parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
#gnn
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--inference', type=bool, default=False)
parser.add_argument('--teacher_model', type=str, default='GCN')
parser.add_argument('--lr_teacher_model', type=float, default=0.01)#arxiv:0.01 cora:0.001 pubmed:0.001
parser.add_argument('--save', type=int, default=1)
#loop and validation
parser.add_argument('--teacher_model_loop', type=int, default=600)
parser.add_argument('--teacher_val_stage', type=int, default=50)
args = parser.parse_args()
print(args)

device='cuda'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
torch.cuda.set_device(args.gpu_id)
print("Let's use", torch.cuda.device_count(), "GPUs!")

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def generate_labels_syn():
    from collections import Counter
    counter = Counter(labels.cpu().numpy())#每个class进行数数量统计 字典
    num_class_dict = {}

    sorted_counter = sorted(counter.items(), key=lambda x:x[1])#对次数进行排序,每一个元素为{class,n}
    labels_syn = []
    syn_class_indices = {}

    for ix, (c, num) in enumerate(sorted_counter):
        num_class_dict[c] = math.ceil(num * args.reduction_rate)
        syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
        labels_syn += [c] * num_class_dict[c]

    return labels_syn, num_class_dict


#训练大图
def train_teacher():
    start = time.perf_counter() 
    optimizer_origin=torch.optim.Adam(teacher_model.parameters(), lr=args.lr_teacher_model)
    
    #用于minibatch训练的loader
    # if args.nlayers == 1:
    #     sizes = [15]
    # elif args.nlayers == 2:
    #     sizes = [10, 5]
    # elif args.nlayers == 3:
    #     sizes = [15, 10, 5]
    # elif args.nlayers == 4:
    #     sizes = [15, 10, 5, 5]
    # else:
    #     sizes = [15, 10, 5, 5, 5]
    # train_loader=NeighborSampler(adj,#返回的是一个batch的loader，里面可能有很多个batch
    #     node_idx=torch.LongTensor(idx_train),
    #     sizes=[-1,-1,-1],#可以适当调整下，不一定要全-1 
    #     batch_size=args.batch_size,#越小越久
    #     num_workers=12, 
    #     return_e_id=False,
    #     num_nodes=len(labels),
    #     shuffle=False
    # )

    best_val=0
    best_test=0
    for it in range(args.teacher_model_loop+1):
        #whole graph
        teacher_model.train()
        optimizer_origin.zero_grad()
        output = teacher_model.forward(feat.to(device), adj.to(device))[idx_train]
        loss = F.nll_loss(output, labels_train)
        loss.backward()
        optimizer_origin.step()

        #subgraph 
        # teacher_model.train()
        # loss=torch.tensor(0.0).to(device)
        # start = time.perf_counter()
        # for batch_size, n_id, adjs in train_loader:
        #     if args.nlayers == 1:
        #         adjs = [adjs]
        #     adjs = [adj.to(device) for adj in adjs]
        #     optimizer_origin.zero_grad()
        #     output = teacher_model.forward_sampler(feat[n_id].to(device), adjs)
        #     loss = F.nll_loss(output, labels[n_id[:batch_size]])
        #     loss.backward()
        #     optimizer_origin.step()
        # end = time.perf_counter()
        # print('Epoch',it,'用时:',end-start, '秒')

        if(it%args.teacher_val_stage==0):
            if args.inference==True:
                output = teacher_model.inference(feat, inference_loader, device)
            else:
                output = teacher_model.predict(feat.to(device), adj.to(device))
            acc_train = utils.accuracy(output[idx_train], labels_train)
            acc_val = utils.accuracy(output[idx_val], labels_val)
            acc_test = utils.accuracy(output[idx_test], labels_test)
            print(f'Epoch: {it:02d}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Train: {100 * acc_train.item():.2f}%, '
                    f'Valid: {100 * acc_val.item():.2f}% '
                    f'Test: {100 * acc_test.item():.2f}%')
            if(acc_val>best_val):
                best_val=acc_val
                best_test=acc_test
                if args.save:
                    torch.save(teacher_model.state_dict(), f'{root}/saved_model/teacher/{args.dataset}_{args.teacher_model}_{args.nlayers}_{args.hidden}_{args.dropout}_{args.activation}_{args.seed}.pt')
        
    end = time.perf_counter()
    print("Best Test:", best_test)
    print('大图训练用时:', round(end-start), '秒')
    return


def test_nas():
    if args.teacher_model=='GCN':#每个模型有自己的一套参数设置，teacher和student一样
        teacher_model = GCN_PYG(nfeat=d, nhid=512, nclass=nclass, dropout=0.3, nlayers=3, norm='BatchNorm', act='sigmoid').to(device)
    elif args.teacher_model=='SGC':
        teacher_model = SGC_PYG(nfeat=d, nhid=512, nclass=nclass, dropout=0, nlayers=3, norm=None, sgc=True).to(device)
    else:
        teacher_model = SAGE_PYG(nfeat=d, nhid=512, nclass=nclass, dropout=0.3, nlayers=3, norm='BatchNorm', act='elu').to(device)   

    start = time.perf_counter() 
    optimizer_origin=torch.optim.Adam(teacher_model.parameters(), lr=args.lr_teacher_model)

    best_val=0
    best_test=0
    for it in range(args.teacher_model_loop+1):
        #whole graph
        teacher_model.train()
        optimizer_origin.zero_grad()
        output = teacher_model.forward(feat.to(device), adj.to(device))[idx_train]
        loss = F.nll_loss(output, labels_train)
        loss.backward()
        optimizer_origin.step()

        if(it%args.teacher_val_stage==0):
            if args.inference==True:
                output = teacher_model.inference(feat, inference_loader, device)
            else:
                output = teacher_model.predict(feat.to(device), adj.to(device))
            acc_train = utils.accuracy(output[idx_train], labels_train)
            acc_val = utils.accuracy(output[idx_val], labels_val)
            acc_test = utils.accuracy(output[idx_test], labels_test)
            print(f'Epoch: {it:02d}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Train: {100 * acc_train.item():.2f}%, '
                    f'Valid: {100 * acc_val.item():.2f}% '
                    f'Test: {100 * acc_test.item():.2f}%')
            if(acc_val>best_val):
                best_val=acc_val
                best_test=acc_test
                if args.save:
                    torch.save(teacher_model.state_dict(), f'{root}/saved_model/teacher/{args.dataset}_{args.teacher_model}_3_512_0.3_sigmoid_{args.seed}.pt')#根据实际的最好模型修改后缀
        
    end = time.perf_counter()
    print("Best Test:", best_test)
    print('大图训练用时:', round(end-start), '秒')
    return


if __name__ == '__main__':
    #获得大图的所有数据
    root=os.path.abspath(os.path.dirname(__file__))
    data = get_dataset(args.dataset, args.normalize_features)#get a Pyg2Dpr class, contains all index, adj, labels, features
    adj, feat=utils.to_tensor(data.adj, data.features, device='cpu')
    labels=torch.LongTensor(data.labels).to(device)
    idx_train, idx_val, idx_test=data.idx_train, data.idx_val, data.idx_test
    labels_train, labels_val, labels_test=labels[idx_train], labels[idx_val], labels[idx_test]
    d = feat.shape[1]
    nclass= int(labels.max()+1)
    del data
    gc.collect()

    if utils.is_sparse_tensor(adj):
        adj = utils.normalize_adj_tensor(adj, sparse=True)
    else:
        adj = utils.normalize_adj_tensor(adj)
    adj=SparseTensor(row=adj._indices()[0], col=adj._indices()[1],value=adj._values(), sparse_sizes=adj.size()).t()
    #用于inference的dataloader，inference不需要放在gpu上，且inference次数不多，sizes可以选-1
    if args.inference:
        inference_loader=NeighborSampler(adj,#返回的是一个batch的loader，里面可能有很多个batch
            sizes=[-1], 
            batch_size=args.batch_size,
            num_workers=12, 
            return_e_id=False,
            num_nodes=len(labels),
            shuffle=False
        )

    #teacher_model
    if args.teacher_model=='GCN':#每个模型有自己的一套参数设置，teacher和student一样
        teacher_model = GCN_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm='BatchNorm', act=args.activation).to(device)
    elif args.teacher_model=='SGC':
        teacher_model = SGC_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm=None, sgc=True, act=args.activation).to(device)
    else:
        teacher_model = SAGE_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm='BatchNorm', act=args.activation).to(device)  

    print("训练大图！")
    train_teacher()
    teacher_model.load_state_dict(torch.load(f'{root}/saved_model/teacher/{args.dataset}_{args.teacher_model}_{args.nlayers}_{args.hidden}_{args.dropout}_{args.activation}_{args.seed}.pt'))
    if args.inference==True:
        output = teacher_model.inference(feat, inference_loader, device)
    else:
        output = teacher_model.predict(feat.to(device), adj.to(device))
    acc_train = utils.accuracy(output[idx_train], labels_train)
    acc_val = utils.accuracy(output[idx_val], labels_val)
    acc_test = utils.accuracy(output[idx_test], labels_test)
    print(f'Teacher model results:,'
            f'Train: {100 * acc_train.item():.2f}% '
            f'Valid: {100 * acc_val.item():.2f}% '
            f'Test: {100 * acc_test.item():.2f}%')
