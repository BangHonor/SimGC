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
from models.sgc_multi import SGC_Multi as SGC_Multi_PYG
from models.parametrized_adj import PGE


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=1, help='gpu id')
parser.add_argument('--parallel_gpu_ids', type=list, default=[0,1], help='gpu id')
parser.add_argument('--dataset', type=str, default='reddit2')
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
parser.add_argument('--teacher_model', type=str, default='SGC')
parser.add_argument('--validation_model', type=str, default='GCN')
parser.add_argument('--model', type=str, default='GCN')
#ratio
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=0.001)
#condensation
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.05)
parser.add_argument('--lr_model', type=float, default=0.001)
parser.add_argument('--lr_teacher_model', type=float, default=0.01)
parser.add_argument('--alignment', type=int, default=1)
parser.add_argument('--feat_alpha', type=float, default=100, help='feat loss term.')
parser.add_argument('--smoothness', type=int, default=1)
parser.add_argument('--smoothness_alpha', type=float, default=0.1, help='smoothness loss term.')
parser.add_argument('--threshold', type=float, default=0.01, help='adj threshold.')
parser.add_argument('--save', type=int, default=1)
#loop and validation
parser.add_argument('--teacher_model_loop', type=int, default=600)
parser.add_argument('--condensing_loop', type=int, default=2500)#reddit+reddit2:2500 flickr:3500
parser.add_argument('--student_model_loop', type=int, default=3000)#SAGE:5000 else:3000
parser.add_argument('--teacher_val_stage', type=int, default=50)
parser.add_argument('--student_val_stage', type=int, default=100)
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
    counter = Counter(data.labels_train)
    num_class_dict = {}

    sorted_counter = sorted(counter.items(), key=lambda x:x[1])
    labels_syn = []
    syn_class_indices = {}
    idx_selected = []

    for ix, (c, num) in enumerate(sorted_counter):
        num_class_dict[c] = math.ceil(num * args.reduction_rate)
        syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
        labels_syn += [c] * num_class_dict[c]

        index = (data.labels_train == c)
        index = np.arange(len(data.labels_train))[index]
        tmp =  np.random.permutation(index)[:num_class_dict[c]]
        tmp = list(tmp)
        idx_selected = idx_selected + tmp

    features = feat_train[idx_selected]

    return features, labels_syn, num_class_dict


def generate_labels_syn():
    from collections import Counter
    counter = Counter(labels_train.cpu().numpy())
    num_class_dict = {}

    sorted_counter = sorted(counter.items(), key=lambda x:x[1])
    labels_syn = []
    syn_class_indices = {}

    for ix, (c, num) in enumerate(sorted_counter):
        num_class_dict[c] = math.ceil(num * args.reduction_rate)
        syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
        labels_syn += [c] * num_class_dict[c]

    return labels_syn, num_class_dict


#training teacher model
def train_teacher():
    start = time.perf_counter() 
    optimizer_origin=torch.optim.Adam(teacher_model.parameters(), lr=args.lr_teacher_model)

    best_val=0
    best_test=0
    for it in range(args.teacher_model_loop+1):
        #whole graph
        teacher_model.train()
        optimizer_origin.zero_grad()
        output = teacher_model.forward(feat_train.to(device), adj_train.to(device))
        loss = F.nll_loss(output, labels_train)
        loss.backward()
        optimizer_origin.step()

        if(it%args.teacher_val_stage==0):
            acc_train = utils.accuracy(output, labels_train)

            if args.inference==True:
                output = teacher_model.inference(feat_val, inference_loader_val, device)
            else:
                output = teacher_model.predict(feat_val.to(device), adj_val.to(device))
            acc_val = utils.accuracy(output, labels_val)

            if args.inference==True:
                output = teacher_model.inference(feat_test, inference_loader_test, device)
            else:
                output = teacher_model.predict(feat_test.to(device), adj_test.to(device))
            acc_test = utils.accuracy(output, labels_test)

            print(f'Epoch: {it:02d}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Train: {100 * acc_train.item():.2f}% '
                    f'Valid: {100 * acc_val.item():.2f}% '
                    f'Test: {100 * acc_test.item():.2f}%')
            if(acc_val>best_val):
                best_val=acc_val
                best_test=acc_test
                torch.save(teacher_model.state_dict(), f'{root}/saved_model/teacher/{args.dataset}_{args.teacher_model}_{args.seed}.pt')
        
    end = time.perf_counter()
    print("Best Test:", best_test)
    print('Training teacher model:', round(end-start), 's')
    return


def train_syn():
    start = time.perf_counter()
    if args.validation_model=='GCN':
        validation_model = GCN_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm='BatchNorm').to(device)
    elif args.validation_model=='SGC':
        validation_model = SGC_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, nlayers=args.nlayers, sgc=True).to(device)
    optimizer = optim.Adam(validation_model.parameters(), lr=args.lr_model)
    optimizer_feat = optim.Adam([feat_syn], lr=args.lr_feat)
    optimizer_pge = optim.Adam(pge.parameters(), lr=args.lr_adj)

    #alignment
    concat_feat=feat_train.to(device)
    temp=feat_train
    for i in range(args.nlayers):
        aggr=teacher_model.convs[0].propagate(adj_train.to(device), x=temp.to(device)).detach()
        concat_feat=torch.cat((concat_feat,aggr),dim=1)
        temp=aggr

    concat_feat_mean=[]
    concat_feat_std=[]
    coeff=[]
    coeff_sum=0
    for c in range(nclass):
        if c in num_class_dict:
            index = torch.where(labels_train==c)
            coe = num_class_dict[c] / max(num_class_dict.values())
            coeff_sum+=coe
            coeff.append(coe)
            concat_feat_mean.append(concat_feat[index].mean(dim=0).to(device))
            concat_feat_std.append(concat_feat[index].std(dim=0).to(device))
        else:
            coeff.append(0)
            concat_feat_mean.append([])
            concat_feat_std.append([])
    coeff_sum=torch.tensor(coeff_sum).to(device)
    
    best_val=0
    best_test=0
    for i in range(args.condensing_loop+1):
        teacher_model.eval()
        optimizer_pge.zero_grad()
        optimizer_feat.zero_grad()
        
        adj_syn = pge(feat_syn).to(device)
        adj_syn[adj_syn<args.threshold]=0
        edge_index_syn = torch.nonzero(adj_syn).T
        edge_weight_syn = adj_syn[edge_index_syn[0], edge_index_syn[1]]

        # smoothness loss
        feat_difference=torch.exp(-0.5 * torch.pow(feat_syn[edge_index_syn[0]]-feat_syn[edge_index_syn[1]], 2))
        smoothness_loss=-torch.dot(edge_weight_syn,torch.mean(feat_difference,1).flatten())/torch.sum(edge_weight_syn)

        edge_index_syn, edge_weight_syn = gcn_norm(edge_index_syn, edge_weight_syn, n)
        concat_feat_syn=feat_syn.to(device)
        temp=feat_syn
        for j in range(args.nlayers):
            aggr_syn=teacher_model.convs[0].propagate(edge_index_syn, x=temp, edge_weight=edge_weight_syn, size=None)
            concat_feat_syn=torch.cat((concat_feat_syn,aggr_syn),dim=1)
            temp=aggr_syn

        #inversion loss
        output_syn = teacher_model.forward(feat_syn, edge_index_syn, edge_weight=edge_weight_syn)
        hard_loss = F.nll_loss(output_syn, labels_syn)

        #alignment loss
        # concat_feat_syn=output_syn
        concat_feat_loss=torch.tensor(0.0).to(device)
        loss_fn=nn.MSELoss()
        for c in range(nclass):
            if c in num_class_dict:
                index=torch.where(labels_syn==c)
                concat_feat_mean_loss=coeff[c]*loss_fn(concat_feat_mean[c],concat_feat_syn[index].mean(dim=0))
                concat_feat_std_loss=coeff[c]*loss_fn(concat_feat_std[c],concat_feat_syn[index].std(dim=0))
                if feat_syn[index].shape[0]!=1:
                    concat_feat_loss+=(concat_feat_mean_loss+concat_feat_std_loss)
                else:
                    concat_feat_loss+=(concat_feat_mean_loss)
        concat_feat_loss=concat_feat_loss/coeff_sum

        #total loss
        loss=hard_loss+args.feat_alpha*concat_feat_loss+args.smoothness_alpha*smoothness_loss
        loss.backward()
        if i%50<10:
            optimizer_pge.step()
        else:
            optimizer_feat.step()

        if i>=100 and i%100==0:
            adj_syn=pge.inference(feat_syn).detach().to(device)
            adj_syn[adj_syn<args.threshold]=0
            adj_syn.requires_grad=False
            edge_index_syn=torch.nonzero(adj_syn).T
            edge_weight_syn= adj_syn[edge_index_syn[0], edge_index_syn[1]]
            edge_index_syn, edge_weight_syn=gcn_norm(edge_index_syn, edge_weight_syn, n)

            teacher_output_syn = teacher_model.predict(feat_syn, edge_index_syn, edge_weight=edge_weight_syn)
            acc = utils.accuracy(teacher_output_syn, labels_syn)
            print('Epoch {}'.format(i),"Teacher on syn accuracy= {:.4f}".format(acc.item()))
        
            validation_model.initialize()
            for j in range(args.student_model_loop):
                validation_model.train()
                optimizer.zero_grad()
                output_syn = validation_model.forward(feat_syn, edge_index_syn, edge_weight=edge_weight_syn)
                loss=F.nll_loss(output_syn, labels_syn)
                loss.backward()
                optimizer.step()

                if j%args.student_val_stage==0:
                    if args.inference==True:
                        output = validation_model.inference(feat_val, inference_loader_val, device)
                    else:
                        output = validation_model.predict(feat_val.to(device), adj_val.to(device))
                    acc_val = utils.accuracy(output, labels_val)

                    if args.inference==True:
                        output = validation_model.inference(feat_test, inference_loader_test, device)
                    else:
                        output = validation_model.predict(feat_test.to(device), adj_test.to(device))
                    acc_test = utils.accuracy(output, labels_test)
                    if(acc_val>best_val):
                        best_val=acc_val
                        best_test=acc_test
                        if args.save:
                            if args.alignment == 1 and args.smoothness == 1:
                                torch.save(feat_syn, f'{root}/saved_ours/feat_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt')
                                torch.save(pge.state_dict(), f'{root}/saved_ours/pge_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt')
                            if args.alignment == 1 and args.smoothness == 0:
                                torch.save(feat_syn, f'{root}/saved_ours/feat_without_smoothness_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt')
                                torch.save(pge.state_dict(), f'{root}/saved_ours/pge_without_smoothness_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt')
                            if args.alignment == 0 and args.smoothness == 1:
                                torch.save(feat_syn, f'{root}/saved_ours/feat_without_alignment_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt')
                                torch.save(pge.state_dict(), f'{root}/saved_ours/pge_without_alignment_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt')
                            if args.alignment == 0 and args.smoothness == 0:
                                torch.save(feat_syn, f'{root}/saved_ours/feat_without_both_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt')
                                torch.save(pge.state_dict(), f'{root}/saved_ours/pge_without_both_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt')
                            torch.save(validation_model.state_dict(), f'{root}/saved_model/student/{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_3_256_0.5_relu_1.pt')

            print('Epoch {}'.format(i), "Best test acc:",best_test)

    end = time.perf_counter()
    print('Condensation Duration:',round(end-start), 's')
    print("Best test acc:",best_test)


def test_nas():
    if args.teacher_model=='GCN':
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
        output = teacher_model.forward(feat_train.to(device), adj_train.to(device))
        loss = F.nll_loss(output, labels_train)
        loss.backward()
        optimizer_origin.step()

        if(it%args.teacher_val_stage==0):
            acc_train = utils.accuracy(output, labels_train)

            if args.inference==True:
                output = teacher_model.inference(feat_val, inference_loader_val, device)
            else:
                output = teacher_model.predict(feat_val.to(device), adj_val.to(device))
            acc_val = utils.accuracy(output, labels_val)

            if args.inference==True:
                output = teacher_model.inference(feat_test, inference_loader_test, device)
            else:
                output = teacher_model.predict(feat_test.to(device), adj_test.to(device))
            acc_test = utils.accuracy(output, labels_test)
            print(f'Epoch: {it:02d}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Train: {100 * acc_train.item():.2f}% '
                    f'Valid: {100 * acc_val.item():.2f}% '
                    f'Test: {100 * acc_test.item():.2f}%')
            if(acc_val>best_val):
                best_val=acc_val
                best_test=acc_test
                torch.save(teacher_model.state_dict(), f'{root}/saved_model/teacher/{args.dataset}_{args.teacher_model}_3_512_0.3_sigmoid_{args.seed}.pt')
        
    end = time.perf_counter()
    print("Best Test:", best_test)
    return


if __name__ == '__main__':
    root=os.path.abspath(os.path.dirname(__file__))
    data = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data, keep_ratio=args.keep_ratio)

    adj_train, feat_train=utils.to_tensor(data.adj_train, data.feat_train, device='cpu')
    adj_val, feat_val=utils.to_tensor(data.adj_val, data.feat_val, device='cpu')
    adj_test, feat_test=utils.to_tensor(data.adj_test, data.feat_test, device='cpu')
    labels=torch.LongTensor(data.labels).to(device)
    labels_train=torch.LongTensor(data.labels_train).to(device)
    labels_val=torch.LongTensor(data.labels_val).to(device)
    labels_test=torch.LongTensor(data.labels_test).to(device)

    print(adj_train._indices().cpu().numpy().nbytes/2+labels_train.cpu().numpy().nbytes+feat_train.cpu().numpy().nbytes)
    
    d = feat_train.shape[1]
    nclass= int(labels.max()+1)
    # del data
    # gc.collect()

    if utils.is_sparse_tensor(adj_train):
        adj_train = utils.normalize_adj_tensor(adj_train, sparse=True)
        adj_val = utils.normalize_adj_tensor(adj_val, sparse=True)
        adj_test = utils.normalize_adj_tensor(adj_test, sparse=True)
    else:
        adj_train = utils.normalize_adj_tensor(adj_train)
        adj_val = utils.normalize_adj_tensor(adj_val)
        adj_test = utils.normalize_adj_tensor(adj_test)
    adj_train=SparseTensor(row=adj_train._indices()[0], col=adj_train._indices()[1],value=adj_train._values(), sparse_sizes=adj_train.size()).t()
    adj_val=SparseTensor(row=adj_val._indices()[0], col=adj_val._indices()[1],value=adj_val._values(), sparse_sizes=adj_val.size()).t()
    adj_test=SparseTensor(row=adj_test._indices()[0], col=adj_test._indices()[1],value=adj_test._values(), sparse_sizes=adj_test.size()).t()
    
    if args.inference:
        inference_loader_train=NeighborSampler(adj_train,
            sizes=[-1], 
            batch_size=args.batch_size,
            num_workers=12, 
            return_e_id=False,
            num_nodes=len(labels_val),
            shuffle=False
        )
        inference_loader_val=NeighborSampler(adj_val,
            sizes=[-1], 
            batch_size=args.batch_size,
            num_workers=12, 
            return_e_id=False,
            num_nodes=len(labels_val),
            shuffle=False
        )
        inference_loader_test=NeighborSampler(adj_test,
            sizes=[-1], 
            batch_size=args.batch_size,
            num_workers=12, 
            return_e_id=False,
            num_nodes=len(labels_test),
            shuffle=False
        )

    #teacher_model
    if args.teacher_model=='GCN':
        teacher_model = GCN_PYG(nfeat=d, nhid=256, nclass=nclass, dropout=0.5, nlayers=2, norm='BatchNorm').to(device)
    elif args.teacher_model=='SGC':
        teacher_model = SGC_PYG(nfeat=d, nhid=256, nclass=nclass, dropout=0, nlayers=2, norm=None, sgc=True).to(device)
    else:
        teacher_model = SGC_Multi_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, K=args.nlayers, nlayers=3, norm='BatchNorm').to(device)  

    if not os.path.exists(root+'/saved_model/teacher/'+args.dataset+'_'+args.teacher_model+'_'+str(args.seed)+'.pt'):
        print("Traning Teacher!")
        train_teacher()
    teacher_model.load_state_dict(torch.load(f'{root}/saved_model/teacher/{args.dataset}_{args.teacher_model}_{args.seed}.pt'))

    if args.inference==True:
        output_train = teacher_model.inference(feat_train, inference_loader_train, device)
        output_val = teacher_model.inference(feat_val, inference_loader_val, device)
        output_test = teacher_model.inference(feat_test, inference_loader_test, device)
    else:
        output_train = teacher_model.predict(feat_train.to(device), adj_train.to(device))
        output_val = teacher_model.predict(feat_val.to(device), adj_val.to(device))
        output_test = teacher_model.predict(feat_test.to(device), adj_test.to(device))
    acc_train = utils.accuracy(output_train, labels_train)
    acc_val = utils.accuracy(output_val, labels_val)
    acc_test = utils.accuracy(output_test, labels_test)
    print(f'Teacher model results:,'
            f'Train: {100 * acc_train.item():.2f}% '
            f'Valid: {100 * acc_val.item():.2f}% '
            f'Test: {100 * acc_test.item():.2f}%')
    
    labels_syn, num_class_dict = generate_labels_syn()
    labels_syn = torch.LongTensor(labels_syn).to(device)
    n = len(labels_syn)
    feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
    feat_syn.data.copy_(torch.randn(feat_syn.size()))

    pge = PGE(nfeat=d, nnodes=n, device=device, args=args).to(device)

    del data
    gc.collect()

    if args.alignment == 0:
        args.feat_alpha = 0
    if args.smoothness == 0:
        args.smoothness_alpha = 0
    if args.alignment == 1 and args.smoothness == 1:
        if not os.path.exists(root+'/saved_ours/feat_'+args.dataset+'_'+args.teacher_model+'_'+args.validation_model+'_'+str(args.reduction_rate)+'_'+str(args.seed)+'.pt'):
            print("Condensing!")
            train_syn()
        feat_syn=torch.load(f'{root}/saved_ours/feat_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt').to(device)
        pge.load_state_dict(torch.load(f'{root}/saved_ours/pge_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt'))
    elif args.alignment == 0 and args.smoothness == 1:
        if not os.path.exists(root+'/saved_ours/feat_without_alignment_'+args.dataset+'_'+args.teacher_model+'_'+args.validation_model+'_'+str(args.reduction_rate)+'_'+str(args.seed)+'.pt'):
            print("Condensing!")
            train_syn()
        feat_syn=torch.load(f'{root}/saved_ours/feat_without_alignment_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt').to(device)
        pge.load_state_dict(torch.load(f'{root}/saved_ours/pge_without_alignment_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt'))
    elif args.alignment == 1 and args.smoothness == 0:
        if not os.path.exists(root+'/saved_ours/feat_without_smoothness_'+args.dataset+'_'+args.teacher_model+'_'+args.validation_model+'_'+str(args.reduction_rate)+'_'+str(args.seed)+'.pt'):
            print("Condensing!")
            train_syn()
        feat_syn=torch.load(f'{root}/saved_ours/feat_without_smoothness_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt').to(device)
        pge.load_state_dict(torch.load(f'{root}/saved_ours/pge_without_smoothness_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt'))
    else:
        if not os.path.exists(root+'/saved_ours/feat_without_both_'+args.dataset+'_'+args.teacher_model+'_'+args.validation_model+'_'+str(args.reduction_rate)+'_'+str(args.seed)+'.pt'):
            print("Condensing!")
            train_syn()
        feat_syn=torch.load(f'{root}/saved_ours/feat_without_both_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt').to(device)
        pge.load_state_dict(torch.load(f'{root}/saved_ours/pge_without_both_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}.pt'))   

    #training on the condensed graph
    start = time.perf_counter()
    if args.model=='GCN':
        model = GCN_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm='BatchNorm', act=args.activation).to(device)
    elif args.model=='SGC':
        model = SGC_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=0, nlayers=args.nlayers, sgc=True, act=args.activation).to(device)
    elif args.model=='SAGE':
        model = SAGE_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm='BatchNorm', act=args.activation).to(device)   
    elif args.model=='GIN':
        model = GIN_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm='BatchNorm', act=args.activation).to(device)
    elif args.model=='JKNet':
        model = JKNet_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers+1, norm='BatchNorm', jk='cat', act=args.activation).to(device)
    else:
        model = MLP_PYG(channel_list=[d, args.hidden, nclass], in_channels=d, hidden_channels=args.hidden, out_channels=nclass, dropout=[args.dropout, args.dropout], num_layers=args.nlayers, norm='BatchNorm', act=args.activation).to(device)
    model.initialize()
    optimizer=optim.Adam(model.parameters(), lr=args.lr_model)

    adj_syn=pge.inference(feat_syn).detach().to(device)
    adj_syn[adj_syn<args.threshold]=0
    adj_syn.requires_grad=False
    edge_index_syn=torch.nonzero(adj_syn).T
    edge_weight_syn= adj_syn[edge_index_syn[0], edge_index_syn[1]]
    edge_index_syn, edge_weight_syn=gcn_norm(edge_index_syn, edge_weight_syn, n)
    teacher_output_syn = teacher_model.predict(feat_syn, edge_index_syn, edge_weight=edge_weight_syn)#forward or predict
    acc = utils.accuracy(teacher_output_syn, labels_syn)
    print("Teacher on syn accuracy= {:.4f}".format(acc.item()))
    memory = feat_syn.element_size() * feat_syn.nelement()
    memory1 = edge_index_syn.element_size() * edge_index_syn.nelement()
    memory2 = edge_weight_syn.element_size() * edge_weight_syn.nelement()
    print(memory+memory1+memory2) 

    #training on the condensed graph
    best_val=0
    best_test=0
    for j in range(args.student_model_loop+1):
        model.train()
        optimizer.zero_grad()
        if args.model!='MLP':
            output_syn = model.forward(feat_syn, edge_index_syn, edge_weight=edge_weight_syn)
        else:
            output_syn = model.forward(feat_syn)
        loss=F.nll_loss(output_syn, labels_syn)
        loss.backward()
        optimizer.step()

        if j%args.student_val_stage==0:
            if args.inference==False:
                if args.model!='MLP':
                    output = model.predict(feat_train.to(device), adj_train.to(device))
                else:
                    output = model.predict(feat_train.to(device))
            else:
                output = model.inference(feat_train, inference_loader_train, device)
            acc_train = utils.accuracy(output, labels_train)

            if args.inference==False:
                if args.model!='MLP':
                    output = model.predict(feat_val.to(device), adj_val.to(device))
                else:
                    output = model.predict(feat_val.to(device))
            else:
                output = model.inference(feat_val, inference_loader_val, device)
            acc_val = utils.accuracy(output, labels_val)

            if args.inference==False:
                if args.model!='MLP':
                    output = model.predict(feat_test.to(device), adj_test.to(device))
                else:
                    output = model.predict(feat_test.to(device))
            else:
                output = model.inference(feat_test, inference_loader_test, device)
            acc_test = utils.accuracy(output, labels_test)
            
            print(f'Epoch: {j:02d}, '
                    f'Train: {100 * acc_train.item():.2f}% '
                    f'Valid: {100 * acc_val.item():.2f}% '
                    f'Test: {100 * acc_test.item():.2f}%')
            
            if(acc_val>best_val):
                best_val=acc_val
                best_test=acc_test
                torch.save(model.state_dict(), f'{root}/saved_model/student/{args.dataset}_{args.teacher_model}_{args.model}_{args.reduction_rate}_{args.nlayers}_{args.hidden}_{args.dropout}_{args.activation}_{args.seed}.pt')

    end = time.perf_counter()
    print('Training on the condensed graph:',round(end-start), 's')
    print("Best Test Acc:",best_test)
    
