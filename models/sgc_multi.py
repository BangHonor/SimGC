import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from models.mlp import MLP 
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm

class SGC_Multi(MessagePassing):
    r"""The simple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper

    .. math::
        \mathbf{X}^{\prime} = {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        nfeat (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        nclass (int): Size of each output sample.
        K (int, optional): Number of hops :math:`K`. (default: :obj:`1`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X}` on
            first execution, and will use the cached version for further
            executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:**
          node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_x: Optional[Tensor]

    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout, K: int = 2, nlayers: int = 2, norm: Union[str, Callable, None] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.nfeat = nfeat
        self.nclass = nclass
        self.K = K
        # self.mlp = MLP(channel_list=[nfeat, 256, 256, nclass], nfeat=nfeat, nhid=256, nclass=nclass, dropout=[0.5, 0.5, 0.5], num_layers=3, norm='BatchNorm')
        channel_list=[nfeat]
        channel_list.extend([nhid]*(nlayers-1))
        channel_list.append(nclass)
        self.mlp = MLP(channel_list=channel_list, nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=[dropout]*nlayers, num_layers=nlayers, norm=norm)

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.initialize()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        for k in range(self.K):#只能用于full-batch，mini-batch显然连续propagate不合适
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                size=None)

        return self.mlp(x)

    def forward_sampler(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        for i, (adj, _, size) in enumerate(edge_index):
            x = self.propagate(adj, x=x, edge_weight=edge_weight)

        return self.mlp(x)
    
    @torch.no_grad()
    def predict(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        self.eval()
        return self.forward(x, edge_index, edge_weight=edge_weight)
    
    @torch.no_grad()
    def inference(self, x_all: Tensor, loader: NeighborSampler,
                  device: Optional[torch.device] = None,
                  progress_bar: bool = False) -> Tensor:
        self.eval()
        for i in range(self.K):
            xs: List[Tensor] = []
            for batch_size, n_id, adj in loader:
                x = x_all[n_id].to(device)#全局index
                edge_index = adj.adj_t.to(device)#只有SparseTensor而没有edge_index
                x = self.propagate(edge_index, x=x)[:batch_size]
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        x_all=self.mlp(x_all.to(device))
        return F.log_softmax(x_all,dim=1).to(device)
    
    
    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.nfeat}, '
                f'{self.nclass}, K={self.K})')
