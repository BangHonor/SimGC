import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch_geometric.typing import Adj, OptTensor

from torch_geometric.loader import NeighborSampler
from torch_geometric.nn.conv import (
    EdgeConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    MessagePassing,
    PNAConv,
    SAGEConv,
    SGConv,
    APPNP
)
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)

class BasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        nfeat (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        nhid (int): Size of each hidden sample.
        nlayers (int): Number of message passing layers.
        nclass (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`nclass`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nlayers: int,
        nclass: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        sgc: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.temp_layers = nlayers
        if sgc==False:
            self.nlayers = nlayers
        else:
            self.nlayers = 1    
            nlayers = 1

        self.dropout = dropout
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.act_first = act_first
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs
        self.sgc=sgc

        if nclass is not None:
            self.nclass = nclass
        else:
            self.nclass = nhid

        self.convs = ModuleList()
        if nlayers > 1:
            self.convs.append(
                self.init_conv(nfeat, nhid, **kwargs))
            if isinstance(nfeat, (tuple, list)):
                nfeat = (nhid, nhid)
            else:
                nfeat = nhid
        for _ in range(nlayers - 2):
            self.convs.append(
                self.init_conv(nfeat, nhid, **kwargs))
            if isinstance(nfeat, (tuple, list)):
                nfeat = (nhid, nhid)
            else:
                nfeat = nhid
        if nclass is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(nfeat, nclass, **kwargs))
        else:
            self.convs.append(
                self.init_conv(nfeat, nhid, **kwargs))

        self.norms = None
        if norm is not None and sgc==False:
            norm_layer = normalization_resolver(
                norm,
                nhid,
                **(norm_kwargs or {}),
            )
            self.norms = ModuleList()
            for _ in range(nlayers - 1):
                self.norms.append(copy.deepcopy(norm_layer))
            if jk is not None:
                self.norms.append(copy.deepcopy(norm_layer))

        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, nhid, nlayers)

        if jk is not None:
            if jk == 'cat':
                nfeat = nlayers * nhid
            else:
                nfeat = nhid
            self.lin = Linear(nfeat, self.nclass)

    def init_conv(self, nfeat: Union[int, Tuple[int, int]],
                  nclass: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def initialize(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        *,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
    ) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
        """
        xs: List[Tensor] = []
        for i in range(self.nlayers):
            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight,
                                  edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            if i == self.nlayers - 1 and self.jk_mode is None:
                break
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x

        return F.log_softmax(x,dim=1)
    
    def forward_sampler(self, x, adjs):
        if self.sgc==False:
            for i, (adj, _, size) in enumerate(adjs):
                x = self.convs[i](x, adj)#每一层采样的节点不一样，邻接矩阵也不一样
                if i != self.nlayers - 1:
                    x = self.norms[i](x)
                    x = self.act(x)
                    x = F.dropout(x, self.dropout, training=self.training)
        else:#只有一层conv
            x = self.convs[0].forward_sampler(x, adjs)
        
        return F.log_softmax(x,dim=1)
        
    @torch.no_grad()
    def predict(
        self,
        x: Tensor,
        edge_index: Adj,
        *,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
    ) -> Tensor:

        self.eval()
        return self.forward(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)

    @torch.no_grad()
    def inference(self, x_all: Tensor, loader: NeighborSampler,
                  device: Optional[torch.device] = None,
                  progress_bar: bool = False) -> Tensor:
        self.eval()
        assert self.jk_mode is None or self.jk_mode == 'last'
        assert isinstance(loader, NeighborSampler)
        assert not self.training
        # assert not loader.shuffle  # TODO (matthias) does not work :(
        if progress_bar:
            pbar = tqdm(total=len(self.convs) * len(loader))
            pbar.set_description('Inference')

        for i in range(self.temp_layers):
            xs: List[Tensor] = []
            for batch_size, n_id, adj in loader:
                x = x_all[n_id].to(device)#全局index
                edge_index = adj.adj_t.to(device)#只有SparseTensor而没有edge_index
                if self.sgc==False:
                    x = self.convs[i](x, edge_index)[:batch_size]
                else:
                    x = self.convs[0].propagate(edge_index, x=x)[:batch_size]
                    if i==self.temp_layers-1:
                        x=self.convs[0].lin(x)
                if self.sgc==False:
                    if i == self.nlayers - 1 and self.jk_mode is None:
                        xs.append(x.cpu())
                        if progress_bar:
                            pbar.update(1)
                        continue
                    if self.act is not None and self.act_first:
                        x = self.act(x)
                    if self.norms is not None:
                        x = self.norms[i](x)
                    if self.act is not None and not self.act_first:
                        x = self.act(x)
                    if i == self.nlayers - 1 and hasattr(self, 'lin'):
                        x = self.lin(x)
                xs.append(x.cpu())
                if progress_bar:
                    pbar.update(1)
            x_all = torch.cat(xs, dim=0)
                
        if progress_bar:
            pbar.close()

        return F.log_softmax(x_all,dim=1).to(device)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.nfeat}, '
                f'{self.nclass}, nlayers={self.nlayers})')


class GCN(BasicGNN):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.

    Args:
        nfeat (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        nhid (int): Size of each hidden sample.
        nlayers (int): Number of message passing layers.
        nclass (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`nclass`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GCNConv`.
    """
    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(self, nfeat: int, nclass: int,
                  **kwargs) -> MessagePassing:
        return GCNConv(nfeat, nclass, **kwargs)


class GraphSAGE(BasicGNN):
    r"""The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~torch_geometric.nn.SAGEConv` operator for message passing.

    Args:
        nfeat (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        nhid (int): Size of each hidden sample.
        nlayers (int): Number of message passing layers.
        nclass (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`nclass`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,    
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """
    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(self, nfeat: Union[int, Tuple[int, int]],
                  nclass: int, **kwargs) -> MessagePassing:
        return SAGEConv(nfeat, nclass, project=False, **kwargs)


class GIN(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.

    Args:
        nfeat (int): Size of each input sample.
        nhid (int): Size of each hidden sample.
        nlayers (int): Number of message passing layers.
        nclass (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`nclass`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """
    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(self, nfeat: int, nclass: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [nfeat, nclass, nclass],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINConv(mlp, train_eps=True, **kwargs)#不大适合add因为邻居个数变化非常大


class GAT(BasicGNN):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATConv` or
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
    respectively.

    Args:
        nfeat (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        nhid (int): Size of each hidden sample.
        nlayers (int): Number of message passing layers.
        nclass (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`nclass`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.GATv2Conv`.
    """
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, nfeat: Union[int, Tuple[int, int]],
                  nclass: int, **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (nclass != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and nclass % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{nclass}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            nclass = nclass // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(nfeat, nclass, heads=heads, concat=concat,
                    dropout=self.dropout, **kwargs)


class PNA(BasicGNN):
    r"""The Graph Neural Network from the `"Principal Neighbourhood Aggregation
    for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper, using the
    :class:`~torch_geometric.nn.conv.PNAConv` operator for message passing.

    Args:
        nfeat (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        nhid (int): Size of each hidden sample.
        nlayers (int): Number of message passing layers.
        nclass (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`nclass`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.PNAConv`.
    """
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, nfeat: int, nclass: int,
                  **kwargs) -> MessagePassing:
        return PNAConv(nfeat, nclass, **kwargs)


class EdgeCNN(BasicGNN):
    r"""The Graph Neural Network from the `"Dynamic Graph CNN for Learning on
    Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper, using the
    :class:`~torch_geometric.nn.conv.EdgeConv` operator for message passing.

    Args:
        nfeat (int): Size of each input sample.
        nhid (int): Size of each hidden sample.
        nlayers (int): Number of message passing layers.
        nclass (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`nclass`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.EdgeConv`.
    """
    supports_edge_weight = False
    supports_edge_attr = False

    def init_conv(self, nfeat: int, nclass: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [2 * nfeat, nclass, nclass],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return EdgeConv(mlp, **kwargs)


class SGC(BasicGNN):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.

    Args:
        nfeat (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        nhid (int): Size of each hidden sample.
        nlayers (int): Number of message passing layers.
        nclass (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`nclass`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GCNConv`.
    """
    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(self, nfeat: int, nclass: int,
                  **kwargs) -> MessagePassing:
        return SGConv(nfeat, nclass, self.temp_layers, **kwargs)
    

class JKNet(BasicGNN):
    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(self, nfeat: int, nclass: int,
                  **kwargs) -> MessagePassing:
        return GCNConv(nfeat, nclass, **kwargs)
    

__all__ = ['GCN', 'GraphSAGE', 'GIN', 'GAT', 'PNA', 'EdgeCNN', 'SGC', 'JKNet']