import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList
from tqdm import tqdm

from torch_geometric.nn.conv import (
    MessagePassing,
)
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj, OptTensor, PairTensor

class APPNP(torch.nn.Module):
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
        nclass: Optional[int] = None,
        nlayers: Optional[int] = 1,
        K: Optional[int] = 5,
        alpha: float = 0.1,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.nlayers=nlayers
        self.K=K
        self.alpha = alpha

        self.dropout = dropout
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs

        self.supports_edge_weight=True
        self.supports_edge_attr=True

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
        self.convs.append(
            self.init_conv(nfeat, nclass, **kwargs))

        self.norms = None
        if norm is not None:
            norm_layer = normalization_resolver(
                norm,
                nhid,
                **(norm_kwargs or {}),
            )
            self.norms = ModuleList()
            for _ in range(nlayers - 1):
                self.norms.append(copy.deepcopy(norm_layer))


    def init_conv(self, nfeat: Union[int, Tuple[int, int]],
                  nclass: int, **kwargs) -> MessagePassing:
        return Linear(nfeat, nclass, bias=True, weight_initializer='glorot')

    def initialize(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()

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
        
        for i in range(self.nlayers):
            x = self.convs[i](x)
            if i == self.nlayers - 1:
                break
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = x
        message=MessagePassing(aggr="add")
        for k in range(self.K):
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = message.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            x = x * (1 - self.alpha)
            x += self.alpha * h

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

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.nfeat}, '
                f'{self.nclass}, nlayers={self.nlayers})')