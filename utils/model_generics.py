import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .scatter_ops import scatter_softmax, scatter
from dataclasses import dataclass

from typing import Union, Tuple, List, Optional, Callable

class EquivariantData:
    """
    Implements a dataclass to store the scalar and vector embeddings for model data.
    """
    def __init__(self, scalars: torch.Tensor, vectors: torch.Tensor):
        self.scalars = scalars
        self.vectors = vectors

        assert self.scalars.shape[0] == self.vectors.shape[0], "EquivariantData scalars and vectors must have the same number of elements!"
        assert self.vectors.shape[-1] == 3, "EquivariantData vectors must end with a dimension of size 3!"
        assert self.scalars.device == self.vectors.device, "EquivariantData scalars and vectors must be on the same device!"
    
    def __repr__(self) -> str:
        return f"EquivariantData(scalars={self.scalars}, vectors={self.vectors})"

    def clone(self) -> 'EquivariantData':
        return EquivariantData(self.scalars.clone(), self.vectors.clone())

    def clone_detach(self) -> 'EquivariantData':
        return EquivariantData(self.scalars.clone().detach(), self.vectors.clone().detach())

    @property
    def device(self) -> torch.device:
        return self.scalars.device
    
    @property
    def shape(self) -> tuple:
        return (self.scalars.shape, self.vectors.shape)

    def to_tuple(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tensor of scalars and a tensor of vectors.
        """
        if self.vectors.dim() == 2:
            return (self.scalars, self.vectors.unsqueeze(1))
        return (self.scalars, self.vectors)
    
    def get_zeroslike(self) -> 'EquivariantData':
        return EquivariantData(torch.zeros_like(self.scalars), torch.zeros_like(self.vectors))

    def get_indices(self, indices: torch.Tensor) -> 'EquivariantData':
        """
        Returns indices from the scalars and vectors.
        """
        return EquivariantData(self.scalars[indices], self.vectors[indices])
    
    def set_indices(self, indices: torch.Tensor, new_data: 'EquivariantData') -> None:
        """
        Sets indices from the scalars and vectors to new_data.
        """
        self.scalars[indices] = new_data.scalars
        self.vectors[indices] = new_data.vectors
    
    def num_nodes(self) -> int:
        return self.scalars.shape[0]


class HeteroGATv2(nn.Module):
    """
    Implements a GATv2 that operates over multiple node types.

    Given multiple types of nodes which need to pass messages to a single type of node 
        (e.g. protein nodes to protein nodes and ligand nodes to protein nodes),
        this module implements a GATv2 layer for each node type and then merges the updates with 
        a shared softmax attention-weighted update layer with ProteinMPNN-like dense node updates if necessary.

    Instantiates minimal HomoGATv2 layers for each node type to use `aggregate_node_update_and_presoftmax_atten` to get node updates and attention coefficients.
    """
    def __init__(self, num_subgraphs_to_merge: int, node_embedding_dim: Union[List[Tuple[int, int]], int], edge_embedding_dim: Union[List[int], int], atten_head_aggr_layers: int, num_attention_heads: int, dropout: float, use_mlp_node_update: bool, use_residual_node_update: bool, compute_edge_updates: bool, num_vectors: int, output_node_embedding_dim: Optional[int] = None, output_edge_embedding_dim: Optional[int] = None, **kwargs):
        super(HeteroGATv2, self).__init__()
        self.use_residual_node_update = use_residual_node_update
        self.compute_edge_updates = compute_edge_updates
        self.num_vectors = num_vectors

        # Sanity check input combinations.
        output_node_embedding_dim = overloaded_input_type_check(node_embedding_dim, output_node_embedding_dim)
        output_edge_embedding_dim = overloaded_input_type_check(edge_embedding_dim, output_edge_embedding_dim)

        # Construct HomoGATv2 layers for each node type, to use `aggregate_node_update_and_presoftmax_atten`
        self.subgats = nn.ModuleList([
            HomoGATv2(
                node_embedding_dim if isinstance(node_embedding_dim, int) else node_embedding_dim[idx],
                edge_embedding_dim if isinstance(edge_embedding_dim, int) else edge_embedding_dim[idx], 
                num_attention_heads, 
                dropout, 
                num_vectors = self.num_vectors,
                use_mlp_node_update = use_mlp_node_update, 
                update_edges = False, 
                atten_head_aggr_layers = 0,
                output_node_embedding_dim=output_node_embedding_dim if not isinstance(node_embedding_dim, int) else None,
                output_edge_embedding_dim=output_edge_embedding_dim if not isinstance(edge_embedding_dim, int) else None,
                **kwargs
            )
            for idx in range(num_subgraphs_to_merge)
        ])

        # TODO: choose a node embedding dim with additional kwarg.
        self.dense_residual_node_update = None
        if self.use_residual_node_update:
            self.dense_residual_node_update = DenseResidualNodeUpdate(output_node_embedding_dim, dropout)

        self.final_atten_aggr = AttentionAggregationModule(atten_head_aggr_layers, output_node_embedding_dim, num_attention_heads)
        self.vectors_update_layer = GVP((output_node_embedding_dim, self.num_vectors), (output_node_embedding_dim, self.num_vectors), vector_gate=True)
        self.equivariant_layer_norm = EquivariantLayerNorm((output_node_embedding_dim, self.num_vectors), vector_only=True)

        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.leakyrelu = nn.LeakyReLU(0.2)
    
    def compute_edge_update(self, updated_nodes: EquivariantData, prev_nodes: EquivariantData, input_node_list: List[Tuple[EquivariantData, EquivariantData]], edge_attr_list: List[torch.Tensor], edge_index_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Computes edge updates for each subgraph using updated node embeddings.
        Args:
            updated_nodes: The node embedding updates produced by GAT convolution - (num_nodes, node_embedding_dim) 
            prev_nodes: The node embeddings input to forward pass - (num_nodes, node_embedding_dim) 
            input_node_list: List of tuples of (source, sink) node data tensors for each subgraph. [((num_nodes_1, node_embedding_dim), (num_nodes_2, node_embedding_dim)), ...]
            edge_attr_list: List of previous edge attribute tensors for each subgraph. [(num_edges, edge_embedding_dim), ...]
            edge_index_list: List of previous edge index tensors for each subgraph. [(2, num_edges), ...]
                Indices map from source -> sink nodes in corresponding input_node_list elements.

        Returns:
            List of updated edge attr tensors for each subgraph. [(num_edges, edge_embedding_dim), ...]
        """
        output = []
        # Compute edge updates with updated node embeddings.
        for subgat_idx, subgat in enumerate(self.subgats):

            # Unpack node and edge attributes for this subgraph.
            source_nodes, sink_nodes = input_node_list[subgat_idx]
            source_node_scalars, sink_node_scalars = source_nodes.scalars, sink_nodes.scalars
            curr_eattr = edge_attr_list[subgat_idx]
            curr_eidx = edge_index_list[subgat_idx]

            # use updated nodes in place of previous nodes if necessary.
            if source_node_scalars is prev_nodes.scalars:
                source_node_scalars = updated_nodes.scalars
            if sink_node_scalars is prev_nodes.scalars:
                sink_node_scalars = updated_nodes.scalars

            # Compute edge updates with updated node embeddings.
            curr_eattr = subgat.compute_edge_update(source_node_scalars[curr_eidx[0]], sink_node_scalars[curr_eidx[1]], curr_eattr) # type: ignore
            output.append(curr_eattr)
        return output
    
    def aggregate_node_update(self, all_presoftmax_atten: torch.Tensor, all_node_updates: torch.Tensor, all_sink_edges: torch.Tensor, prev_sink_nodes: EquivariantData) -> EquivariantData:
        # Compute attention coefficients and attention-weighted node updates.
        atten = scatter_softmax(all_presoftmax_atten, all_sink_edges, dim=1)
        atten = self.dropout(atten)
        node_update = all_node_updates * atten
        node_update = scatter(node_update, all_sink_edges, dim=1, reduce='sum', dim_size=prev_sink_nodes.num_nodes())

        # Aggregate attention head updates with linear layers or mean depeding on atten_head_aggr_layers.
        node_update = self.final_atten_aggr(node_update)

        # Computes dense residual node updates if necessary
        if self.use_residual_node_update and self.dense_residual_node_update is not None:
            node_update = self.dense_residual_node_update(prev_sink_nodes.scalars, node_update)

        # Update node vectors with GVP layer
        ### Note: setting prev_sink_nodes.scalars to node_update directly messes up gradients.
        output_node_equidata = self.equivariant_layer_norm(self.vectors_update_layer(EquivariantData(node_update, prev_sink_nodes.vectors.clone())))

        return output_node_equidata

    def forward(self, prev_sink_nodes: EquivariantData, node_list: List[Tuple[EquivariantData, EquivariantData]], edge_attr_list: List[torch.Tensor], edge_index_list: List[torch.Tensor]) -> Tuple[EquivariantData, List[torch.Tensor]]:
        """
        Assumes sink nodes are the same for all subgraphs.
        """
        num_prev_sink_nodes = prev_sink_nodes.num_nodes()

        # Compute node updates and attention coefficients.
        all_presoftmax_atten, all_node_updates, all_sink_edges = [], [], []
        for subgat_idx, subgat in enumerate(self.subgats):
            # Unpack node and edge attributes for this subgraph.
            source_nodes, sink_nodes = node_list[subgat_idx]
            curr_eattr = edge_attr_list[subgat_idx]
            curr_eidx = edge_index_list[subgat_idx]

            # Compute node updates and attention coefficients.
            node_update, atten = subgat.aggregate_node_update_and_presoftmax_atten(source_nodes.scalars[curr_eidx[0]], sink_nodes.scalars[curr_eidx[1]], curr_eattr) # type: ignore
            all_sink_edges.append(curr_eidx[1])
            all_node_updates.append(node_update)
            all_presoftmax_atten.append(atten)
        
        # Concatenate all node updates and attention coefficients.
        all_sink_edges = torch.cat(all_sink_edges, dim=0)
        all_node_updates = torch.cat(all_node_updates, dim=1)
        all_presoftmax_atten = torch.cat(all_presoftmax_atten, dim=1)
        
        # Compute node update from attention coefficients and attention-weighted node updates.
        nodes = self.aggregate_node_update(all_presoftmax_atten, all_node_updates, all_sink_edges, prev_sink_nodes)

        # Compute edge updates with updated node embeddings.
        new_eattr = self.compute_edge_update(nodes, prev_sink_nodes, node_list, edge_attr_list, edge_index_list)

        assert nodes.num_nodes() == num_prev_sink_nodes, f"Number of nodes changed during HeteroGATv2 forward pass! {nodes.shape[0]} -> {num_prev_sink_nodes}"
        return nodes, new_eattr
    
    def preexpanded_edges_forward(self, prev_sink_nodes: EquivariantData, expanded_nodes: List[Tuple[torch.Tensor, torch.Tensor]], edge_index_list: List[torch.Tensor], edge_attr_list: List[torch.Tensor]) -> EquivariantData:
        """
        """
        num_prev_sink_nodes = prev_sink_nodes.num_nodes()

        # Compute node updates and attention coefficients.
        all_presoftmax_atten, all_node_updates, all_sink_edges = [], [], []
        for subgat_idx, subgat in enumerate(self.subgats):
            # Unpack node and edge attributes for this subgraph.
            curr_exp_source, curr_exp_sink = expanded_nodes[subgat_idx]
            curr_eattr = edge_attr_list[subgat_idx]
            curr_eidx = edge_index_list[subgat_idx]

            # Compute node updates and attention coefficients.
            node_update, atten = subgat.aggregate_node_update_and_presoftmax_atten(curr_exp_source, curr_exp_sink, curr_eattr) # type: ignore
            all_sink_edges.append(curr_eidx[1])
            all_node_updates.append(node_update)
            all_presoftmax_atten.append(atten)

        # Concatenate all node updates and attention coefficients.
        all_sink_edges = torch.cat(all_sink_edges, dim=0)
        all_node_updates = torch.cat(all_node_updates, dim=1)
        all_presoftmax_atten = torch.cat(all_presoftmax_atten, dim=1)

        # Compute node update from attention coefficients and attention-weighted node updates.  
        nodes = self.aggregate_node_update(all_presoftmax_atten, all_node_updates, all_sink_edges, prev_sink_nodes)

        assert not self.compute_edge_updates, "Edge updates not implemented for preexpanded_edges_forward!"
        assert nodes.num_nodes() == num_prev_sink_nodes, f"Number of nodes changed during HeteroGATv2 forward pass! {nodes.shape[0]} -> {num_prev_sink_nodes}"
        return nodes


class HomoGATv2(nn.Module):
    """
    Implements a GATv2 that operates over with a single node type graphs.
        TODO: Implement dense residual node updates for homogeneous graphs as well.
    """
    def __init__(self, node_embedding_dim: Union[Tuple[int, int], int], edge_embedding_dim: int, num_attention_heads: int, dropout: float, update_edges: bool, use_mlp_node_update: bool, atten_head_aggr_layers: int, num_vectors: int, atten_dimension_upscale_factor: int, output_node_embedding_dim: Optional[int] = None, **kwargs):
        super(HomoGATv2, self).__init__()
        self.num_vectors = num_vectors
        self.edge_embedding_dim = edge_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.atten_head_aggr_layers = atten_head_aggr_layers
        self.atten_dimension_upscale_factor = atten_dimension_upscale_factor
        self.use_mlp_node_update = use_mlp_node_update
        self.update_edges = update_edges
        assert num_attention_heads > 0, "Number of attention heads must be greater than 0!"

        # Sanity check input combinations.
        output_node_embedding_dim = overloaded_input_type_check(node_embedding_dim, output_node_embedding_dim)
        self.node_embedding_dim = output_node_embedding_dim 

        # Identify the input dimension of the featurization, allow nodes to come in in with different shapes.
        if isinstance(node_embedding_dim, int):
            featurization_input_dim = (2 * node_embedding_dim) + edge_embedding_dim
        else:
            featurization_input_dim = sum(node_embedding_dim) + edge_embedding_dim
            
        self.atten_dim = atten_dimension_upscale_factor * output_node_embedding_dim if self.use_mlp_node_update else output_node_embedding_dim

        self.gatW = nn.Linear(featurization_input_dim, num_attention_heads * self.atten_dim, bias=False)
        self.gatA = Parameter(torch.Tensor(num_attention_heads, self.atten_dim, 1))
        nn.init.xavier_uniform_(self.gatA)

        # Implements mean aggregation when atten_head_aggr_layers <= 0, otherwise uses linear layers with GELU activations.
        self.final_atten_aggr = AttentionAggregationModule(atten_head_aggr_layers, output_node_embedding_dim, num_attention_heads)

        # Override node update with dense proteinMPNN-like update.
        self.dense_node_update_layers = None
        if use_mlp_node_update: 
            self.dense_node_update_layers = DenseMLP(featurization_input_dim, output_node_embedding_dim, output_node_embedding_dim)
        
        # Optionally, compute an edge update with dense proteinMPNN-like update.
        self.linear_edge_updates = None
        if self.update_edges:
            self.linear_edge_updates = DenseMLP(featurization_input_dim, edge_embedding_dim, edge_embedding_dim)
            self.edge_norm = nn.LayerNorm(edge_embedding_dim)

        self.vectors_update_layer = GVP((self.node_embedding_dim, self.num_vectors), (self.node_embedding_dim, self.num_vectors), vector_gate=True)
        self.equivariant_layer_norm = EquivariantLayerNorm((self.node_embedding_dim, self.num_vectors))

        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.leakyrelu = nn.LeakyReLU(0.2)
    
    def aggregate_node_update_and_presoftmax_atten(self, source_nodes: torch.Tensor, sink_nodes: torch.Tensor, e_attr: torch.Tensor):
        """
        Computes node updates and attention coefficients given source and sink nodes and edge attributes.
        Inputs:
            source_nodes: (num_edges, node_embedding_dim),
            sink_nodes: (num_edges, node_embedding_dim),
            e_attr: (num_edges, edge_embedding_dim),
        Returns:
            node_update: (num_attention_heads, num_edges, node_embedding_dim),
            atten: (num_attention_heads, num_edges, 1),
        """
        # Concatenate all features for this edge.
        edge_features_exp = torch.cat([source_nodes, e_attr, sink_nodes], dim=1)

        # Compute node updates. 
        node_update = self.gatW(edge_features_exp)
        node_update = node_update.reshape(-1, self.num_attention_heads, self.atten_dim).permute(1, 0, 2)

        # Compute attention coefficients.
        atten = torch.bmm(self.leakyrelu(node_update), self.gatA)

        # Override node update with dense proteinMPNN-like update.
        if self.use_mlp_node_update and self.dense_node_update_layers is not None:
            node_update = self.dense_node_update_layers(edge_features_exp).unsqueeze(0).expand(self.num_attention_heads, -1, -1)
        
        return node_update, atten

    def compute_edge_update(self, source_nodes: torch.Tensor, sink_nodes:torch.Tensor, e_attr: torch.Tensor) -> torch.Tensor:
        """
        If self.update_edges is True, computes an edge update with dense 3-layer-MLP update.
            Otherwise, returns the original edge attributes.
        """
        if self.update_edges and self.linear_edge_updates is not None:
            edge_update = torch.cat([source_nodes, e_attr, sink_nodes], dim=1)
            edge_update = self.linear_edge_updates(edge_update)
            e_attr = self.edge_norm(e_attr + self.dropout(edge_update))

        return e_attr

    def forward(self, nodes: EquivariantData, e_attr: torch.Tensor, e_idx: torch.Tensor) -> Tuple[EquivariantData, torch.Tensor]:
        """
        Computes node updates for a single node-type graph.

        Inputs:
            nodes: [num_nodes, node_embedding_dim],
            e_attr: [num_edges, edge_embedding_dim],
            e_idx: [source -> sink] [2, num_edges]

        Returns:
            nodes: [num_nodes, node_embedding_dim],
            e_attr: [num_edges, edge_embedding_dim],
        """
        input_num_nodes = nodes.num_nodes()
        node_update, atten = self.aggregate_node_update_and_presoftmax_atten(nodes.scalars[e_idx[0]], nodes.scalars[e_idx[1]], e_attr)

        # Compute attention coefficients and attention-weighted node updates.
        atten = scatter_softmax(atten, e_idx[1], dim=1)
        atten = self.dropout(atten)
        node_update = node_update * atten
        node_update = scatter(node_update, e_idx[1], dim=1, reduce='sum', dim_size=input_num_nodes)

        # Aggregate attention head updates with linear layers or mean depeding on atten_head_aggr_layers.
        nodes.scalars = self.final_atten_aggr(node_update)
        
        # Update node vectors with GVP layer.
        all_update = self.vectors_update_layer(nodes)
        nodes = self.equivariant_layer_norm(all_update)

        # Compute edge updates with updated node embeddings.
        e_attr = self.compute_edge_update(nodes.scalars[e_idx[0]], nodes.scalars[e_idx[1]], e_attr)

        assert nodes.num_nodes() == input_num_nodes, f"Number of nodes changed during HomoGATv2 forward pass! {nodes.shape[0]} -> {input_num_nodes}"
        return nodes, e_attr


class AttentionAggregationModule(nn.Module):
    """
    Module responsible for aggregating updates in batched tensors from multi-head attention.
    """
    def __init__(self, atten_head_aggr_layers: int, node_embedding_dim: int, num_attention_heads: int):
        super(AttentionAggregationModule, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.node_embedding_dim = node_embedding_dim
        self.atten_head_aggr_layers = atten_head_aggr_layers

        # If atten_head_aggr_layers <= 0, mean aggregation is used.
        self.final_atten_aggr = None
        if atten_head_aggr_layers > 0:
            # Automatically adds Gelu + Linear layers to final updates if attention_head_aggr_layers >1.
            self.final_atten_aggr = nn.Sequential(
                *([nn.Linear(node_embedding_dim * num_attention_heads, node_embedding_dim)] + 
                    ((atten_head_aggr_layers - 1) * [nn.GELU(), nn.Linear(node_embedding_dim, node_embedding_dim)])) 
            )

    def forward(self, node_update: torch.Tensor) -> torch.Tensor:
        """
        When initialized with atten_head_aggr_layers == 0, returns mean of attention head updates.
        Otherwise uses atten_head_aggr_layers linear layers with GELU activations to aggregate attention head updates.
        """
        if self.atten_head_aggr_layers > 0 and self.final_atten_aggr is not None:
            node_update = self.final_atten_aggr(node_update.permute(1, 0, 2).reshape(-1, self.num_attention_heads * self.node_embedding_dim))
        else:
            node_update = node_update.mean(dim=0)
        return node_update


class DenseMLP(nn.Module):
    """
    Just a 3-layer MLP with GELU activations.
    """
    def __init__(self, input_dim: int, latent_dim: int, output_dim: int, mlp_dropout: float = 0.0):
        super(DenseMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Dropout(mlp_dropout),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(mlp_dropout),
            nn.GELU(),
            nn.Linear(latent_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class DenseResidualNodeUpdate(nn.Module):
    """
    Dense residual node update layer from ProteinMPNN.
    Takes in a node update and the original node features and returns a node update.
    """
    def __init__(self, input_dim: int, dropout: float):
        super(DenseResidualNodeUpdate, self).__init__()
        self.node_norm1 = nn.LayerNorm(input_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.GELU(),
            nn.Linear(2 * input_dim, input_dim),
        )
        self.node_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, prev_nodes, node_update):
        assert prev_nodes.shape == node_update.shape, f"Node update and previous node embeddings must have the same shape! {prev_nodes.shape} != {node_update.shape}"
        prev_nodes = self.node_norm1(prev_nodes + self.dropout(node_update))
        node_update = self.layers(prev_nodes)
        prev_nodes = self.node_norm2(prev_nodes + self.dropout(node_update))
        return prev_nodes


def overloaded_input_type_check(embedding_dim: Union[Tuple[int, int], List[Tuple[int, int]], List[int], int], output_embedding_dim: Optional[int]):
    """
    Checks that embedding_dim and output_embedding_dim are compatible.
    Raises ValueError if not.
    """
    if not isinstance(embedding_dim, int):
        if output_embedding_dim is None:
            raise ValueError("output_embedding_dim is required when passing a list of embedding_dims!")
    else:
        if output_embedding_dim is not None:
            raise ValueError("Passing output_embedding_dim only supported for list of embedding_dims!")
        output_embedding_dim = embedding_dim
    return output_embedding_dim


class GVP(nn.Module):
    '''
    Geometric Vector Perceptron implementation adapted from https://github.com/drorlab/gvp-pytorch 
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(
        self, in_dims: Tuple[int, int], 
        out_dims: Tuple[int, int], 
        h_dim: Optional[int] = None, 
        activations: Tuple[Optional[Callable], Optional[Callable]] = (F.gelu, torch.sigmoid), 
        vector_gate: bool = False
    ):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        self.scalar_act, self.vector_act = activations

        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)

            if self.si and self.so:
                self.ws = nn.Linear(self.h_dim + self.si, self.so)
            else:
                self.scalar_act = None

            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: 
                    self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

    @property
    def device(self):
        return next(self.parameters()).device
        
    def forward(self, x: EquivariantData) -> EquivariantData:
        '''
        :param x: tuple (s, V) of `torch.Tensor`, 
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x.to_tuple()
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)    
            vn = _norm_no_nan(vh, axis=-2)

            if self.si and self.so:
                s = self.ws(torch.cat([s, vn], -1))

            if self.vo: 
                v = self.wv(vh) 
                v = torch.transpose(v, -1, -2)
                if self.vector_gate: 
                    assert self.si and self.so, 'Vector gating requires scalar channels.'
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3, device=self.device)

        if self.scalar_act:
            s = self.scalar_act(s)

        if self.vo:
            return EquivariantData(s, v) # type: ignore
        else:
            return EquivariantData(s, torch.empty(s.shape[0], 0, 3, device=self.device))


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


class EquivariantLayerNorm(nn.Module):
    '''
    Adapted from https://github.com/drorlab/gvp-pytorch 

    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims, vector_only=False):
        super(EquivariantLayerNorm, self).__init__()
        self.s, self.v = dims
    
        self.vector_only = vector_only
        if not self.vector_only:
            self.scalar_norm = nn.LayerNorm(self.s)
        
    def forward(self, x: EquivariantData) -> Union[EquivariantData, torch.Tensor]:
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if not self.vector_only:
            if not self.v and x.vectors.numel() == 0:
                return self.scalar_norm(x.scalars)
            s, v = x.to_tuple()
            vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
            vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
            s = self.scalar_norm(s)
        else:
            s, v = x.to_tuple()
            vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
            vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return EquivariantData(s, v / vn)


class DenseGVP(nn.Module):
    """
    Implements a GVP layer with a single hidden layer and vector gates.
    """
    def __init__(self, input_dim: Tuple[int, int], latent_dim: Tuple[int, int], output_dim: Tuple[int, int], dropout: float, intermediate_norm: bool = False, vector_only_norm: bool = False):
        super(DenseGVP, self).__init__()
        self.l1 = GVP(input_dim, latent_dim, activations=(F.gelu, torch.sigmoid), vector_gate=True)
        self.l2 = GVP(latent_dim, latent_dim, activations=(F.gelu, torch.sigmoid), vector_gate=True)
        self.l3 = GVP(latent_dim, output_dim, activations=(F.gelu, torch.sigmoid), vector_gate=True)
        self.dropout = EquivariantDropout(dropout)

        assert not (intermediate_norm and vector_only_norm), "Cannot have both intermediate_norm and vector_only_norm!"
        if intermediate_norm:
            self.norm1 = EquivariantLayerNorm(latent_dim)
            self.norm2 = EquivariantLayerNorm(latent_dim)
        if vector_only_norm:
            self.norm1 = EquivariantLayerNorm(latent_dim, vector_only=True)
            self.norm2 = EquivariantLayerNorm(latent_dim, vector_only=True)
    
    def forward(self, x: EquivariantData):
        if hasattr(self, 'norm1') and hasattr(self, 'norm2'):
            return self.l3(self.dropout(self.norm2(self.l2(self.dropout(self.norm1(self.l1(x)))))))
        else:
            return self.l3(self.dropout(self.l2(self.dropout(self.l1(x)))))


class _VDropout(nn.Module):
    '''
    Borrowed from https://github.com/drorlab/gvp-pytorch 
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class EquivariantDropout(nn.Module):
    '''
    Borrowed from https://github.com/drorlab/gvp-pytorch 
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, drop_rate):
        super(EquivariantDropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x: EquivariantData):
        '''
        :param x: EquivariantData
        '''
        s, v = x.to_tuple()
        return EquivariantData(self.sdropout(s), self.vdropout(v))
