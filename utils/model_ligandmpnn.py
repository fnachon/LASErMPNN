
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from typing import Optional, Tuple, Union
from dataclasses import dataclass


from .build_rotamers import RotamerBuilder
from .ligand_featurization_ligandmpnn import LigandFeaturizer
from .pdb_dataset import BatchData
from .constants import aa_short_to_idx


@dataclass
class Sampled_Output:
    """
    Stores the output of a sample from the model.
    """
    sequence_logits: torch.Tensor
    sampled_sequence_indices: torch.Tensor

    def to(self, device: torch.device) -> 'Sampled_Output':
        return Sampled_Output(self.sequence_logits.to(device), self.sampled_sequence_indices.to(device))


def create_sampling_output(num_residues: int, device: torch.device) -> Sampled_Output:
    """
    Initializes the output tensors to zeros.
    """
    sequence_logits = torch.zeros((num_residues, 21), device=device)
    sampled_sequence_indices = torch.zeros((num_residues,), dtype=torch.long, device=device)

    return Sampled_Output(sequence_logits, sampled_sequence_indices)


def minp_warp_logits(logits: torch.Tensor, min_p: float, min_indices_to_keep: int = 1) -> torch.Tensor:
    """
    Given an input tensor [B, N] of logits and a threshold minimum probability fraction min_p, 
        under which all probabilities are set to 0. The min_p probability threshold is scaled 
        by the top probability of each sequence in the batch so the threshold is
        relative to the top token's probability.  
    
    Returns a tensor of the same shape with pre-softmax logits set to -inf for softmax.

    Adapted from: 
        https://github.com/menhguin/minp_paper/blob/main/implementation
    """
    # Setting min_p to 0.0 will return the original logits.
    if min_p == 0.0:
        return logits

    # Convert logits to probabilities
    probs = logits.softmax(dim=-1)
    # Get the probability of the top token for each sequence in the batch
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    # Calculate the actual min_p threshold by scaling min_p with the top token's probability
    scaled_min_p = min_p * top_probs
    # Mask out the probabilities below the threshold
    indices_to_remove = probs < scaled_min_p
    sorted_indices = probs.argsort(dim=-1, descending=True)
    sorted_indices_to_remove = torch.gather(indices_to_remove, dim=-1, index=sorted_indices)
    # Keep at least min_indices_to_keep indices
    sorted_indices_to_remove[:, :min_indices_to_keep] = False
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    # Mask out the probabilities below the threshold
    scores_processed = logits.masked_fill(indices_to_remove, float('-Inf'))
    return scores_processed


class LigandMPNN(nn.Module):
    def __init__(
        self, 
        node_embedding_dim: int, protein_edge_embedding_dim: int,
        prot_prot_edge_rbf_params: dict, lig_prot_edge_rbf_params: dict, 
        num_encoder_layers: int, num_decoder_layers: int,
        use_angle_features: bool = True, **kwargs
    ):
        super(LigandMPNN, self).__init__()

        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = protein_edge_embedding_dim

        self.rotamer_builder = RotamerBuilder(5.0)
        self.ligand_featurizer = LigandFeaturizer(**kwargs)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.ligand_encoder_layers = nn.ModuleList([
            LigandMPNNDecoderLayer(node_embedding_dim, node_embedding_dim * 2, scale=-1, dropout=kwargs['dropout']) for _ in range(2)
        ])

        self.protein_encoder_layers = nn.ModuleList([
            LigandMPNNProteinEncoderLayer(node_embedding_dim, node_embedding_dim * 2, dropout=kwargs['dropout']) for _ in range(num_encoder_layers)
        ])
        self.ligand_to_protein_encoder_layers = nn.ModuleList([
            LigandMPNNDecoderLayer(node_embedding_dim, node_embedding_dim * 2, dropout=kwargs['dropout']) for _ in range(2)
        ])

        self.protein_decoder_layers = nn.ModuleList([
            LigandMPNNDecoderLayer(node_embedding_dim, (node_embedding_dim * 3), dropout=kwargs['dropout']) for _ in range(num_decoder_layers)
        ])

        self.W_lig_in = nn.Linear(self.ligand_featurizer.output_dim, node_embedding_dim)
        self.W_c = nn.Linear(node_embedding_dim, node_embedding_dim)
        self.context_norm = nn.LayerNorm(node_embedding_dim)
        self.dropout = nn.Dropout(kwargs['dropout'])

        self.lig_lig_rbf_encoding = RBF_Encoding(**{'num_bins': 75, 'bin_min': 0.0, 'bin_max': 20.0})
        self.prot_prot_rbf_encoding = RBF_Encoding(**prot_prot_edge_rbf_params)
        self.lig_prot_rbf_encoding = RBF_Encoding(**lig_prot_edge_rbf_params)

        self.lig_lig_edge_input_layer = nn.Linear(self.lig_lig_rbf_encoding.num_bins, protein_edge_embedding_dim)
        self.prot_prot_edge_input_layer = nn.Linear(self.prot_prot_rbf_encoding.num_bins * 25, protein_edge_embedding_dim)
        self.use_angle_features = use_angle_features
        if self.use_angle_features:
            self.lig_prot_edge_input_layer = nn.Linear((self.lig_prot_rbf_encoding.num_bins * 5) + 4, protein_edge_embedding_dim)
        else:
            self.lig_prot_edge_input_layer = nn.Linear(self.lig_prot_rbf_encoding.num_bins * 5, protein_edge_embedding_dim)

        # 20 AAs + X + NotDecoded
        self.sequence_label_embedding = nn.Embedding(22, node_embedding_dim)
        self.sequence_output_layer = nn.Linear(node_embedding_dim, 21)

        self.gelu = nn.GELU()

    @property
    def device(self) -> torch.device:
        """
        Returns the device that the model is currently on when addressed as model.device
        """
        return next(self.parameters()).device
    
    def apply_encoding_layers(self, batch: BatchData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        """
        assert batch.ligand_data is not None, "Ligand data must be defined in batch data, even if filled with empty tensors."

        # Encode edges with RBFs.
        lig_lig_eattr = self.lig_lig_edge_input_layer(self.lig_lig_rbf_encoding(batch.ligand_data.lig_lig_edge_distance).flatten(start_dim=1))
        pr_pr_eattr = self.prot_prot_edge_input_layer(self.prot_prot_rbf_encoding(batch.pr_pr_edge_distance).flatten(start_dim=1))

        # Add angle features to the ligand-protein edges if desired.
        pre_lig_pr_dim_red = self.lig_prot_rbf_encoding(batch.lig_pr_edge_distance).flatten(start_dim=1)
        if self.use_angle_features:
            pre_lig_pr_dim_red = torch.cat([
                pre_lig_pr_dim_red, _make_angle_features(batch.backbone_coords, batch.ligand_data.lig_coords, batch.lig_pr_edge_index)
            ], dim=1)
        lig_pr_eattr = self.lig_prot_edge_input_layer(pre_lig_pr_dim_red)

        # Initialize protein nodes to zeros, representations will be built-up by encoding process.
        prot_nodes = torch.zeros((batch.num_residues, self.node_embedding_dim), device=self.device)

        # Apply ligand-ligand encoder layers.
        lig_nodes = self.W_lig_in(batch.ligand_data.lig_nodes)
        for ligand_enc_layer in self.ligand_encoder_layers:
            expanded_node_and_edge = torch.cat([lig_nodes[batch.ligand_data.lig_lig_edge_index[0]], lig_lig_eattr], dim=1)
            lig_nodes = ligand_enc_layer(expanded_node_and_edge, lig_nodes, batch.ligand_data.lig_lig_edge_index)
        
        # Apply protein-protein encoder layers.
        for enc_layer in self.protein_encoder_layers:
            prot_nodes, pr_pr_eattr = enc_layer(prot_nodes, pr_pr_eattr, batch.pr_pr_edge_index)
        
        # Apply ligand-protein encoder layers.
        context_nodes = self.W_c(prot_nodes)
        for enc_layer in self.ligand_to_protein_encoder_layers:
            node_and_edge = torch.cat([lig_nodes[batch.lig_pr_edge_index[0]], lig_pr_eattr], dim=1)
            context_nodes = enc_layer(node_and_edge, context_nodes, batch.lig_pr_edge_index)
        
        prot_nodes = prot_nodes + self.context_norm(self.dropout(context_nodes))

        return prot_nodes, pr_pr_eattr
    

    def forward(
            self, batch: BatchData, return_nodes: bool = False, return_unconditional_probabilities: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Main teacher-forced forward pass for training the model in a supervised manner.
        Args:
            return_unconditional_probabilities: to return the probabilities of all residues being the first to be decoded.
            return_full_conditional_probabilities: to return the probabilities of all residues being decoded before the current residue.
        """
        # Input sanity checks.
        assert batch.pr_pr_edge_index is not None, "Protein-protein edge index must be specified in batch data."
        assert batch.pr_pr_edge_distance is not None, "Protein-protein edge distance must be specified in batch data."
        assert batch.decoding_order is not None, "Decoding order must be specified in batch data."

        # Apply encoding layers to build up protein and ligand representations.
        prot_nodes, pr_pr_eattr = self.apply_encoding_layers(batch)

        # Create protein-protein edge mask, if True then source node was decoded before target node. Self-edges are False.
        ### Argsort on decoding order maps from node index to decoding order, so we can use this to compare decoding order of source and target nodes.
        decoding_order_sort_indices = batch.decoding_order.argsort()

        if return_unconditional_probabilities:
            # Provide nothing for all residues (mimics treating all residues as the first to be decoded).
            pr_pr_edge_mask = torch.zeros_like(batch.pr_pr_edge_index[0], dtype=torch.bool)
        else:
            # Otherwise, provide only residues that would be decoded before the current residue if we were autoregressively sampling.
            pr_pr_edge_mask = (decoding_order_sort_indices[batch.pr_pr_edge_index[0]] < decoding_order_sort_indices[batch.pr_pr_edge_index[1]])

        unmasked_edge_indices = batch.pr_pr_edge_index[:, pr_pr_edge_mask]
        unmasked_edge_attrs = pr_pr_eattr[pr_pr_edge_mask]
        masked_edge_indices = batch.pr_pr_edge_index[:, ~pr_pr_edge_mask]
        masked_edge_attrs = pr_pr_eattr[~pr_pr_edge_mask]

        sequence_embedding_nodes = self.sequence_label_embedding(batch.sequence_indices)
        sequence_embedding_for_teacher_forcing = sequence_embedding_nodes[unmasked_edge_indices[0]]

        # Use decoding order mask to selectively provide label data for nodes that have been previously decoded.
        source_node_edge_features_exp_unmasked = torch.cat([sequence_embedding_for_teacher_forcing, unmasked_edge_attrs], dim=1)

        # Fill sequence representations with 22nd mask token for nodes not yet decoded.
        masked_sequence_embeddings = self.sequence_label_embedding(torch.full_like(batch.sequence_indices, fill_value=21))

        # Append edge features and encoder node features to nodes that have not yet been decoded, dont add protein node embeddings to unmasked nodes yet as these will be updated in decoding process.
        source_node_edge_features_exp_masked = torch.cat([masked_sequence_embeddings[masked_edge_indices[0]], masked_edge_attrs, prot_nodes[masked_edge_indices[0]]], dim=1)

        # Iteratively update protein nodes with teacher-forced decoding.
        for decoder_layer in self.protein_decoder_layers:
            # Provides node features for unmasked source nodes at current step of decoding.
            curr_nodes_exp = prot_nodes[unmasked_edge_indices[0]]
            curr_edge_features_unmasked = torch.cat([source_node_edge_features_exp_unmasked, curr_nodes_exp], dim=1)

            # Add self-edges to nodes that have not yet been decoded.
            all_edge_features = torch.cat([source_node_edge_features_exp_masked, curr_edge_features_unmasked], dim=0)
            all_edge_indices = torch.cat([masked_edge_indices, unmasked_edge_indices], dim=1)

            # Update node embeddings with teacher-forced edges.
            prot_nodes = decoder_layer(all_edge_features, prot_nodes, all_edge_indices)
        
        # Compute sequence logits from final protein node embeddings.
        sequence_logits = self.sequence_output_layer(prot_nodes)

        return sequence_logits
    
    @torch.no_grad()
    def sample(
            self, batch: BatchData, sequence_sample_temperature: Optional[Union[float, torch.Tensor]] = None, 
            disabled_residues: Optional[list] = ['X'], 
            disable_pbar: bool = False, return_encoder_embeddings: bool = False, seq_min_p: float = 0.0,
            ignore_chain_mask_zeros: bool = False, disable_charged_residue_mask: Optional[torch.Tensor] = None, repack_all: bool = False
    ) -> Sampled_Output:
        """
        If ignore_chain_mask_zeros is true, than ONLY sample residues that are True/1 in the chain mask and return XAA residues for all unsampled residues.
        otherwise, samples all residues EXCEPT those that are True/1 in the chain mask as it will take the input sequence/rotamer for these residues.
        disable_charged
        """
        # Input sanity checks.
        assert batch.pr_pr_edge_index is not None, "Protein-protein edge index must be specified in batch data."
        assert batch.pr_pr_edge_distance is not None, "Protein-protein edge distance must be specified in batch data."
        assert batch.lig_pr_edge_index is not None, "Protein-ligand edge index must be specified in batch data."
        assert batch.lig_pr_edge_distance is not None, "Protein-ligand edge distance must be specified in batch data."
        assert batch.decoding_order is not None, "Decoding order must be specified in batch data."
        if sequence_sample_temperature is not None:
            assert (isinstance(sequence_sample_temperature, torch.Tensor) and (sequence_sample_temperature.shape[0] == batch.num_residues or sequence_sample_temperature.numel() == 1)) or isinstance(sequence_sample_temperature, (int, float)), f"Sequence sample temperature must be a scalar or a tensor of shape (num_residues,). Got {sequence_sample_temperature}."

        # 1 in chain mask tells us to sample sequence, a 0 tells us to use the input sequence stored in batch.sequence_indices.
        chain_mask = batch.chain_mask.long()

        # Apply encoding layers to build up protein and ligand representations.
        prot_nodes, pr_pr_eattr = self.apply_encoding_layers(batch)

        # Initialize sequence embeddings to 'NotDecoded' nodes.
        sequence_embeddings = self.sequence_label_embedding(torch.full_like(batch.sequence_indices, fill_value=21))

        # Expanded source encoder node + edge features for source nodes that have not yet been decoded.
        masked_sequence_embedding = sequence_embeddings.clone()[batch.pr_pr_edge_index[0]]
        source_node_edge_features_exp_masked = torch.cat([masked_sequence_embedding, pr_pr_eattr, prot_nodes[batch.pr_pr_edge_index[0]]], dim=1)

        # Drops the NaN padding from the decoding order and flattens the batch dimension so 
        #   we can get sort indices of the same shape as number of nodes
        decoding_order_sort_indices = batch.decoding_order.argsort(dim=-1)[~batch.decoding_order.isnan()]

        # Initialize output tensors.
        prot_node_stack = [prot_nodes] + [torch.zeros_like(prot_nodes) for _ in range(self.num_decoder_layers)]
        output_tensors = create_sampling_output(batch.num_residues, self.device)

        # Iteratively decode protein nodes.
        for idx in tqdm(range(batch.decoding_order.shape[1]), total=batch.decoding_order.shape[1], leave=False, dynamic_ncols=True, desc='Running model.sample()', disable=disable_pbar):
            # Select current row in the batched decoding order.
            node_idces = batch.decoding_order[:, idx]
            node_idces = node_idces[~node_idces.isnan()].long()

            # 1 in chain mask tells us to take sequence/chi from input data, a 0 tells us to sample it with the model.
            curr_chain_mask = chain_mask[node_idces]

            if ignore_chain_mask_zeros:
                node_idces = node_idces[curr_chain_mask.bool()]
                if node_idces.numel() == 0:
                    continue

            # Get just the edges that are incident to the current node.
            edges_sink_curr_nodes_mask = torch.isin(batch.pr_pr_edge_index[1], node_idces)
            sink_curr_edge_indices = batch.pr_pr_edge_index[:, edges_sink_curr_nodes_mask]
            sink_curr_edge_features = pr_pr_eattr[edges_sink_curr_nodes_mask, :]

            # True if source node was decoded before target node, False otherwise.
            pr_pr_edge_mask = (decoding_order_sort_indices[sink_curr_edge_indices[0]] < decoding_order_sort_indices[sink_curr_edge_indices[1]])
            unmasked_edge_indices = sink_curr_edge_indices[:, pr_pr_edge_mask]
            unmasked_edge_attrs = sink_curr_edge_features[pr_pr_edge_mask]
            masked_edge_indices = sink_curr_edge_indices[:, ~pr_pr_edge_mask]
            masked_self_edges = source_node_edge_features_exp_masked[edges_sink_curr_nodes_mask][~pr_pr_edge_mask]

            lig_prot_edges_sink_curr_nodes_mask = torch.isin(batch.lig_pr_edge_index[1], node_idces)
            curr_lig_prot_eidx = batch.lig_pr_edge_index[:, lig_prot_edges_sink_curr_nodes_mask]

            # Extract previously decoded source node features.
            source_prev_sampled_labels = sequence_embeddings[unmasked_edge_indices[0]]

            curr_sink_node_edge_features_unmasked = torch.cat([source_prev_sampled_labels, unmasked_edge_attrs], dim=1)
            for layer_idx, decoder_layer in enumerate(self.protein_decoder_layers):
                # Provides node features for unmasked source nodes at current step of decoding.
                curr_idx_node_scalars_exp = prot_node_stack[layer_idx][unmasked_edge_indices[0]]
                curr_decoded_edge_features = torch.cat([curr_sink_node_edge_features_unmasked, curr_idx_node_scalars_exp], dim=1)

                # Add self-edges for nodes that have not yet been decoded.
                all_curr_edge_features = torch.cat([masked_self_edges, curr_decoded_edge_features], dim=0)
                all_curr_edge_indices = torch.cat([masked_edge_indices, unmasked_edge_indices], dim=1)

                # Update node embeddings for current step of decoding using edges terminating at current sink nodes.
                # updated_curr_nodes = decoder_layer(curr_decoded_edge_features, prot_node_stack[layer_idx], curr_decoded_edge_indices, lig_nodes, curr_lig_prot_eattr, curr_lig_prot_eidx).get_indices(node_idces)
                prot_nodes_ = decoder_layer(all_curr_edge_features, prot_node_stack[layer_idx], all_curr_edge_indices)
                prot_node_stack[layer_idx + 1] = prot_nodes_

            # Convert node embeddings to logits for sequence prediction.
            curr_out_logits = self.sequence_output_layer(prot_node_stack[-1][node_idces])
            if disabled_residues is not None:
                sampling_residue_mask = ~(curr_chain_mask.bool()) if not ignore_chain_mask_zeros else curr_chain_mask.bool()
                for res_short in disabled_residues:
                    curr_out_logits[sampling_residue_mask, aa_short_to_idx[res_short]] = torch.finfo(curr_out_logits.dtype).min
            
            if disable_charged_residue_mask is not None:
                curr_disable_charged_mask = disable_charged_residue_mask[node_idces]
                curr_out_logits[curr_disable_charged_mask, aa_short_to_idx['K']] = float('-Inf')
                curr_out_logits[curr_disable_charged_mask, aa_short_to_idx['R']] = float('-Inf')
                curr_out_logits[curr_disable_charged_mask, aa_short_to_idx['D']] = float('-Inf')
                curr_out_logits[curr_disable_charged_mask, aa_short_to_idx['E']] = float('-Inf')

            # Sample sequence indices if temperature is specified, otherwise take argmax.
            if sequence_sample_temperature is None:
                curr_out_sample = curr_out_logits.argmax(dim=-1)
            else:
                curr_out_logits = minp_warp_logits(curr_out_logits, seq_min_p)
                if isinstance(sequence_sample_temperature, torch.Tensor) and (sequence_sample_temperature.numel() > 1):
                    curr_out_probs = torch.softmax(curr_out_logits / sequence_sample_temperature[node_idces].unsqueeze(-1), dim=-1)
                else:
                    curr_out_probs = torch.softmax(curr_out_logits / sequence_sample_temperature, dim=-1)
                curr_out_sample = torch.distributions.Categorical(probs=curr_out_probs).sample()
            
            # Use chain_mask to select from input sequence for partial-sequence design as needed.
            if not ignore_chain_mask_zeros:
                sampled_or_fixed_sequence_idx = (curr_chain_mask * batch.sequence_indices[node_idces]) + ((1 - curr_chain_mask) * curr_out_sample)
            else:
                sampled_or_fixed_sequence_idx = curr_out_sample

            # Update sequence embeddings with sampled or fixed sequence indices.
            sequence_embeddings[node_idces] = self.sequence_label_embedding(sampled_or_fixed_sequence_idx)

            # Create masks for whether each chi angle is defined for each residue being decoded.
            #   Handle X residues by converting to Gly and sanity check we didn't sample them unless provided by chain_mask.
            sampled_x_residue_mask = torch.full_like(sampled_or_fixed_sequence_idx, aa_short_to_idx['X']) == sampled_or_fixed_sequence_idx
            if sampled_x_residue_mask.any().item() and (~batch.chain_mask[node_idces][sampled_x_residue_mask]).any().item():
                # Allow X residues to be sampled if not disabling them or chain_mask is 0, otherwise crashes with assertion error.
                assert (disabled_residues is None) or (not 'X' in disabled_residues), "Sampled an X residue when sampling X residues is disabled."
            x_to_gly_sampled_or_fixed_sequence_idx = sampled_or_fixed_sequence_idx.clone()
            x_to_gly_sampled_or_fixed_sequence_idx[sampled_x_residue_mask] = aa_short_to_idx['G']

            # Store sequence logits and sampled sequence indices in output tensors.
            output_tensors.sequence_logits[node_idces] = curr_out_logits
            output_tensors.sampled_sequence_indices[node_idces] = sampled_or_fixed_sequence_idx

        if ignore_chain_mask_zeros:
            output_tensors.sampled_sequence_indices[~batch.chain_mask] = aa_short_to_idx['X']
        
        if return_encoder_embeddings:
            raise NotImplementedError

        return output_tensors


class PositionWiseFeedForward(nn.Module):
    def __init__(self, io_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()

        self.W_in = nn.Linear(io_dim, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, io_dim)
        self.act = nn.GELU()
    
    def forward(self, nodes):
        return self.W_out(self.act(self.W_in(nodes)))


class LigandMPNNProteinEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.1, scale=30):
        super(LigandMPNNProteinEncoderLayer, self).__init__()

        self.forward1 = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.forward2 = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.dense = PositionWiseFeedForward(hidden_dim, 4 * hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = scale
    
    def forward(self, nodes, edges, edge_index):
        # Node Update
        source_nodes_exp = nodes[edge_index[0]]
        sink_nodes_exp = nodes[edge_index[1]]

        h_EV = torch.cat([source_nodes_exp, sink_nodes_exp, edges], dim=1)
        h_message = self.forward1(h_EV)
        node_update = scatter(h_message, edge_index[1], dim=0, dim_size=nodes.shape[0], reduce='sum') / self.scale
        nodes = self.norm1(nodes + self.dropout(node_update))

        node_update = self.dense(nodes)
        nodes = self.norm2(nodes + self.dropout(node_update))

        # Handle Edge Update
        source_nodes_exp = nodes[edge_index[0]]
        sink_nodes_exp = nodes[edge_index[1]]
        h_EV = torch.cat([source_nodes_exp, sink_nodes_exp, edges], dim=1)
        edge_update = self.forward2(h_EV)
        edges = self.norm3(edges + self.dropout(edge_update))

        return nodes, edges
    

class LigandMPNNDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.1, scale=30):
        super(LigandMPNNDecoderLayer, self).__init__()

        self.forward1 = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dense = PositionWiseFeedForward(hidden_dim, 4 * hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = scale

    def forward(self, masked_node_and_edge, nodes, e_idx):

        h_EV = torch.cat([masked_node_and_edge, nodes[e_idx[1]]], dim=1)
        h_message = self.forward1(h_EV)
        if self.scale == -1:
            node_update = scatter(h_message, e_idx[1], dim=0, dim_size=nodes.shape[0], reduce='mean')
        else:
            node_update = scatter(h_message, e_idx[1], dim=0, dim_size=nodes.shape[0], reduce='sum') / self.scale
        nodes = self.norm1(nodes + self.dropout(node_update))

        node_update = self.dense(nodes)
        nodes = self.norm2(nodes + self.dropout(node_update))

        return nodes


class RBF_Encoding(nn.Module):
    """
    Implements the RBF Encoding from ProteinMPNN as a module that can get stored in the model.
    """
    def __init__(self, num_bins: int, bin_min: float, bin_max: float):
        super(RBF_Encoding, self).__init__()
        self.num_bins = num_bins
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.D_sigma =  (bin_max - bin_min) / num_bins
        self.register_buffer('D_mu', torch.linspace(bin_min, bin_max, num_bins).view([1,-1]))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances in last dimension to RBF encoding in an expanded (num_bins) dimension
            (N, M)  -->  (N, M, num_bins)
        """
        D_expand = torch.unsqueeze(distances, -1)
        rbf_encoding = torch.exp(-((D_expand - self.D_mu) / self.D_sigma)**2) + 1e-4
        return rbf_encoding


def _make_angle_features(bb_coords, lig_coords, lig_pr_eidx):
    """
    Computes local angular features for ligand atoms relative to the protein backbone.
    Partially generated from LigandMPNN code by ChatGPT.
    """
    # Expand the frames to the size of the edges (E, 3).
    lig_coords_exp = lig_coords[lig_pr_eidx[0]]
    bb_coords_exp = bb_coords[lig_pr_eidx[1]]
    N_exp, Ca_exp, C_exp = bb_coords_exp[:, [0, 1, 3]].unbind(dim=1)

    # 'Gram-Schmidt' orthogonalization
    v1 = N_exp - Ca_exp 
    v2 = C_exp - Ca_exp 
    e1 = F.normalize(v1, dim=-1)  # (N, 3)
    e1_v2_dot = torch.einsum("ni, ni -> n", e1, v2)[:, None]  # (N, 1)
    u2 = v2 - e1 * e1_v2_dot  # (N, 3)
    e2 = torch.nn.functional.normalize(u2, dim=-1)  # (N, 3)
    e3 = torch.cross(e1, e2, dim=-1)  # (N, 3)
    R_residue = torch.stack((e1, e2, e3), dim=-1)  # (N, 3, 3)
    
    # Compute local vectors
    Y_local = lig_coords_exp - Ca_exp # Assuming Y is (N, 3)
    local_vectors = torch.einsum("nqp, np -> nq", R_residue, Y_local)  # (N, 3)
    
    rxy = torch.sqrt(local_vectors[:, 0] ** 2 + local_vectors[:, 1] ** 2 + 1e-8)  # (N,)
    f1 = local_vectors[:, 0] / rxy  # (N,)
    f2 = local_vectors[:, 1] / rxy  # (N,)
    
    rxyz = torch.norm(local_vectors, dim=-1) + 1e-8  # (N,)
    f3 = rxy / rxyz  # (N,)
    f4 = local_vectors[:, 2] / rxyz  # (N,)
    
    f = torch.stack([f1, f2, f3, f4], dim=-1)  # (N, 4)
    return f
