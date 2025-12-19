import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import atomic_number_to_atom, atom_to_period_idx, atom_to_group_idx, sequence_index_to_atomic_numbers, MAX_NUM_PROTONATED_RESIDUE_ATOMS, atom_to_atomic_number, period_group_tuple_to_atom_symbol 
from typing import Sequence, Tuple

class LigandFeaturizer(nn.Module):
    """
    Converts ligand node features to node features for input to model.
    """
    def __init__(self, build_hydrogens, **kwargs):
        super(LigandFeaturizer, self).__init__()

        # Whether to remove hydrogens from the featurization.
        self.build_hydrogens = build_hydrogens

        # Define helper dictionaries.
        # Fill in NaN values for Lanthanides and Actinides with their own group number.
        self.atom_to_period_idx = atom_to_period_idx
        self.atom_to_group_idx = {x: (y if (y == y) else max(atom_to_group_idx.values()) + 1) for x,y in atom_to_group_idx.items()}

        # The dimension of the concatenated OHEs for each atom.
        self.num_period_classes = max(self.atom_to_period_idx.values()) + 1
        self.num_group_classes = max(self.atom_to_group_idx.values()) + 1
        self.embedding_dim = self.num_period_classes + self.num_group_classes

        # Create tensor mapping from atomic number as index 
        self.register_buffer('atomic_number_idx_to_period_idx', torch.tensor([self.atom_to_period_idx[x] for x in atomic_number_to_atom.values()]))
        self.register_buffer('atomic_number_idx_to_group_idx',  torch.tensor([self.atom_to_group_idx[x] for x in atomic_number_to_atom.values()]))
        self.register_buffer('sequence_index_to_atomic_number_index', (sequence_index_to_atomic_numbers - 1).clamp_min_(-1)) # type: ignore

        atomic_number_index_mask = self.sequence_index_to_atomic_number_index == -1
        residue_period_indices = self.atomic_number_idx_to_period_idx[self.sequence_index_to_atomic_number_index] # type: ignore
        residue_period_indices[atomic_number_index_mask] = -1
        residue_group_indices = self.atomic_number_idx_to_group_idx[self.sequence_index_to_atomic_number_index] # type: ignore
        residue_group_indices[atomic_number_index_mask] = -1

        amino_acid_index_to_featurization_indices = torch.full((21, MAX_NUM_PROTONATED_RESIDUE_ATOMS + 1, self.embedding_dim), torch.nan)
        encodings = torch.cat([
            F.one_hot(residue_period_indices[~atomic_number_index_mask], num_classes=self.num_period_classes),
            F.one_hot(residue_group_indices[~atomic_number_index_mask], num_classes=self.num_group_classes)
        ], dim=1)

        amino_acid_index_to_featurization_indices[~atomic_number_index_mask] = encodings.float()
        
        self.register_buffer('amino_acid_index_to_featurization', amino_acid_index_to_featurization_indices)
        self.register_buffer('amino_acid_index_to_is_hydrogen_mask', self.sequence_index_to_atomic_number_index == 0) # type: ignore

        self.output_dim = int(self.amino_acid_index_to_featurization.shape[-1])  # type: ignore

        hydrogen_encoding = self.encode_ligand_from_atom_string_sequence(['H'])
        self.register_buffer('hydrogen_encoding', hydrogen_encoding)

    def drop_hydrogens_from_coord_tensor(self, coord_tensor: torch.Tensor, residue_index: torch.Tensor) -> torch.Tensor:
        """
        Removes hydrogens coordinates from tensor of coordinates.
            TODO: should probably implement building with/without hydrogens as part of RotamerBuilder object.
        """
        return coord_tensor[~self.amino_acid_index_to_is_hydrogen_mask[residue_index]].float() # type: ignore

    def masked_generate_ligand_nodes_from_amino_acid_labels(self, sequence_indices: torch.Tensor, batch_indices: torch.Tensor, subbatch_indices: torch.Tensor, not_nanmask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Using precomputed tensors of features for each 
        """
        expanded_features = self.amino_acid_index_to_featurization[sequence_indices] # type: ignore
        expanded_atomic_numbers = self.sequence_index_to_atomic_number_index[sequence_indices] # type: ignore

        flat_features_exp = expanded_features.flatten(end_dim=1) 
        flat_atomic_numbers = expanded_atomic_numbers.flatten()
        features_exp_mask = ~(flat_features_exp.isnan().any(dim=-1)) & not_nanmask.flatten()

        if not self.build_hydrogens:
            is_hydrogen_mask = self.amino_acid_index_to_is_hydrogen_mask[sequence_indices].flatten(end_dim=1) # type: ignore
            features_exp_mask = features_exp_mask & (~is_hydrogen_mask)

        flat_indices = batch_indices.reshape(-1, 1).expand(expanded_features.shape[:2]).flatten()
        flat_subbatch_indices = subbatch_indices.reshape(-1, 1).expand(expanded_features.shape[:2]).flatten()
        return flat_features_exp[features_exp_mask].float(), flat_indices[features_exp_mask], flat_subbatch_indices[features_exp_mask], flat_atomic_numbers[features_exp_mask]
    
    def generate_ligand_nodes_from_amino_acid_labels(self, sequence_indices: torch.Tensor, batch_indices: torch.Tensor, subbatch_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Using precomputed tensors of features for each 
        """
        expanded_features = self.amino_acid_index_to_featurization[sequence_indices] # type: ignore
        expanded_atomic_numbers = self.sequence_index_to_atomic_number_index[sequence_indices] # type: ignore

        flat_features_exp = expanded_features.flatten(end_dim=1) 
        flat_atomic_numbers = expanded_atomic_numbers.flatten()
        features_exp_mask = ~(flat_features_exp.isnan().any(dim=-1))
        is_hydrogen_mask = self.amino_acid_index_to_is_hydrogen_mask[sequence_indices].flatten(end_dim=1) # type: ignore

        if not self.build_hydrogens:
            features_exp_mask = features_exp_mask & (~is_hydrogen_mask)

        flat_indices = batch_indices.reshape(-1, 1).expand(expanded_features.shape[:2]).flatten()
        flat_subbatch_indices = subbatch_indices.reshape(-1, 1).expand(expanded_features.shape[:2]).flatten()
        return flat_features_exp[features_exp_mask].float(), flat_indices[features_exp_mask], flat_subbatch_indices[features_exp_mask], flat_atomic_numbers[features_exp_mask]
    
    def encode_ligand_from_atomic_number(self, ligand_atomic_numbers: torch.Tensor):
        """
        """
        # Encode ligand as OHE of period and group.
        encoding = torch.cat([
            F.one_hot(self.atomic_number_idx_to_period_idx[ligand_atomic_numbers - 1], num_classes=self.num_period_classes), # type: ignore
            F.one_hot(self.atomic_number_idx_to_group_idx[ligand_atomic_numbers - 1], num_classes=self.num_group_classes)], dim=1) # type: ignore
        return encoding.float()

    def encode_ligand_from_atomic_number_indices(self, ligand_atomic_number_indices: torch.Tensor):
        return self.encode_ligand_from_atomic_number(ligand_atomic_number_indices + 1)

    def encode_ligand_from_atom_string_sequence(self, atom_string_sequence: Sequence, device: torch.device = torch.device('cpu')):
        # Convert atom strings to atomic numbers.
        ligand_atomic_numbers = [atom_to_atomic_number[x] for x in atom_string_sequence]

        # Convert to tensor and encode.
        return self.encode_ligand_from_atomic_number(torch.tensor(ligand_atomic_numbers, device=device))

    def encoding_to_is_hydrogen_mask(self, ligand_encoding: torch.Tensor):
        return (ligand_encoding == self.hydrogen_encoding).all(dim=-1)
    
    def encoding_to_atom_string_sequence(self, ligand_encoding: torch.Tensor):
        """
        """
        output = []

        # Get the period and group indices.
        period_idx = ligand_encoding[:, :self.num_period_classes].argmax(dim=-1)
        group_idx = ligand_encoding[:, self.num_period_classes:].argmax(dim=-1)
        for period, group in torch.stack([period_idx, group_idx]).T.cpu().tolist():
            symbol = period_group_tuple_to_atom_symbol[(period, group)]
            output.append(symbol)

        return output
    