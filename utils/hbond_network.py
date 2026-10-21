import torch
import torch.nn as nn
import torch.nn.functional as F
from .ligand_featurization import LigandFeaturizer
from .build_rotamers import RotamerBuilder
from .constants import ON_ON_HYDROGEN_BOND_DISTANCE_CUTOFF, ON_S_HYDROGEN_BOND_DISTANCE_CUTOFF, S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF, MAX_NUM_PROTONATED_RESIDUE_ATOMS, MIN_HBOND_ANGLE, COVALENT_HYDROGEN_BOND_MAX_DISTANCE, GENERIC_COVALENT_BOND_MAX_DISTANCE, POSSIBLE_HYBRIDIZATION_LIST, atom_to_atomic_number, aa_to_hbond_donor_index_map, aa_to_hbond_acceptor_map, aa_short_to_idx, aa_idx_to_short
from .scatter_ops import scatter
from typing import Tuple, Optional

MIN_HBOND_DISTANCE = 2.3

def compute_angles(donor_coords: torch.Tensor, h_coords: torch.Tensor, acceptor_coords: torch.Tensor) -> torch.Tensor:
    """
    Compute the angle between the donor-hydrogen bond and the acceptor-donor bond.
    inputs:
        donor_coords: (N, 1, 3)
        h_coords: (N, M, 3)
        acceptor_coords: (N, 1, 3)

    returns:
        angles: (N, M) in degrees
    """
    b_to_a = donor_coords - h_coords
    b_to_c = acceptor_coords - h_coords
    angle = torch.acos((b_to_a * b_to_c).sum(dim=-1) / (torch.norm(b_to_a, dim=-1) * torch.norm(b_to_c, dim=-1))).rad2deg()
    return angle


def compute_hbonding_connected_component(first_shell_indices, fs_node_hbonding_mask, hbond_network_eidx, include_fs_indirect_hbonding=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative BFS-like search for hydrogen bonding connected component.
    Returns:
        hbond_connected_idces: indices of the protein residues in the hydrogen bonding connected component.
        shell_index: shell index of each residue in the connected component.
    """
    # Identify the first shell residues that are connected by hydrogen bonds.
    curr_hbond_network_eidx = hbond_network_eidx
    if include_fs_indirect_hbonding:
        hbond_connected_idces = first_shell_indices[torch.isin(first_shell_indices, hbond_network_eidx) | fs_node_hbonding_mask]
    else:
        hbond_connected_idces = first_shell_indices[fs_node_hbonding_mask]
    shell_index = torch.zeros_like(hbond_connected_idces, dtype=torch.long)

    niter = 0
    # Continue outward by shells until no new residues are added.
    while True:
        # Identify the edges that are connected to our previously identified hydrogen bonding indices.
        connected_mask = torch.isin(curr_hbond_network_eidx[0], hbond_connected_idces) | torch.isin(curr_hbond_network_eidx[1], hbond_connected_idces)
        hbond_connected_residues = curr_hbond_network_eidx[:,  connected_mask]

        # Get the unique set of indices that are connected to our current hydrogen bonding indices.
        new_indices = hbond_connected_residues.flatten().unique()
        new_indices = new_indices[~torch.isin(new_indices, hbond_connected_idces)]

        if new_indices.shape[0] == 0:
            # If we have no new indices, finished searching.
            return hbond_connected_idces, shell_index

        # Add the new indices to our list of hydrogen bonding indices.
        hbond_connected_idces = torch.cat([hbond_connected_idces, new_indices])
        shell_index = torch.cat([
            shell_index,
            torch.full((new_indices.shape[0],), niter + 1, dtype=torch.long, device=shell_index.device)
        ])

        # Remove the edges that we have already visited.
        curr_hbond_network_eidx = curr_hbond_network_eidx[:, ~connected_mask]
        niter += 1


class RigorousHydrogenBondNetworkDetector(nn.Module):
    def __init__(self):
        super(RigorousHydrogenBondNetworkDetector, self).__init__()

        self.register_buffer('hbonding_atom_indices', torch.tensor([atom_to_atomic_number['N'], atom_to_atomic_number['O'], atom_to_atomic_number['S']]) - 1)
        self.register_buffer('nitrogen_or_oxygen_indices', torch.tensor([atom_to_atomic_number['N'], atom_to_atomic_number['O']]) - 1)
        self.register_buffer('sulfur_indices', torch.tensor([atom_to_atomic_number['S']]) - 1)
        self.register_buffer('sidechain_start_index', torch.tensor(3))

        # For getting the atom indices of the attached hydrogens to the donor atoms.
        hbond_aa_donor_index_set = [aa_short_to_idx[x] for x in aa_to_hbond_donor_index_map]
        self.register_buffer('hbond_aa_donor_index_set', torch.tensor(hbond_aa_donor_index_set))

        # Convert the aa_to_hbond_donor_index_map to a tensor.
        hbond_donor_to_hydrogen_index_map = torch.full((20, MAX_NUM_PROTONATED_RESIDUE_ATOMS, max(len(x) for x in aa_to_hbond_donor_index_map.values())), MAX_NUM_PROTONATED_RESIDUE_ATOMS, dtype=torch.long)
        for res, hbond_map in aa_to_hbond_donor_index_map.items():
            for hbond_donor, hydrogen_index_list in hbond_map.items():
                for hidx, hydrogen_index in enumerate(hydrogen_index_list):
                    hbond_donor_to_hydrogen_index_map[aa_short_to_idx[res], hbond_donor, hidx] = hydrogen_index
        self.register_buffer('hbond_donor_aa_idx_to_donor_heavy_pair', hbond_donor_to_hydrogen_index_map)

        hbond_aa_acceptor_index_set = [aa_short_to_idx[x] for x in aa_to_hbond_acceptor_map]
        self.register_buffer('hbond_aa_acceptor_index_set', torch.tensor(hbond_aa_acceptor_index_set))

        # Since every residue has O, don't need a remap tensor.
        hbond_acceptor_aa_idx_to_acceptor_indices = torch.full((20, MAX_NUM_PROTONATED_RESIDUE_ATOMS), False, dtype=torch.bool)
        for res, indices in aa_to_hbond_acceptor_map.items():
            for index in indices:
                hbond_acceptor_aa_idx_to_acceptor_indices[aa_short_to_idx[res], index] = True
        self.register_buffer('hbond_acceptor_aa_idx_to_acceptor_indices', hbond_acceptor_aa_idx_to_acceptor_indices)

        # Get the indices of the backbone donor atoms.
        self.register_buffer('backbone_donor_index', torch.tensor([0])) # Index of 'N' in fa_coords tensors.

    def _compute_protein_protein_hbond_contacts(
            self, ligand_featurizer, edge_sequence_indices, edge_tensor_indices, source_residue_atom_indices, sink_residue_atom_indices, 
            distance_matrix, fa_coords, pr_pr_eidx, bb_nh_coords, use_min_distance=False, return_hbond_values=False
        ):
        # Expands sequence indices to the atomic numbers of each atom. Remove the padding index.
        edge_sequence_atomic_numbers = ligand_featurizer.sequence_index_to_atomic_number_index[edge_sequence_indices]
        edge_sequence_indices = edge_sequence_indices[:, edge_tensor_indices]

        # Get atomic numbers and distances of source and sink atoms.
        source_atomic_numbers = edge_sequence_atomic_numbers[0][edge_tensor_indices, source_residue_atom_indices]
        sink_atomic_numbers = edge_sequence_atomic_numbers[1][edge_tensor_indices, sink_residue_atom_indices]
        distances = distance_matrix[edge_tensor_indices, source_residue_atom_indices, sink_residue_atom_indices]

        # Filter for only pairs of atoms that are capable of hydrogen bonding.
        possible_hydrogen_bonding_atoms_mask = torch.isin(source_atomic_numbers, self.hbonding_atom_indices) & torch.isin(sink_atomic_numbers, self.hbonding_atom_indices) # type: ignore

        # Apply hydrogen bond candidate mask to all tensors.
        edge_tensor_indices = edge_tensor_indices[possible_hydrogen_bonding_atoms_mask]
        source_residue_atom_indices = source_residue_atom_indices[possible_hydrogen_bonding_atoms_mask]
        sink_residue_atom_indices = sink_residue_atom_indices[possible_hydrogen_bonding_atoms_mask]
        source_atomic_numbers = source_atomic_numbers[possible_hydrogen_bonding_atoms_mask]
        sink_atomic_numbers = sink_atomic_numbers[possible_hydrogen_bonding_atoms_mask]
        distances = distances[possible_hydrogen_bonding_atoms_mask]
        edge_sequence_indices = edge_sequence_indices[:, possible_hydrogen_bonding_atoms_mask]

        # Identify from this subset the different types of hydrogen bonds.
        oxy_or_nit_pair = torch.isin(source_atomic_numbers, self.nitrogen_or_oxygen_indices) & torch.isin(sink_atomic_numbers, self.nitrogen_or_oxygen_indices) # type: ignore
        sulfur_sulfur_pair = torch.isin(source_atomic_numbers, self.sulfur_indices) & torch.isin(sink_atomic_numbers, self.sulfur_indices) # type: ignore
        oxy_nit_to_sulfur_pair = (~oxy_or_nit_pair) & (~sulfur_sulfur_pair)

        # Apply distance cutoffs to each type of hydrogen bond.
        coarse_hydrogen_bond_mask = torch.zeros(edge_tensor_indices.shape[0], dtype=torch.bool, device=edge_tensor_indices.device)
        if use_min_distance:
            coarse_hydrogen_bond_mask[oxy_or_nit_pair] = (distances[oxy_or_nit_pair] < ON_ON_HYDROGEN_BOND_DISTANCE_CUTOFF) & (distances[oxy_or_nit_pair] > MIN_HBOND_DISTANCE)
            coarse_hydrogen_bond_mask[oxy_nit_to_sulfur_pair] = (distances[oxy_nit_to_sulfur_pair] < ON_S_HYDROGEN_BOND_DISTANCE_CUTOFF) & (distances[oxy_nit_to_sulfur_pair] > MIN_HBOND_DISTANCE)
            coarse_hydrogen_bond_mask[sulfur_sulfur_pair] = (distances[sulfur_sulfur_pair] < S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF) & (distances[sulfur_sulfur_pair] > MIN_HBOND_DISTANCE)
        else:
            coarse_hydrogen_bond_mask[oxy_or_nit_pair] = distances[oxy_or_nit_pair] < ON_ON_HYDROGEN_BOND_DISTANCE_CUTOFF
            coarse_hydrogen_bond_mask[oxy_nit_to_sulfur_pair] = distances[oxy_nit_to_sulfur_pair] < ON_S_HYDROGEN_BOND_DISTANCE_CUTOFF
            coarse_hydrogen_bond_mask[sulfur_sulfur_pair] = distances[sulfur_sulfur_pair] < S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF

        # Use coarse mask to subset the edge information.
        edge_tensor_indices = edge_tensor_indices[coarse_hydrogen_bond_mask]
        source_residue_atom_indices = source_residue_atom_indices[coarse_hydrogen_bond_mask]
        sink_residue_atom_indices = sink_residue_atom_indices[coarse_hydrogen_bond_mask]
        edge_sequence_indices = edge_sequence_indices[:, coarse_hydrogen_bond_mask]

        # Use sequence indices and atom indices to identify donor/acceptor pairs,
        source_donor_hydrogen_indices = self.hbond_donor_aa_idx_to_donor_heavy_pair[edge_sequence_indices[0], source_residue_atom_indices] # type: ignore
        source_donor_mask = source_donor_hydrogen_indices != MAX_NUM_PROTONATED_RESIDUE_ATOMS
        source_backbone_h_donor_mask = source_residue_atom_indices == self.backbone_donor_index
        source_is_acceptor = self.hbond_acceptor_aa_idx_to_acceptor_indices[edge_sequence_indices[0], source_residue_atom_indices] # type: ignore

        sink_donor_hydrogen_indices = self.hbond_donor_aa_idx_to_donor_heavy_pair[edge_sequence_indices[1], sink_residue_atom_indices]  # type: ignore
        sink_donor_mask = sink_donor_hydrogen_indices != MAX_NUM_PROTONATED_RESIDUE_ATOMS
        sink_backbone_h_donor_mask = sink_residue_atom_indices == self.backbone_donor_index
        sink_is_acceptor = self.hbond_acceptor_aa_idx_to_acceptor_indices[edge_sequence_indices[1], sink_residue_atom_indices] # type: ignore

        # Handle the case where the source residue is the donor and the sink residue is an acceptor.
        source_pair_mask = source_donor_mask.any(dim=-1) & sink_is_acceptor
        pair1_donor_coords = fa_coords[pr_pr_eidx[0][edge_tensor_indices[source_pair_mask]]].gather(1, source_residue_atom_indices[source_pair_mask].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3))
        pair1_donorh_coords = fa_coords[pr_pr_eidx[0][edge_tensor_indices[source_pair_mask]]].gather(1, source_donor_hydrogen_indices[source_pair_mask].unsqueeze(-1).expand(-1, -1, 3))
        pair1_acceptor_coords = fa_coords[pr_pr_eidx[1][edge_tensor_indices[source_pair_mask]]].gather(1, sink_residue_atom_indices[source_pair_mask].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3))
        pair1_angles = compute_angles(pair1_acceptor_coords, pair1_donorh_coords, pair1_donor_coords)
        pair1_angle_value = ((pair1_angles - MIN_HBOND_ANGLE) / (180.0 - MIN_HBOND_ANGLE)).clamp(min=0.0)
        pair1_angle_valid_mask = (pair1_angles > MIN_HBOND_ANGLE).any(dim=-1)
        source_pair_mask[source_pair_mask.clone()] = pair1_angle_valid_mask

        backbone_pair1_mask = source_backbone_h_donor_mask & sink_is_acceptor
        backbone_pair1_donor_coords = fa_coords[pr_pr_eidx[0][edge_tensor_indices[backbone_pair1_mask]]].gather(1, source_residue_atom_indices[backbone_pair1_mask].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3))
        backbone_pair1_donor_hydrogens = bb_nh_coords[pr_pr_eidx[0][edge_tensor_indices[backbone_pair1_mask]]]
        backbone_pair1_acceptor_coords = fa_coords[pr_pr_eidx[1][edge_tensor_indices[backbone_pair1_mask]]].gather(1, sink_residue_atom_indices[backbone_pair1_mask].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3))
        backbone_pair1_angles = compute_angles(backbone_pair1_acceptor_coords, backbone_pair1_donor_hydrogens.unsqueeze(1), backbone_pair1_donor_coords)
        backbone_pair1_value = ((backbone_pair1_angles - MIN_HBOND_ANGLE) / (180.0 - MIN_HBOND_ANGLE)).clamp(min=0.0)
        backbone_pair1_angle_valid_mask = (backbone_pair1_angles > MIN_HBOND_ANGLE).any(dim=-1)
        backbone_pair1_mask[backbone_pair1_mask.clone()] = backbone_pair1_angle_valid_mask

        # Handle the case where the sink residue is the donor and the source residue is an acceptor.
        sink_pair_mask = sink_donor_mask.any(dim=-1) & source_is_acceptor
        pair2_donor_coords = fa_coords[pr_pr_eidx[1][edge_tensor_indices[sink_pair_mask]]].gather(1, sink_residue_atom_indices[sink_pair_mask].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3))
        pair2_donorh_coords = fa_coords[pr_pr_eidx[1][edge_tensor_indices[sink_pair_mask]]].gather(1, sink_donor_hydrogen_indices[sink_pair_mask].unsqueeze(-1).expand(-1, -1, 3))
        pair2_acceptor_coords = fa_coords[pr_pr_eidx[0][edge_tensor_indices[sink_pair_mask]]].gather(1, source_residue_atom_indices[sink_pair_mask].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3))
        pair2_angles = compute_angles(pair2_donor_coords, pair2_donorh_coords, pair2_acceptor_coords)
        pair2_angle_value = ((pair2_angles - MIN_HBOND_ANGLE) / (180.0 - MIN_HBOND_ANGLE)).clamp(min=0.0)
        pair2_angle_valid_mask = (pair2_angles > MIN_HBOND_ANGLE).any(dim=-1)
        sink_pair_mask[sink_pair_mask.clone()] = pair2_angle_valid_mask

        backbone_pair2_mask = sink_backbone_h_donor_mask & source_is_acceptor
        backbone_pair2_donor_coords = fa_coords[pr_pr_eidx[1][edge_tensor_indices[backbone_pair2_mask]]].gather(1, sink_residue_atom_indices[backbone_pair2_mask].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3))
        backbone_pair2_donor_hydrogens = bb_nh_coords[pr_pr_eidx[1][edge_tensor_indices[backbone_pair2_mask]]]
        backbone_pair2_acceptor_coords = fa_coords[pr_pr_eidx[0][edge_tensor_indices[backbone_pair2_mask]]].gather(1, source_residue_atom_indices[backbone_pair2_mask].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3))
        backbone_pair2_angles = compute_angles(backbone_pair2_donor_coords, backbone_pair2_donor_hydrogens.unsqueeze(1), backbone_pair2_acceptor_coords)
        backbone_pair2_value = ((backbone_pair2_angles - MIN_HBOND_ANGLE) / (180.0 - MIN_HBOND_ANGLE)).clamp(min=0.0)
        backbone_pair2_angle_valid_mask = (backbone_pair2_angles > MIN_HBOND_ANGLE).any(dim=-1)
        backbone_pair2_mask[backbone_pair2_mask.clone()] = backbone_pair2_angle_valid_mask

        # Aggregate masks to get the final hydrogen bond network.
        final_mask = source_pair_mask | backbone_pair1_mask | sink_pair_mask | backbone_pair2_mask

        values_tensor = torch.zeros_like(final_mask, dtype=torch.float)
        values_tensor[source_pair_mask] = values_tensor[source_pair_mask] + pair1_angle_value[pair1_angle_valid_mask].nan_to_num().sum(dim=-1)
        values_tensor[backbone_pair1_mask] = values_tensor[backbone_pair1_mask] + backbone_pair1_value[backbone_pair1_angle_valid_mask].nan_to_num().sum(dim=-1)
        values_tensor[sink_pair_mask] = values_tensor[sink_pair_mask] + pair2_angle_value[pair2_angle_valid_mask].nan_to_num().sum(dim=-1)
        values_tensor[backbone_pair2_mask] = values_tensor[backbone_pair2_mask] + backbone_pair2_value[backbone_pair2_angle_valid_mask].nan_to_num().sum(dim=-1)

        if not return_hbond_values:
            return edge_tensor_indices[final_mask], source_residue_atom_indices[final_mask], sink_residue_atom_indices[final_mask]
        return edge_tensor_indices[final_mask], source_residue_atom_indices[final_mask], sink_residue_atom_indices[final_mask], values_tensor[final_mask]

    def _compute_protein_ligand_hbond_contacts(
            self, ligand_atom_atomic_numbers_exp, sink_residue_sequence_indices, 
            distance_matrix, edge_tensor_indices, sink_residue_atom_indices, ligand_featurizer,
            full_lig_lig_eidx, hbond_capable_ligand_node_mask, ligand_atomic_number_indices, 
            ligand_coords, fs_hbonding_lig_pr_edges, fa_coords, 
            ligand_atom_hybridization_idces: Optional[torch.Tensor] = None, return_actually_hbonding=False, use_min_distance=False, return_values=False
    ):
        # Get the atomic numbers for the ligand atoms and protein atoms we identified.
        ligand_atom_atomic_numbers_exp = ligand_atom_atomic_numbers_exp[edge_tensor_indices]
        sink_index_atomic_numbers = ligand_featurizer.sequence_index_to_atomic_number_index[sink_residue_sequence_indices]
        sink_index_atomic_numbers = sink_index_atomic_numbers[edge_tensor_indices, sink_residue_atom_indices]
        sink_residue_sequence_indices = sink_residue_sequence_indices[edge_tensor_indices]
        distances = distance_matrix[edge_tensor_indices, sink_residue_atom_indices]

        # Remove atoms in residue that aren't capable of hydrogen bonding.
        possible_hbonding_atoms_mask = torch.isin(sink_index_atomic_numbers, self.hbonding_atom_indices) # type: ignore
        ligand_atom_atomic_numbers_exp = ligand_atom_atomic_numbers_exp[possible_hbonding_atoms_mask]
        sink_index_atomic_numbers = sink_index_atomic_numbers[possible_hbonding_atoms_mask]
        edge_tensor_indices = edge_tensor_indices[possible_hbonding_atoms_mask]
        sink_residue_atom_indices = sink_residue_atom_indices[possible_hbonding_atoms_mask]
        sink_residue_sequence_indices = sink_residue_sequence_indices[possible_hbonding_atoms_mask]
        distances = distances[possible_hbonding_atoms_mask]

        # Create masks by hydrogen bond type.
        oxy_or_nit_pair = torch.isin(ligand_atom_atomic_numbers_exp, self.nitrogen_or_oxygen_indices) & torch.isin(sink_index_atomic_numbers, self.nitrogen_or_oxygen_indices) # type: ignore
        sulfur_sulfur_pair = torch.isin(ligand_atom_atomic_numbers_exp, self.sulfur_indices) & torch.isin(sink_index_atomic_numbers, self.sulfur_indices) # type: ignore
        oxy_nit_to_sulfur_pair = (~oxy_or_nit_pair) & (~sulfur_sulfur_pair)

        # Apply distance cutoffs to each type of hydrogen bond atom pair.
        coarse_hydrogen_bond_mask = torch.zeros(edge_tensor_indices.shape[0], dtype=torch.bool, device=edge_tensor_indices.device)
        if use_min_distance:
            coarse_hydrogen_bond_mask[oxy_or_nit_pair] = (distances[oxy_or_nit_pair] < ON_ON_HYDROGEN_BOND_DISTANCE_CUTOFF) & (distances[oxy_or_nit_pair] > MIN_HBOND_DISTANCE)
            coarse_hydrogen_bond_mask[oxy_nit_to_sulfur_pair] = (distances[oxy_nit_to_sulfur_pair] < ON_S_HYDROGEN_BOND_DISTANCE_CUTOFF) & (distances[oxy_nit_to_sulfur_pair] > MIN_HBOND_DISTANCE)
            coarse_hydrogen_bond_mask[sulfur_sulfur_pair] = (distances[sulfur_sulfur_pair] < S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF) & (distances[sulfur_sulfur_pair] > MIN_HBOND_DISTANCE)
        else:
            coarse_hydrogen_bond_mask[oxy_or_nit_pair] = distances[oxy_or_nit_pair] < ON_ON_HYDROGEN_BOND_DISTANCE_CUTOFF
            coarse_hydrogen_bond_mask[oxy_nit_to_sulfur_pair] = distances[oxy_nit_to_sulfur_pair] < ON_S_HYDROGEN_BOND_DISTANCE_CUTOFF
            coarse_hydrogen_bond_mask[sulfur_sulfur_pair] = distances[sulfur_sulfur_pair] < S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF

        edge_tensor_indices = edge_tensor_indices[coarse_hydrogen_bond_mask]
        sink_index_atomic_numbers = sink_index_atomic_numbers[coarse_hydrogen_bond_mask]
        sink_residue_atom_indices = sink_residue_atom_indices[coarse_hydrogen_bond_mask]
        sink_residue_sequence_indices = sink_residue_sequence_indices[coarse_hydrogen_bond_mask]

        # Figure out donor/acceptor pairs for each edge.
        # TODO: can use pretrained ligand encoder features to predict hybridizations and num attached hydrogens.
        # TODO: should identify intra-molecular hydrogen bonds as part of an eventual BUN calculation.
        ### Start by figuring out which ligand atoms are donors and acceptors.
        ### Remove self-edges from the ligand-ligand graph.
        lig_lig_eidx = full_lig_lig_eidx[:, full_lig_lig_eidx[0] != full_lig_lig_eidx[1]]
        ### Remove non-N/O/S atoms from sink indices.
        lig_lig_eidx = lig_lig_eidx[:, hbond_capable_ligand_node_mask[lig_lig_eidx[1]]]

        ### Identify ligand donors:
        ##### Get just the edges that are connected to hydrogens (and therefore, our donors are in the sink indices).
        donor_sink_edges = lig_lig_eidx[:, ligand_atomic_number_indices[lig_lig_eidx[0]] == 0]
        donor_sink_edges = donor_sink_edges[:, torch.cdist(ligand_coords[donor_sink_edges[0]].unsqueeze(1), ligand_coords[donor_sink_edges[1]].unsqueeze(1)).flatten() < COVALENT_HYDROGEN_BOND_MAX_DISTANCE]

        ### Identify ligand acceptors
        # Add 1 to sp2 atoms and 2 to sp atoms and if the sum of these offsets plus num connected atoms is less than 4, it's an acceptor.
        hybridization_offsets = torch.zeros(ligand_coords.shape[0], dtype=torch.long, device=ligand_coords.device)
        if ligand_atom_hybridization_idces is not None:
            sp_mask = (ligand_atom_hybridization_idces == POSSIBLE_HYBRIDIZATION_LIST.index('SP'))
            sp2_mask = (ligand_atom_hybridization_idces == POSSIBLE_HYBRIDIZATION_LIST.index('SP2'))
            hybridization_offsets[sp2_mask] = 1
            hybridization_offsets[sp_mask] = 2

        ##### Coarsely define acceptor as any hydrogen bond capable atom not forming 4 different bonds.
        sink_acceptor_edges = lig_lig_eidx[:, torch.cdist(ligand_coords[lig_lig_eidx[0]].unsqueeze(1), ligand_coords[lig_lig_eidx[1]].unsqueeze(1)).flatten() < GENERIC_COVALENT_BOND_MAX_DISTANCE]
        ligand_acceptor_indices, acceptor_bond_counts = sink_acceptor_edges[1].unique(return_counts=True)
        acceptor_bond_counts += hybridization_offsets[ligand_acceptor_indices]
        ligand_acceptor_indices = ligand_acceptor_indices[acceptor_bond_counts < 4]

        ### Identify donor and acceptor pairs in ligand-protein edges:
        # ligand_donor_edges = fs_hbonding_lig_pr_edges[:, coarse_hydrogen_bond_mask & torch.isin(fs_hbonding_lig_pr_edges[0], source_hydrogen_edges[1])]
        ligand_donor_mask = torch.isin(fs_hbonding_lig_pr_edges[0][edge_tensor_indices], donor_sink_edges[1])
        protein_acceptor_mask = self.hbond_acceptor_aa_idx_to_acceptor_indices[sink_residue_sequence_indices, sink_residue_atom_indices] # type: ignore

        # Aggregate the ligand nodes that can be a donor/acceptor into a mask
        ligand_actually_hbond_capable_mask = torch.zeros(ligand_coords.shape[0], dtype=torch.bool, device=ligand_coords.device)
        ligand_actually_hbond_capable_mask[donor_sink_edges[1].unique()] = True
        ligand_actually_hbond_capable_mask[ligand_acceptor_indices] = True

        ##################
        # Handle case where the ligand is the donor and the protein is the acceptor.
        pair1_edge_index_mask = ligand_donor_mask & protein_acceptor_mask
        lig_donor_indices = fs_hbonding_lig_pr_edges[0][edge_tensor_indices[pair1_edge_index_mask]]
        pr_acceptor_indices = fs_hbonding_lig_pr_edges[1][edge_tensor_indices[pair1_edge_index_mask]]
        ### Get the donor and acceptor coordinates.
        pair1_donor_coords = ligand_coords[lig_donor_indices]
        pair1_acceptor_coords = fa_coords[pr_acceptor_indices, sink_residue_atom_indices[pair1_edge_index_mask]]
        ### Figure out the hydrogen atoms attached to the donor atoms.
        source_edges = donor_sink_edges[0][torch.isin(donor_sink_edges[1], lig_donor_indices)]
        preremap_sink = (donor_sink_edges[1][torch.isin(donor_sink_edges[1], lig_donor_indices)])
        pair1_angle_values = None
        if lig_donor_indices.shape[0] != 0:
            ##### Create a remapping tensor that maps indices in the donor_sink_edges to the ligand_donor_indices.
            pair1_remap = torch.full((lig_donor_indices.max() + 1,), -1, dtype=torch.long, device=lig_donor_indices.device)
            pair1_remap[lig_donor_indices] = torch.arange(lig_donor_indices.shape[0], dtype=torch.long, device=lig_donor_indices.device)
            remapped_sink = pair1_remap[preremap_sink]
            ### Expand the donor and acceptor coordinates to the number of hydrogen atoms attached to the donor atoms.
            pair1_donor_hydrogens = ligand_coords[source_edges]
            pair1_donor_coords_exp = pair1_donor_coords[remapped_sink]
            pair1_acceptor_coords_exp = pair1_acceptor_coords[remapped_sink]
            ### Compute the angles between the donor hydrogen atoms and the donor-acceptor pairs.
            pair1_angles = compute_angles(pair1_donor_coords_exp, pair1_donor_hydrogens, pair1_acceptor_coords_exp)
            pair1_angle_values = ((pair1_angles - MIN_HBOND_ANGLE) / (180.0 - MIN_HBOND_ANGLE)).clamp(min=0.0)
            pair1_angle_valid_mask = scatter(pair1_angles, remapped_sink, reduce='max', dim_size=int(pair1_edge_index_mask.sum())) > MIN_HBOND_ANGLE
            pair1_angle_values = scatter(pair1_angle_values, remapped_sink, reduce='sum', dim_size=int(pair1_edge_index_mask.sum()))
            pair1_edge_index_mask[pair1_edge_index_mask.clone()] = pair1_angle_valid_mask
        else:
            pair1_edge_index_mask[:] = False

        ##################
        # Handle case where ligand is acceptor and protein is donor.
        pr_donor_hydrogen_indices = self.hbond_donor_aa_idx_to_donor_heavy_pair[sink_residue_sequence_indices, sink_residue_atom_indices] # type: ignore
        pair2_edge_index_mask = torch.isin(fs_hbonding_lig_pr_edges[:, edge_tensor_indices][0], ligand_acceptor_indices) & (pr_donor_hydrogen_indices != MAX_NUM_PROTONATED_RESIDUE_ATOMS).any(dim=-1)
        pair2_donor_coords = fa_coords[fs_hbonding_lig_pr_edges[1][edge_tensor_indices[pair2_edge_index_mask]]].gather(1, sink_residue_atom_indices[pair2_edge_index_mask].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3))
        pair2_donorh_coords = fa_coords[fs_hbonding_lig_pr_edges[1][edge_tensor_indices[pair2_edge_index_mask]]].gather(1, pr_donor_hydrogen_indices[pair2_edge_index_mask].unsqueeze(-1).expand(-1, -1, 3))
        pair2_acceptor_coords = ligand_coords[fs_hbonding_lig_pr_edges[0][edge_tensor_indices[pair2_edge_index_mask]]].unsqueeze(1)
        pair2_angles = compute_angles(pair2_donor_coords, pair2_donorh_coords, pair2_acceptor_coords)
        pair2_angle_values = ((pair2_angles - MIN_HBOND_ANGLE) / (180.0 - MIN_HBOND_ANGLE)).clamp(min=0.0)
        pair2_angle_valid_mask = (pair2_angles > MIN_HBOND_ANGLE).any(dim=-1)
        pair2_edge_index_mask[pair2_edge_index_mask.clone()] = pair2_angle_valid_mask
        ##################

        # Merge the two edge masks and prepare for output.
        final_mask = pair1_edge_index_mask | pair2_edge_index_mask
        values_tensor = torch.zeros_like(final_mask, dtype=torch.float)
        if pair1_angle_values is not None:
            values_tensor[pair1_edge_index_mask] = values_tensor[pair1_edge_index_mask] + pair1_angle_values[pair1_angle_valid_mask].nan_to_num().sum(dim=-1)
        values_tensor[pair2_edge_index_mask] = values_tensor[pair2_edge_index_mask] + pair2_angle_values[pair2_angle_valid_mask].nan_to_num().sum(dim=-1)

        if return_actually_hbonding:
            if return_values:
                return edge_tensor_indices[final_mask], sink_residue_atom_indices[final_mask], ligand_actually_hbond_capable_mask, values_tensor[final_mask]
            return edge_tensor_indices[final_mask], sink_residue_atom_indices[final_mask], ligand_actually_hbond_capable_mask
        if return_values:
            return edge_tensor_indices[final_mask], sink_residue_atom_indices[final_mask], values_tensor[final_mask]
        return edge_tensor_indices[final_mask], sink_residue_atom_indices[final_mask]

    def compute_protein_sidechain_hbonding_graph(self, fa_coords, bb_nh_coords, sequence_indices, full_pr_pr_eidx, ligand_featurizer, return_unique=True, sidechain_only=False, source_sidechain_only=False):
        """
        Rather than only using polar atom distances, exactly computes hydrogen bond network using bond distances and angles involving donor and acceptor atoms.
        """
        assert not (sidechain_only and source_sidechain_only), 'Cannot have both sidechain_only and source_sidechain_only set to True.'

        # Remove self-edges from the protein-protein graph.
        nonself_edge_mask = full_pr_pr_eidx[0] != full_pr_pr_eidx[1]
        pr_pr_eidx = full_pr_pr_eidx[:, nonself_edge_mask]

        # Compute distance between KNN-graph connected residues not adjacent in protein sequence (E, 15, 15).
        source_coords = fa_coords[pr_pr_eidx[0]]
        sink_coords = fa_coords[pr_pr_eidx[1]]
        distance_matrix = torch.cdist(source_coords, sink_coords)

        # Get sequence indices of source and sink residues.
        edge_sequence_indices = sequence_indices[pr_pr_eidx]

        # Start with least restrictive hydrogen bond type (S->S) and use distances that pass this filter to construct pairs of atomic numbers.
        edge_tensor_indices, source_residue_atom_indices, sink_residue_atom_indices = (distance_matrix < S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF).nonzero(as_tuple=True)

        # Filter the indices to only include the hydrogen bond contacts.
        edge_tensor_indices, source_residue_atom_indices, sink_residue_atom_indices = self._compute_protein_protein_hbond_contacts( # type: ignore
            ligand_featurizer, edge_sequence_indices, edge_tensor_indices, source_residue_atom_indices, 
            sink_residue_atom_indices, distance_matrix, fa_coords, pr_pr_eidx, bb_nh_coords
        )

        # Remove the backbone-only hydrogen bonds depending on input flags with new mask.
        #     source_sidechain_only: requires that source edge residue is a sidechain atom. Allows .unique on the sink edges to get only sidechain->backbone and sidechain->sidechain hbonds.
        sidechain_involved_hydrogen_bonds = (source_residue_atom_indices > self.sidechain_start_index)
        if not sidechain_only and not source_sidechain_only:
            # Require that at least one of the interacting residues is a sidechain atom.
            sidechain_involved_hydrogen_bonds |= (sink_residue_atom_indices > self.sidechain_start_index)
        elif sidechain_only:
            # Require that both interacting residues are sidechain atoms.
            sidechain_involved_hydrogen_bonds &= (sink_residue_atom_indices > self.sidechain_start_index)

        edge_tensor_indices = edge_tensor_indices[sidechain_involved_hydrogen_bonds]
        hbond_network_eidx = pr_pr_eidx[:, edge_tensor_indices]

        if not return_unique:
            return hbond_network_eidx
        return hbond_network_eidx.unique(dim=1)

    def compute_first_shell_hbonding_ligand_mask(self, first_shell_indices, fa_coords, sequence_indices, ligand_coords, ligand_atomic_number_indices, lig_pr_eidx, full_lig_lig_eidx, ligand_featurizer, sidechain_only=True, return_values=False):
        """
        """
        # TODO: handle metals & ions.
        # Get the edges that connect hydrogen bond capable ligand atoms to first shell residues.
        hbond_capable_ligand_node_mask = torch.isin(ligand_atomic_number_indices, self.hbonding_atom_indices) # type: ignore
        fs_hbonding_lig_pr_edges = lig_pr_eidx[:, hbond_capable_ligand_node_mask[lig_pr_eidx[0]] & torch.isin(lig_pr_eidx[1], first_shell_indices)]

        # Compute E x 15 distance matrix between ligand and first shell residues.
        distance_matrix = torch.cdist(ligand_coords[fs_hbonding_lig_pr_edges[0]].unsqueeze(1), fa_coords[fs_hbonding_lig_pr_edges[1]]).squeeze(1)

        # Expand the sequence indices and ligand atomic numbers into the edge dimension
        sink_residue_sequence_indices = sequence_indices[fs_hbonding_lig_pr_edges[1]]
        ligand_atom_atomic_numbers_exp = ligand_atomic_number_indices[fs_hbonding_lig_pr_edges[0]]

        # Get sequence indices of residues and edge indices that might be involved in hydrogen bonding.
        edge_tensor_indices, sink_residue_atom_indices = (distance_matrix < S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF).nonzero(as_tuple=True)

        # Filter the contacts that aren't hydrogen bonding with valid donor/acceptor pairs.
        if not return_values:
            edge_tensor_indices, sink_residue_atom_indices = self._compute_protein_ligand_hbond_contacts( # type: ignore
                ligand_atom_atomic_numbers_exp, sink_residue_sequence_indices, 
                distance_matrix, edge_tensor_indices, sink_residue_atom_indices, ligand_featurizer,
                full_lig_lig_eidx, hbond_capable_ligand_node_mask, ligand_atomic_number_indices, ligand_coords,
                fs_hbonding_lig_pr_edges, fa_coords
            ) 

            # If we only want sidechain hydrogen bonds, filter out the backbone hydrogen bonds.
            if sidechain_only:
                edge_tensor_indices = edge_tensor_indices[sink_residue_atom_indices > self.sidechain_start_index]
            
            # Return mask of first shell residues that are involved in hydrogen bonding.
            return torch.isin(first_shell_indices, fs_hbonding_lig_pr_edges[1, edge_tensor_indices])
        else:
            edge_tensor_indices, sink_residue_atom_indices, hbonding_values = self._compute_protein_ligand_hbond_contacts( # type: ignore
                ligand_atom_atomic_numbers_exp, sink_residue_sequence_indices, 
                distance_matrix, edge_tensor_indices, sink_residue_atom_indices, ligand_featurizer,
                full_lig_lig_eidx, hbond_capable_ligand_node_mask, ligand_atomic_number_indices, ligand_coords,
                fs_hbonding_lig_pr_edges, fa_coords, return_values=True
            ) 

            # If we only want sidechain hydrogen bonds, filter out the backbone hydrogen bonds.
            if sidechain_only:
                edge_tensor_indices = edge_tensor_indices[sink_residue_atom_indices > self.sidechain_start_index]
                hbonding_values = hbonding_values[sink_residue_atom_indices > self.sidechain_start_index]

            # Return mask of first shell residues that are involved in hydrogen bonding.
            hbonding_values = scatter(hbonding_values, fs_hbonding_lig_pr_edges[1, edge_tensor_indices], reduce='sum', dim_size=fa_coords.shape[0])
            return torch.isin(first_shell_indices, fs_hbonding_lig_pr_edges[1, edge_tensor_indices]), hbonding_values


    def _compute_hbonding_counts(self, full_atom_coords: torch.Tensor, bb_nh_coords: torch.Tensor, sequence_indices: torch.Tensor, non_adj_prot_eidces: torch.Tensor, ligand_featurizer: LigandFeaturizer):
        """
        Computes the number of hydrogen bonds to each residue sidechain or backbone in the protein.
        """
        edge_index = self.compute_protein_sidechain_hbonding_graph(full_atom_coords, bb_nh_coords, sequence_indices, non_adj_prot_eidces, ligand_featurizer, return_unique=False, source_sidechain_only=True)

        # Return possible hydrogen bond counts:
        output = torch.zeros(full_atom_coords.shape[0], dtype=torch.long, device=full_atom_coords.device)
        indices, counts = edge_index[1, :].unique(return_counts=True)
        output[indices] = counts
        return output
