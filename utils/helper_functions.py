

import torch
import prody as pr
from collections import defaultdict, namedtuple
from .constants import dataset_atom_order, hydrogen_extended_dataset_atom_order, aa_idx_to_short, aa_idx_to_long, aa_short_to_idx
from typing import *
import logomaker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import Sequence, Optional

from torch_scatter import scatter_mean


def mps_safe_double(x: torch.Tensor) -> torch.Tensor:
    """MPS has no float64 support; stay in float32 there and use float64 elsewhere for precision."""
    return x.float() if x.device.type == 'mps' else x.double()


def mps_safe_knn_graph(x: torch.Tensor, k: int, loop: bool = False, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
    """torch_cluster's knn_graph only ships CPU/CUDA kernels; round-trip through CPU on MPS."""
    from torch_cluster import knn_graph
    if x.device.type != 'mps':
        return knn_graph(x, k=k, loop=loop, batch=batch)
    batch_cpu = batch.cpu() if batch is not None else None
    return knn_graph(x.cpu(), k=k, loop=loop, batch=batch_cpu).to(x.device)


def write_xyz_file(file_path: str, atom_symbols: Sequence, coordinates: Sequence, description: str = ""):
    """
    Writes input data to an xyz file.
    Args:
        file_path (str): Path to the output file.
        atom_symbols (Sequence): List-like of atom symbols.
        coordinates (Sequence): List-like of (x,y,z) coordinates.
        description (str): Description of the xyz file.
    """
    num_atoms = len(atom_symbols)

    assert len(coordinates) == num_atoms, f"Number of coordinates ({len(coordinates)}) does not match number of atoms ({num_atoms})."
    assert num_atoms > 0, f"Number of atoms ({num_atoms}) must be greater than 0."

    with open(file_path, 'w') as f:
        # Header information.
        f.write(f"{num_atoms}\n")
        f.write(f"{description}\n")

        # Write atom information.
        for atom_idx in range(num_atoms):
            atom_name = atom_symbols[atom_idx]
            x,y,z = coordinates[atom_idx]
            f.write(f"{atom_name}\t{x}\t{y}\t{z}\n")
    

def write_cif_file():
    assert False, "Implement mmCIF file writing."


Rescode = namedtuple('Rescode', ['chain', 'resnum', 'resname'])
def create_prody_protein_from_coordinate_matrix(full_protein_coords: torch.Tensor, amino_acid_labels: torch.Tensor, bb_nh_coords: Optional[torch.Tensor] = None, return_rescode_list: bool = False) -> Union[pr.AtomGroup, Tuple[pr.AtomGroup, List[Rescode]]]:
    """
    Enables pdb file writing from a coordinate matrix.
    Inputs:
        full_protein_coords (torch.Tensor): A tensor of shape (num_residues, num_atoms, 3) containing the coordinates of the protein.
        amino_acid_labels (torch.Tensor): A tensor of shape (num_residues,) containing the amino acid index labels for each residue.
    Returns:
        protein (pr.AtomGroup): A prody AtomGroup object containing the protein structure.
    """

    using_bb_hydrogens = False
    assert full_protein_coords.device == torch.device('cpu'), "Coordinate matrix must be on the CPU."
    assert amino_acid_labels.device == torch.device('cpu'), "Amino acid labels must be on the CPU."
    if bb_nh_coords is not None:
        assert bb_nh_coords.device == torch.device('cpu'), "Backbone NH coordinates must be on the CPU."
        assert full_protein_coords.shape[0] == bb_nh_coords.shape[0], "Coordinate matrices must have the same shape."
        using_bb_hydrogens = True
    else:
        bb_nh_coords = torch.full((full_protein_coords.shape[0], 3), torch.nan)

    # Determine if the coordinate matrix has hydrogen atoms
    if full_protein_coords.shape[1] == 24:
        with_hydrogen = True
    elif full_protein_coords.shape[1] == 14:
        with_hydrogen = False
    else:
        assert False, "Coordinate matrix must have 14 or 24 columns."

    atom_order_dict = hydrogen_extended_dataset_atom_order if with_hydrogen else dataset_atom_order

    # Drop NaNs from the coordinate matrix and flatten it
    amino_acid_indices = amino_acid_labels.cpu().tolist()

    all_coords = []
    rescode_list = []
    prody_features = defaultdict(list)
    for resnum, (coord, aa_idx, bb_h) in enumerate(zip(full_protein_coords, amino_acid_indices, bb_nh_coords)):
        missing_h = False
        if bb_h.isnan().any():
            missing_h = True
        coord = torch.cat([coord, bb_h.unsqueeze(0)], dim=0)
        coord_mask = coord.isnan().any(dim=1)
        coord = coord[~coord_mask]

        real_aa_idx = aa_idx
        if aa_idx == aa_short_to_idx['X']:
            aa_idx = aa_short_to_idx['G']

        atom_names = [x for idx, x in enumerate(atom_order_dict[aa_idx_to_short[aa_idx]]) if not coord_mask[idx].item()] 
        atom_names += [] if ((not using_bb_hydrogens) or missing_h) else ['H']

        prody_features['resnames'].extend([aa_idx_to_long[real_aa_idx]] * len(atom_names))
        prody_features['atom_labels'].extend(atom_names)
        prody_features['atom_elements'].extend([x[0] for x in atom_names])
        prody_features['resnums'].extend([resnum + 1] * len(atom_names))
        prody_features['chains'].extend(['A'] * len(atom_names))
        prody_features['occupancies'].extend([1.0] * len(atom_names))
        all_coords.append(coord)
        rescode_list.append(Rescode('A', resnum + 1, aa_idx_to_long[real_aa_idx]))

    flattened_coords = torch.cat(all_coords, dim=0)

    assert all([len(x) == len(flattened_coords) for x in prody_features.values()]), f"Prody features must be the same length as the flattened coordinates. {[len(x) for x in prody_features.values()]} != {len(flattened_coords)}"
    
    protein = pr.AtomGroup('LASErMPNN Generated Protein')
    protein.setCoords(flattened_coords)
    protein.setNames(prody_features['atom_labels']) # type: ignore
    protein.setResnames(prody_features['resnames']) # type: ignore
    protein.setResnums(prody_features['resnums']) # type: ignore
    protein.setChids(prody_features['chains']) # type: ignore
    protein.setOccupancies(prody_features['occupancies'])# type: ignore
    protein.setElements(prody_features['atom_elements']) # type: ignore
    protein.setBetas([0.0] * len(flattened_coords)) # type: ignore

    if return_rescode_list:
        return protein, rescode_list

    return protein


def compute_sidechain_rmsd(coords1: torch.Tensor, coords2: torch.Tensor, sequence_indices: torch.Tensor) -> dict:
    """
    Given NaN-padded coordinate tensors, computes the RMSD of the sidechain atoms.
    """
    output = {}

    # Drop 'X' residues from the sequence indices.
    ignore_mask = sequence_indices != aa_short_to_idx['X']

    # Compute RMSD of sidechain atoms since already aligned.
    num_atoms = (~(coords1[ignore_mask][:, 4:].isnan().any(dim=-1))).long().sum(dim=-1)
    distances = torch.linalg.norm(coords1[ignore_mask][:, 4:] - coords2[ignore_mask][:, 4:], dim=-1).nan_to_num(nan=0.0).sum(dim=-1)
    rmsd = distances * (1 / torch.sqrt(num_atoms).clamp_min_(1.0))

    # Compute the mean RMSD of each amino acid type.
    mean_aggr = scatter_mean(rmsd, sequence_indices[ignore_mask], dim_size=20)

    # Loop over each amino acid type and add the mean RMSD to the output.
    for aa_idx, _rmsd in enumerate(mean_aggr.cpu().numpy()):
        if (sequence_indices == aa_idx).any():
            output[aa_idx_to_short[aa_idx]] = _rmsd 

    return output


def make_seq_logo(seq_ls, ax=None, resnum_labels=None, add_ticks=True, num_rows=1):
    'resnums is a list of resnums used to label the x-axis'
    #plt.figure()

    aa_list_abv = list('ARNDCEQGHILKMFPSTWYV')
    aa_to_idx = {aa:i for i, aa in enumerate(aa_list_abv)}
    L = len(seq_ls[0])

    # calculate probability at every position 
    pos = np.zeros([len(seq_ls[0]), 20])

    for s in seq_ls:
        # s = s.replace('-', ' ')
        for i, aa in enumerate(s): 
            pos[i, aa_to_idx[aa]] += 1

    # normalize each row of the matrix by its sum
    pos_normalized = pos / pos.sum(axis=1)[:, np.newaxis]
    loop_df = pd.DataFrame(pos_normalized, columns=aa_list_abv)
    logomaker_df = logomaker.validate_matrix(loop_df, matrix_type='probability')

    if num_rows > 1:
        fig, axes = plt.subplots(num_rows, 1, figsize=(12, num_rows*2.5))
    else:
        axes = [ax]
    for i in range(num_rows):
        # create Logo object
        sl = slice(i*(L//num_rows),((i+1)*(L//num_rows))) 
        df = logomaker_df.iloc[sl]
        ax = axes[i]
        loop_logo = logomaker.Logo(df,
                                #font_name='Helvetica',
                                color_scheme='weblogo_protein',
                                vpad=.1,
                                width=.8,
                                ax=ax)

        if add_ticks:
            loop_logo.style_xticks(anchor=0, spacing=5, rotation=90)
            if resnum_labels is None:
                labels = np.arange(1,len(logomaker_df)+1)[sl]
            else:
                labels = resnum_labels[sl]
            loop_logo.ax.set_xticks(ticks=np.arange(len(logomaker_df))[sl], labels=labels) # type: ignore
        loop_logo.ax.set_xlim([i*(L//num_rows)-1, ((i+1)*(L//num_rows))]) # type: ignore
        loop_logo.ax.set_yticks([]) # type: ignore
    return loop_logo.ax
