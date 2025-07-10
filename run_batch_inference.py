#!/usr/bin/env python3
"""
Batch inference script for LASErMPNN model.
Once model is trained, this script can be used to run inference on a directory of PDB files.

Benjamin Fry (bfry@g.harvard.edu)
"""
import os
import re
import argparse
from typing import *
from pathlib import Path

import torch
import numpy as np
import prody as pr
from tqdm import tqdm

from LASErMPNN.utils.model import Sampled_Output
from LASErMPNN.utils.pdb_dataset import BatchData
from LASErMPNN.run_inference import get_protein_hierview, load_model_from_parameter_dict, sample_model, output_protein_structure, output_ligand_structure, ProteinComplexData

CURR_FILE_DIR_PATH = Path(__file__).parent


def parse_int_chunks_or_string(s):
    """
    Parses the input string and returns a tuple of integers if any integer chunks are found; 
    otherwise, returns the original string.
    Args:
        s (str): The input string to parse.
    Returns:
        tuple[int] or str: A tuple of integers extracted from the string if any are found,
        otherwise the original string.
    """
    try:
        ints = re.findall(r'\d+', s)
        if ints:
            return tuple(map(int, ints))
        else:
            return s
    except:
        return s # Just in case.


def collate_batch_data(input_list: List[BatchData]) -> BatchData:

    index_offset = 0
    ligand_info_output = None
    for batch in input_list:
        new_index_offset = batch.batch_indices.max() + 1
        batch.batch_indices += index_offset
        if batch.unprocessed_ligand_input_data is not None:
            batch.unprocessed_ligand_input_data.lig_batch_indices += index_offset
            if ligand_info_output is None:
                ligand_info_output = batch.unprocessed_ligand_input_data
            else:
                ligand_info_output = ligand_info_output.extend(batch.unprocessed_ligand_input_data)
        index_offset += new_index_offset

    batch_data_dict = {
        'pdb_codes': [pdb_code for batch in input_list for pdb_code in batch.pdb_codes],
        'sequence_indices': torch.cat([batch.sequence_indices for batch in input_list]),
        'chi_angles': torch.cat([batch.chi_angles for batch in input_list]),
        'backbone_coords': torch.cat([batch.backbone_coords for batch in input_list]),
        'phi_psi_angles': torch.cat([batch.phi_psi_angles for batch in input_list]),
        'sidechain_contact_number': torch.cat([batch.sidechain_contact_number for batch in input_list]),
        'residue_burial_counts': torch.cat([batch.residue_burial_counts for batch in input_list]), 
        'batch_indices': torch.cat([batch.batch_indices for batch in input_list]),
        'chain_indices': torch.cat([batch.chain_indices for batch in input_list]),
        'sampled_chain_mask': torch.cat([batch.sampled_chain_mask for batch in input_list]),
        'resnum_indices': torch.cat([batch.resnum_indices for batch in input_list]),
        'chain_mask': torch.cat([batch.chain_mask for batch in input_list]),
        'extra_atom_contact_mask': torch.cat([batch.extra_atom_contact_mask for batch in input_list]),
        'msa_data': torch.cat([batch.msa_data for batch in input_list]),
        'msa_depth_weight': torch.cat([batch.msa_depth_weight for batch in input_list]),
        'unprocessed_ligand_input_data': ligand_info_output,
        'first_shell_ligand_contact_mask': torch.cat([batch.first_shell_ligand_contact_mask for batch in input_list]),
        'sc_mediated_hbond_counts': torch.cat([batch.sc_mediated_hbond_counts for batch in input_list]),
    }
    return BatchData(**batch_data_dict)


def _run_inference(
    model, params: dict, input_file: Union[os.PathLike, pr.HierView, Sequence[os.PathLike]], designs_per_input: int, 
    sequence_temp: Optional[float] = None, chi_temp: Optional[float] = None, 
    chi_min_p: float = 0.0, seq_min_p: float = 0.0, use_water: bool = False, disable_pbar: bool = False,
    ignore_chain_mask_zeros: bool = False, disabled_residues_list: List[str] = ['X'], bb_noise: float = 0.0,
    fix_beta: bool = False, repack_only_input_sequence: bool = False, 
    first_shell_sequence_temp: Optional[float] = None, ignore_ligand: bool = False, 
    budget_residue_sele_string: str='', ala_budget: Optional[int]=None, gly_budget: Optional[int]=None,
    noncanonical_aa_ligand: bool = False, fs_calc_ca_distance: float = 10.0, 
    fs_calc_burial_hull_alpha_value: float = 9.0, fs_no_calc_burial: bool = False,
    disable_charged_fs: bool = False
) -> Tuple[Sampled_Output, torch.Tensor, torch.Tensor, torch.Tensor, BatchData, ProteinComplexData]:
    model.eval()

    # Load the model and run inference.
    protein_hvs = []
    if isinstance(input_file, os.PathLike): 
        protein_hvs.append(get_protein_hierview(str(input_file)))
    elif isinstance(input_file, pr.HierView):
        protein_hvs.append(input_file)
    elif isinstance(input_file, Iterable) and isinstance(input_file[0], os.PathLike):
        for path in input_file:
            protein_hvs.append(get_protein_hierview(str(path)))
    else:
        raise ValueError(f"Unrecognized input_file_path type: {type(input_file)}\n{input_file}")

    data_list = []
    batch_data_list = []
    budget_residue_masks = []
    for protein_hv in protein_hvs:
        data = ProteinComplexData(
            protein_hv, protein_hv.getAtoms().getTitle(), use_input_water=use_water,
            verbose=not disable_pbar, treat_noncanonical_as_ligand=noncanonical_aa_ligand,
            first_shell_ca_distance=fs_calc_ca_distance,
            first_shell_buried_only=(not fs_no_calc_burial),
            first_shell_burial_calc_hull_alpha=fs_calc_burial_hull_alpha_value
        )

        budget_residue_mask = None
        if budget_residue_sele_string != '' and budget_residue_sele_string is not None:
            reference_mask_res_indices = protein_hv.getAtoms().select(f"protein and name CA").getResindices()
            mask_sele_indices = protein_hv.getAtoms().select(f"(same residue as ({budget_residue_sele_string})) and name CA").getResindices()
            budget_residue_mask = torch.from_numpy(
                np.isin(reference_mask_res_indices, mask_sele_indices)
            ).to(model.device).unsqueeze(0).expand(designs_per_input, -1).flatten()
            budget_residue_masks.append(budget_residue_mask)

        batch_data = data.output_batch_data(fix_beta=fix_beta, num_copies=designs_per_input)

        if ignore_ligand:
            batch_data.unprocessed_ligand_input_data.lig_coords = torch.empty(0, 3)
            batch_data.unprocessed_ligand_input_data.lig_batch_indices = torch.empty(0, dtype=torch.long)
            batch_data.unprocessed_ligand_input_data.lig_subbatch_indices = torch.empty(0, dtype=torch.long)
            batch_data.unprocessed_ligand_input_data.lig_burial_maskmask = torch.empty(0, dtype=torch.bool)
            batch_data.unprocessed_ligand_input_data.lig_atomic_numbers = torch.empty(0, dtype=torch.long)

        if repack_only_input_sequence:
            batch_data.chain_mask = torch.ones_like(batch_data.chain_mask)

        data_list.append(data)
        batch_data_list.append(batch_data)

    budget_residue_mask = torch.cat(budget_residue_masks) if len(budget_residue_masks) > 0 else None
    batch_data = collate_batch_data(batch_data_list)

    # Sample a design
    sampled_output = sample_model(
        model, batch_data, sequence_temp, bb_noise, params, 
        disable_pbar=disable_pbar, chi_temp=chi_temp, chi_min_p=chi_min_p, 
        seq_min_p=seq_min_p, ignore_chain_mask_zeros=ignore_chain_mask_zeros, 
        disabled_residues=disabled_residues_list, repack_all=repack_only_input_sequence, 
        fs_sequence_temp=first_shell_sequence_temp,
        budget_residue_mask=budget_residue_mask, ala_budget=ala_budget, gly_budget=gly_budget,
        disable_charged_fs=disable_charged_fs
    )
    full_atom_coords = model.rotamer_builder.build_rotamers(batch_data.backbone_coords, sampled_output.sampled_chi_degrees, sampled_output.sampled_sequence_indices, add_nonrotatable_hydrogens=True)
    assert isinstance(full_atom_coords, torch.Tensor), "unreachable."
    nh_coords = model.rotamer_builder.impute_backbone_nh_coords(full_atom_coords.float(), sampled_output.sampled_sequence_indices, batch_data.phi_psi_angles[:, 0].unsqueeze(-1))
    full_atom_coords = model.rotamer_builder.cleanup_titratable_hydrogens(
        full_atom_coords.float(), sampled_output.sampled_sequence_indices, nh_coords, batch_data, model.hbond_network_detector
    )
    sampled_probs = sampled_output.sequence_logits.softmax(dim=-1).gather(1, sampled_output.sampled_sequence_indices.unsqueeze(-1)).squeeze(-1)

    return sampled_output, full_atom_coords, nh_coords, sampled_probs, batch_data, data_list


def run_inference(
        input_pdb_directory, output_pdb_directory, model_weights_path, sequence_temp, chi_temp, 
        inference_device, designs_per_input, designs_per_batch, inputs_processed_simultaneously, use_water, ignore_key_mismatch, 
        verbose=True, seq_min_p=0.0, chi_min_p=0.0, output_idx_offset=0, disabled_residues='', 
        fix_beta=False, repack_only_input_sequence=False, 
        first_shell_sequence_temp=None, ignore_ligand=False, noncanonical_aa_ligand=False,
        budget_residue_sele_string: str='', ala_budget: Optional[int]=None, gly_budget: Optional[int]=None,
        fs_calc_ca_distance: float = 10.0, fs_calc_burial_hull_alpha_value: float = 9.0,
        fs_no_calc_burial: bool = False, disable_charged_fs: bool = False
):
    sequence_temp = float(sequence_temp) if sequence_temp else None
    chi_temp = float(chi_temp) if chi_temp else None
    disabled_residues_list = disabled_residues.split(',')

    # Load the model
    model, params = load_model_from_parameter_dict(model_weights_path, inference_device, strict=ignore_key_mismatch)
    model.eval()

    input_pdb_directory = Path(input_pdb_directory)
    output_pdb_directory = Path(output_pdb_directory)

    make_subdir = False
    if input_pdb_directory.is_dir():
        all_input_files = [input_pdb_directory / x for x in input_pdb_directory.glob('*.pdb')]
        make_subdir = True
    elif input_pdb_directory.exists() and ('.pdb' in input_pdb_directory.name): # Could be .pdb or .pdb.gz
        all_input_files = [input_pdb_directory]
    elif input_pdb_directory.exists() and input_pdb_directory.suffix == '.txt':
        all_input_files = [Path(x.strip()) for x in open(input_pdb_directory, 'r').readlines() if Path(x.strip()).exists()]
        make_subdir = True
    else:
        print(f'Could not find {input_pdb_directory}')
        raise FileNotFoundError

    # Loop over all files to design.
    all_pdb_files_for_processing = sorted(all_input_files, key=lambda x: parse_int_chunks_or_string(x.stem))
    if verbose:
        print(f"Processing {input_pdb_directory}:")
        print(f"Generating {designs_per_input} designs for {len(all_pdb_files_for_processing)} inputs with {model_weights_path} on {inference_device} at temperature {sequence_temp}")

    input_chunks = np.array_split(all_pdb_files_for_processing, (len(all_pdb_files_for_processing) // inputs_processed_simultaneously) + 1)
    for files_chunk in tqdm(input_chunks):
        # Make an output subdirectory for each input file.
        output_files_chunk = []
        if make_subdir:
            for file in files_chunk:
                output_subdir_path = output_pdb_directory / file.stem
                output_subdir_path.mkdir(exist_ok=True, parents=True)
                output_files_chunk.append(output_subdir_path)
        else:
            output_files_chunk = [output_pdb_directory]
            output_pdb_directory.mkdir(exist_ok=True)

        designs_remaining = designs_per_input
        curr_output_idx_offset = output_idx_offset
        while designs_remaining > 0:
            curr_num_to_design = min(designs_per_batch, designs_remaining)

            sampled_output, full_atom_coords, nh_coords, sampled_probs, batch_data, data_list = _run_inference(
                model, params, list(files_chunk), curr_num_to_design, 
                use_water=use_water, sequence_temp=sequence_temp, chi_temp=chi_temp, chi_min_p=chi_min_p, seq_min_p=seq_min_p, 
                disabled_residues_list=disabled_residues_list, disable_pbar=not verbose,
                fix_beta=fix_beta, repack_only_input_sequence=repack_only_input_sequence,
                first_shell_sequence_temp=first_shell_sequence_temp, ignore_ligand=ignore_ligand,
                budget_residue_sele_string=budget_residue_sele_string, 
                ala_budget=ala_budget, gly_budget=gly_budget,
                noncanonical_aa_ligand=noncanonical_aa_ligand,
                fs_calc_ca_distance=fs_calc_ca_distance, 
                fs_calc_burial_hull_alpha_value=fs_calc_burial_hull_alpha_value,
                fs_no_calc_burial=fs_no_calc_burial, disable_charged_fs=disable_charged_fs
            )

            idx_offset = 0
            for jdx, data in enumerate(data_list):
                for idx in range(curr_num_to_design):
                    # Output the current batch design + ligand and write to disk
                    curr_batch_mask = batch_data.batch_indices == (idx + idx_offset)
                    out_prot = output_protein_structure(full_atom_coords[curr_batch_mask], sampled_output.sampled_sequence_indices[curr_batch_mask], data.residue_identifiers, nh_coords[curr_batch_mask], sampled_probs[curr_batch_mask])

                    out_complex = out_prot
                    try:
                        out_lig = output_ligand_structure(data.ligand_info)
                        out_complex += out_lig
                    except:
                        pass
                    pr.writePDB(str(output_files_chunk[jdx] / f"design_{idx+curr_output_idx_offset}.pdb"), out_complex)
                idx_offset += curr_num_to_design
            
            curr_output_idx_offset += curr_num_to_design
            designs_remaining -= curr_num_to_design


def parse_args(default_weights_path: os.PathLike):
    parser = argparse.ArgumentParser(description='Run batch LASErMPNN inference.')
    parser.add_argument('input_pdb_directory', type=str, help='Path to directory of input .pdb or .pdb.gz files, a single input .pdb or .pdb.gz file, or a .txt file of paths to input .pdb or .pdb.gz files.')
    parser.add_argument('output_pdb_directory', type=str, help='Path to directory to output LASErMPNN designs.')
    parser.add_argument('designs_per_input', type=int, help='Number of designs to generate per input.')
    parser.add_argument('--designs_per_batch', '-b', type=int, default=30, help='Number of designs to generate per batch. If designs_per_input > designs_per_batch, chunks up the inference calls in batches of this size. Default is 30, can increase/decrease depending on available GPU memory.')
    parser.add_argument('--inputs_processed_simultaneously', type=int, default=5, help='When passed a list of multiple files, this is the number of input files to process per pass through the GPU. Useful when generating a few sequences for many input files.')
    parser.add_argument('--model_weights_path', '-w', type=str, default=f'{default_weights_path}', help=f'Path to model weights. Default: {default_weights_path}')

    parser.add_argument('--sequence_temp', type=float, default=None, help='Temperature for sequence sampling.')
    parser.add_argument('--first_shell_sequence_temp', type=float, default=None, help='Temperature for first shell sequence sampling. Can be used to disentangle binding site temperature from global sequence temperature for harder folds.')
    parser.add_argument('--chi_temp', type=float, default=None, help='Temperature for chi sampling.')
    parser.add_argument('--chi_min_p', type=float, default=0.0, help='Minimum probability for chi sampling. Not recommended.')
    parser.add_argument('--seq_min_p', type=float, default=0.0, help='Minimum probability for sequence sampling. Not recommended.')

    parser.add_argument('--device', '-d', dest='inference_device', default='cpu', type=str, help='PyTorch style device string (e.g. "cuda:0").')
    parser.add_argument('--use_water', action='store_true', help='Parses water (resname HOH) as part of a ligand.')
    parser.add_argument('--silent', dest='verbose', action='store_false', help='Silences all output except pbar.')
    parser.add_argument('--ignore_key_mismatch', action='store_false', help='Allows mismatched keys in checkpoint statedict')
    parser.add_argument('--disabled_residues', type=str, default='X', help='Residues to disable in sampling.')
    parser.add_argument('--fix_beta', action='store_true', help='If B-factors are set to 1, fixes the residue and rotamer, if not, designs that position.')
    parser.add_argument('--repack_only_input_sequence', action='store_true', help='Repacks the input sequence without changing the sequence.')
    parser.add_argument('--ignore_ligand', action='store_true', help='Ignore ligand in sampling.')
    parser.add_argument('--budget_residue_sele_string', default=None, help='')
    parser.add_argument('--ala_budget', type=int, default=4, help='')
    parser.add_argument('--gly_budget', type=int, default=0, help='')
    parser.add_argument('--noncanonical_aa_ligand', action='store_true', help='Featurize a noncanonical amino acid as a ligand.')

    parser.add_argument('--fs_calc_ca_distance', type=float, default=10.0, help='Distance between a ligand heavy atom and CA carbon to consider that carbon first shell.')
    parser.add_argument('--fs_calc_burial_hull_alpha_value', type=float, default=9.0, help='Alpha parameter for defining convex hull. May want to try setting to larger values if using folds with larger cavities (ex: ~100.0).')
    parser.add_argument('--fs_no_calc_burial', action='store_true', help='Disable using a burial calculation when selecting first shell residues, if true uses only distance from --fs_calc_ca_distance')
    parser.add_argument('--disable_charged_fs', action='store_true', help='Disable sampling D,K,R,E residues in the first shell around the ligand.')

    parsed_args = parser.parse_args()

    return vars(parsed_args)


if __name__ == "__main__":
    default_weights_path = CURR_FILE_DIR_PATH / 'model_weights/laser_weights_0p1A_noise_ligandmpnn_split.pt'
    run_inference(**parse_args(default_weights_path))