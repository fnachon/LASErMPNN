#!/usr/bin/env python3
"""
Batch inference script for LASErMPNN model.
Once model is trained, this script can be used to run inference on a directory of PDB files.

Benjamin Fry (bfry@g.harvard.edu)
"""
import os
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


def _run_inference(
    model, params: dict, input_file_path: Union[str, pr.HierView], designs_per_input: int, 
    sequence_temp: Optional[float] = None, chi_temp: Optional[float] = None, 
    chi_min_p: float = 0.0, seq_min_p: float = 0.0, use_water: bool = False, disable_pbar: bool = False,
    ignore_chain_mask_zeros: bool = False, disabled_residues_list: List[str] = ['X'], bb_noise: float = 0.0,
    fix_beta: bool = False, repack_only_input_sequence: bool = False, 
    first_shell_sequence_temp: Optional[float] = None, ignore_ligand: bool = False, 
    budget_residue_sele_string: str='', ala_budget: Optional[int]=None, gly_budget: Optional[int]=None,
    noncanonical_aa_ligand: bool = False,
) -> Tuple[Sampled_Output, torch.Tensor, torch.Tensor, torch.Tensor, BatchData, ProteinComplexData]:
    model.eval()

    # Load the model and run inference.
    if isinstance(input_file_path, str): 
        protein_hv = get_protein_hierview(input_file_path)
        data = ProteinComplexData(protein_hv, input_file_path, use_input_water=use_water, verbose=not disable_pbar, treat_noncanonical_as_ligand=noncanonical_aa_ligand)
    else:
        protein_hv = input_file_path
        data = ProteinComplexData(protein_hv, 'input', use_input_water=use_water, verbose=not disable_pbar, treat_noncanonical_as_ligand=noncanonical_aa_ligand)

    budget_residue_mask = None
    if budget_residue_sele_string != '' and budget_residue_sele_string is not None:
        reference_mask_res_indices = protein_hv.getAtoms().select(f"protein and name CA").getResindices()
        mask_sele_indices = protein_hv.getAtoms().select(f"(same residue as ({budget_residue_sele_string})) and name CA").getResindices()
        budget_residue_mask = torch.from_numpy(
            np.isin(reference_mask_res_indices, mask_sele_indices)
        ).to(model.device).unsqueeze(0).expand(designs_per_input, -1).flatten()

    batch_data = data.output_batch_data(fix_beta=fix_beta, num_copies=designs_per_input)

    if ignore_ligand:
        batch_data.unprocessed_ligand_input_data.lig_coords = torch.empty(0, 3)
        batch_data.unprocessed_ligand_input_data.lig_batch_indices = torch.empty(0, dtype=torch.long)
        batch_data.unprocessed_ligand_input_data.lig_subbatch_indices = torch.empty(0, dtype=torch.long)
        batch_data.unprocessed_ligand_input_data.lig_burial_maskmask = torch.empty(0, dtype=torch.bool)
        batch_data.unprocessed_ligand_input_data.lig_atomic_numbers = torch.empty(0, dtype=torch.long)

    if repack_only_input_sequence:
        batch_data.chain_mask = torch.ones_like(batch_data.chain_mask)

    # Sample a design
    sampled_output = sample_model(
        model, batch_data, sequence_temp, bb_noise, params, 
        disable_pbar=disable_pbar, chi_temp=chi_temp, chi_min_p=chi_min_p, 
        seq_min_p=seq_min_p, ignore_chain_mask_zeros=ignore_chain_mask_zeros, 
        disabled_residues=disabled_residues_list, repack_all=repack_only_input_sequence, 
        fs_sequence_temp=first_shell_sequence_temp,
        budget_residue_mask=budget_residue_mask, ala_budget=ala_budget, gly_budget=gly_budget
    )
    full_atom_coords = model.rotamer_builder.build_rotamers(batch_data.backbone_coords, sampled_output.sampled_chi_degrees, sampled_output.sampled_sequence_indices, add_nonrotatable_hydrogens=True)
    assert isinstance(full_atom_coords, torch.Tensor), "unreachable."
    nh_coords = model.rotamer_builder.impute_backbone_nh_coords(full_atom_coords.float(), sampled_output.sampled_sequence_indices, batch_data.phi_psi_angles[:, 0].unsqueeze(-1))
    full_atom_coords = model.rotamer_builder.cleanup_titratable_hydrogens(
        full_atom_coords.float(), sampled_output.sampled_sequence_indices, nh_coords, batch_data, model.hbond_network_detector
    )
    # assert isinstance(nh_coords, torch.Tensor), "unreachable."
    sampled_probs = sampled_output.sequence_logits.softmax(dim=-1).gather(1, sampled_output.sampled_sequence_indices.unsqueeze(-1)).squeeze(-1)
    return sampled_output, full_atom_coords, nh_coords, sampled_probs, batch_data, data


def run_inference(
        input_pdb_directory, output_pdb_directory, model_weights_path, sequence_temp, chi_temp, 
        inference_device, designs_per_input, designs_per_batch, use_water, ignore_key_mismatch, 
        verbose=True, seq_min_p=0.0, chi_min_p=0.0, output_idx_offset=0, disabled_residues='', 
        fix_beta=False, repack_only_input_sequence=False, 
        first_shell_sequence_temp=None, ignore_ligand=False, noncanonical_aa_ligand=False
        budget_residue_sele_string: str='', ala_budget: Optional[int]=None, gly_budget: Optional[int]=None
):
    sequence_temp = float(sequence_temp) if sequence_temp else None
    chi_temp = float(chi_temp) if chi_temp else None
    disabled_residues_list = disabled_residues.split(',')

    # Load the model
    model, params = load_model_from_parameter_dict(model_weights_path, inference_device, strict=ignore_key_mismatch)
    model.eval()

    # Loop over all files to design.
    if verbose:
        print(f"Processing {input_pdb_directory}:")
        print(f"Generating {designs_per_input} designs with {model_weights_path} on {inference_device} at temperature {sequence_temp}")

    make_subdir = False
    if os.path.isdir(input_pdb_directory):
        all_input_files = [os.path.join(input_pdb_directory, x) for x in os.listdir(input_pdb_directory)]
        make_subdir = True
    elif os.path.exists(input_pdb_directory) and '.pdb' in input_pdb_directory:
        all_input_files = [input_pdb_directory]
    elif os.path.exists(input_pdb_directory) and input_pdb_directory.endswith('.txt'):
        all_input_files = [x.strip() for x in open(input_pdb_directory, 'r').readlines() if os.path.exists(x.strip())]
        make_subdir = True
    else:
        print(f'Could not find {input_pdb_directory}')
        raise NotImplementedError

    for file in tqdm([x for x in sorted(all_input_files) if '.pdb' in x]):
        # Make an output subdirectory for each input file.
        output_subdir_path = output_pdb_directory
        if not os.path.exists(output_pdb_directory):
            os.mkdir(output_pdb_directory)
        if make_subdir:
            output_subdir_path = os.path.join(output_pdb_directory, file.rsplit('/', 1)[-1].split('.')[0])
            if not os.path.exists(output_subdir_path):
                os.makedirs(output_subdir_path, exist_ok=True)

        designs_remaining = designs_per_input
        curr_output_idx_offset = output_idx_offset
        while designs_remaining > 0:
            curr_num_to_design = min(designs_per_batch, designs_remaining)

            sampled_output, full_atom_coords, nh_coords, sampled_probs, batch_data, data = _run_inference(
                model, params, file, curr_num_to_design, 
                use_water=use_water, sequence_temp=sequence_temp, chi_temp=chi_temp, chi_min_p=chi_min_p, seq_min_p=seq_min_p, 
                disabled_residues_list=disabled_residues_list, disable_pbar=not verbose,
                fix_beta=fix_beta, repack_only_input_sequence=repack_only_input_sequence,
                first_shell_sequence_temp=first_shell_sequence_temp, ignore_ligand=ignore_ligand,
                budget_residue_sele_string=budget_residue_sele_string, 
                ala_budget=ala_budget, gly_budget=gly_budget,
                noncanonical_aa_ligand=noncanonical_aa_ligand
            )
            
            for idx in range(curr_num_to_design):
                # Output the current batch design + ligand and write to disk
                curr_batch_mask = batch_data.batch_indices == idx
                out_prot = output_protein_structure(full_atom_coords[curr_batch_mask], sampled_output.sampled_sequence_indices[curr_batch_mask], data.residue_identifiers, nh_coords[curr_batch_mask], sampled_probs[curr_batch_mask])

                out_complex = out_prot
                try:
                    out_lig = output_ligand_structure(data.ligand_info)
                    out_complex += out_lig
                except:
                    pass
                pr.writePDB(os.path.join(output_subdir_path, f"design_{idx+curr_output_idx_offset}.pdb"), out_complex)
            
            curr_output_idx_offset += curr_num_to_design
            designs_remaining -= curr_num_to_design


def parse_args(default_weights_path: str):
    parser = argparse.ArgumentParser(description='Run batch LASErMPNN inference.')
    parser.add_argument('input_pdb_directory', type=str, help='Path to directory of input .pdb or .pdb.gz files, a single input .pdb or .pdb.gz file, or a .txt file of paths to input .pdb or .pdb.gz files.')
    parser.add_argument('output_pdb_directory', type=str, help='Path to directory to output LASErMPNN designs.')
    parser.add_argument('designs_per_input', type=int, help='Number of designs to generate per input.')
    parser.add_argument('--designs_per_batch', '-b', type=int, default=30, help='Number of designs to generate per batch. If designs_per_input > designs_per_batch, chunks up the inference calls in batches of this size. Default is 30, can increase/decrease depending on available GPU memory.')
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

    parsed_args = parser.parse_args()

    return vars(parsed_args)


if __name__ == "__main__":
    default_weights_path = CURR_FILE_DIR_PATH / 'model_weights/laser_weights_0p1A_noise_ligandmpnn_split.pt'
    run_inference(**parse_args(default_weights_path))
