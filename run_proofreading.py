import sys
import torch
import prody as pr
import numpy as np
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import asdict

import warnings
warnings.filterwarnings("ignore")

from utils.model import LASErMPNN
from utils.constants import aa_idx_to_short, aa_long_to_idx
from utils.pdb_dataset import UnclusteredProteinChainDataset, collate_sampler_data, ClusteredDatasetSampler, BatchData
from run_inference import load_model_from_parameter_dict, get_protein_hierview, sample_model, ProteinComplexData, run_inference

CURR_FILE_DIR_PATH = Path(__file__).parent

def load_model(weights, device, use_inference_dropout):
    model, training_parameter_dict = load_model_from_parameter_dict(weights, device)

    model.eval()
    if use_inference_dropout:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    return model, training_parameter_dict


def get_forward_pass_probabilities(model, training_parameter_dict, input_protein_path, unconditional=False, ignore_ligand=False):
    with torch.no_grad():
        batch_data: BatchData

        protein_hv = get_protein_hierview(input_protein_path, ignore_ligand=ignore_ligand)
        data = ProteinComplexData(protein_hv, input_protein_path)
        batch_data = data.output_batch_data(fix_beta=False)
        rescodes = np.array([f'{x["resname"]}-{x["resnum"]}' for x in asdict(data)['residue_identifiers']])

        batch_data.to_device(model.device)
        batch_data.construct_graphs(
            model.rotamer_builder,
            model.ligand_featurizer, 
            **training_parameter_dict['model_params']['graph_structure'],
            protein_training_noise = 0.0,
            ligand_training_noise = 0.0,
            subgraph_only_dropout_rate = 0.0,
            num_adjacent_residues_to_drop = 6,
            build_hydrogens = training_parameter_dict['model_params']['build_hydrogens'],
        )
        batch_data.generate_decoding_order(False)

        fs_mask = batch_data.first_shell_ligand_contact_mask.cpu()

        if unconditional:
            sequence_logits, *_ = model.forward(batch_data, return_unconditional_probabilities=True)
            return sequence_logits, fs_mask, batch_data.sequence_indices, rescodes

        sequence_logits, *_ = model.forward(batch_data)
        return sequence_logits, fs_mask, batch_data.sequence_indices, rescodes


@torch.no_grad()
def compute_unconditional_probs(model, param_dict, pdb_file: Path, output_dir: Path, selection_string: str, return_probs: bool = False, ignore_ligand: bool = False):
    sequence_logits, fs_mask, seq_indices, rescodes = get_forward_pass_probabilities(model, param_dict, str(pdb_file), unconditional=True, ignore_ligand=ignore_ligand)

    if len(selection_string) > 0 :
        A = pr.parsePDB(str(pdb_file))
        sele = A.select(f'(same residue as ({selection_string})) and name CA')
        if sele is not None:
            resindices = sele.getResindices()
            new_mask = torch.zeros_like(fs_mask)
            new_mask[resindices] = True
            print(new_mask)
            fs_mask = new_mask
        else:
            raise ValueError('ProDy string is invalid!')

    ylabels = [x for idx, x in enumerate(rescodes) if fs_mask[idx]]
    fig, axes = plt.subplots(figsize=(5, 5), dpi=300)
    fs_probs = sequence_logits.softmax(dim=-1).cpu()[fs_mask].numpy()
    sns.heatmap(
        fs_probs, 
        xticklabels=[aa_idx_to_short[idx] for idx in range(21)], 
        yticklabels=[x for idx, x in enumerate(rescodes) if fs_mask[idx]],
        cmap='Reds', vmin=0.0, vmax=0.5
    )

    torch.save(fs_probs, output_dir / 'unconditional_probs.pt')

    ground_truth_indices = [aa_long_to_idx[x.split('-')[0]] for x in ylabels]
    for i, gt_idx in enumerate(ground_truth_indices):
        plt.text(gt_idx + 0.5, i + 0.61, 'X', color='red', ha='center', va='center', fontsize=16)
    plt.title(f'Unconditional\n{pdb_file.stem}\nProbabilities')
    plt.tight_layout()
    output_path = output_dir / f'unconditional_probs.png'
    plt.savefig(output_path)

    if return_probs:
        return fs_probs

    return fs_mask, ylabels, output_path


@torch.no_grad()
def compute_conditional_probs(model, training_parameter_dict, pdb_file: Path, output_dir: Path, fs_mask, ylabels, n_decoding_orders: int, n_dropouts: int, repack_all: bool):
    stacked_probs = []
    stacked_probs_mean_plus_stdv = []
    for unfixed_index in tqdm(fs_mask.nonzero().flatten().tolist(), total=len(fs_mask.nonzero().flatten().tolist())):
        subbatch = []
        subbatch_stddv = []
        for _ in range(n_dropouts):
            # Get the protein hierview
            protein_hv = get_protein_hierview(str(pdb_file))

            # Set all betas to 1.0 except for the unfixed residue
            protein_hv.getAtoms().setBetas(1.0)
            protein_hv.getAtoms().select(f'resindex {unfixed_index}').setBetas(0.0)

            data = ProteinComplexData(protein_hv, str(pdb_file), verbose=False)
            batch_data = data.output_batch_data(fix_beta=True, num_copies=n_decoding_orders)

            batch_data.to_device(model.device)
            batch_data.construct_graphs(
                model.rotamer_builder,
                model.ligand_featurizer, 
                **training_parameter_dict['model_params']['graph_structure'],
                protein_training_noise = 0.0,
                ligand_training_noise = 0.0,
                subgraph_only_dropout_rate = 0.0,
                num_adjacent_residues_to_drop = 6,
                build_hydrogens = training_parameter_dict['model_params']['build_hydrogens'],
            )
            batch_data.generate_decoding_order(True)

            sampling_output = model.sample(batch_data, sequence_sample_temperature=1.0, chi_angle_sample_temperature=1.0, disabled_residues=['X'], disable_pbar=True, repack_all=repack_all)

            logits_reshaped = sampling_output.sequence_logits.reshape(n_decoding_orders, -1, 21).softmax(dim=-1).mean(dim=0)
            stddev_reshaped = sampling_output.sequence_logits.reshape(n_decoding_orders, -1, 21).softmax(dim=-1).std(dim=0)
            subbatch.append(logits_reshaped[unfixed_index])
            subbatch_stddv.append(stddev_reshaped[unfixed_index])
        stacked_probs.append(torch.stack(subbatch).mean(dim=0))
        stacked_probs_mean_plus_stdv.append(torch.stack(subbatch).mean(dim=0) + torch.stack(subbatch_stddv).mean(dim=0))

    stacked_probs = torch.stack(stacked_probs)
    stacked_probs_mean_plus_stdv = torch.stack(stacked_probs_mean_plus_stdv)
    stacked_probs_mean_plus_stdv = stacked_probs_mean_plus_stdv.cpu().numpy()

    probs_renormed = stacked_probs.cpu().numpy()
    probs_renormed = probs_renormed / probs_renormed.sum(axis=-1, keepdims=True)
    
    torch.save(stacked_probs, output_dir / 'conditional_probs.pt')
    torch.save(stacked_probs_mean_plus_stdv, output_dir / 'conditional_probs_mean_plus_stdv_no_norm.pt')

    mean_plus_stdv_renormed = stacked_probs_mean_plus_stdv / stacked_probs_mean_plus_stdv.sum(axis=-1, keepdims=True)

    fig, axes = plt.subplots(figsize=(5, 5), dpi=150)
    sns.heatmap(
        probs_renormed,
        xticklabels=[aa_idx_to_short[idx] for idx in range(21)], 
        yticklabels=ylabels,
        cmap='Greens', vmin=0.0, vmax=0.5,
    )

    ground_truth_indices = [aa_long_to_idx[x.split('-')[0]] for x in ylabels]
    for i, gt_idx in enumerate(ground_truth_indices):
        plt.text(gt_idx + 0.5, i + 0.61, 'X', color='red', ha='center', va='center', fontsize=16)

    plt.title(f'{pdb_file.stem}\nSingle-Residue Decoding \n(Fully-Conditional) Probability Heatmap', fontsize=12)
    plt.tight_layout()
    output_path = output_dir / f'conditional_probs.png'
    plt.savefig(output_path)

    fig, axes = plt.subplots(figsize=(5, 5), dpi=150)
    sns.heatmap(
        mean_plus_stdv_renormed,
        xticklabels=[aa_idx_to_short[idx] for idx in range(21)], 
        yticklabels=ylabels,
        cmap='Greens', vmin=0.0, vmax=0.5,
    )

    ground_truth_indices = [aa_long_to_idx[x.split('-')[0]] for x in ylabels]
    for i, gt_idx in enumerate(ground_truth_indices):
        plt.text(gt_idx + 0.5, i + 0.61, 'X', color='red', ha='center', va='center', fontsize=16)

    plt.title(f'{pdb_file.stem}\nSingle-Residue Decoding \n(Fully-Conditional) Mean + Std. Dev. Heatmap', fontsize=12)
    plt.tight_layout()
    output_path = output_dir / f'conditional_probs_mean_plus_stddev.png'
    plt.savefig(output_path)
    return output_path


def main(
    pdb_file: str, output_dir: str, device: str, weights: str, 
    disable_inference_dropout: bool, silent: bool, n_decoding_orders: int, 
    n_dropouts: int, repack_all: bool, selection_string: str,
):

    weights_path = Path(weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f'Weights file {weights_path} does not exist.')

    model, param_dict= load_model(weights, device, not disable_inference_dropout)
    if not silent: print(f'Loaded model with weights from `{weights_path.name}`')

    pdb_file_path = Path(pdb_file).resolve()
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(exist_ok=True, parents=True)

    if not silent: print(f'Running unconditional proofreading on {pdb_file_path.name}...')
    fs_mask, ylabels, uncond_output_path = compute_unconditional_probs(model, param_dict, pdb_file_path, output_dir_path, selection_string)
    if not silent: print(f'Unconditional proofreading complete. Output saved to {uncond_output_path}\n')

    if not silent: print('Running `fully-conditional` autoregressive, multi-dropout, multi-decoding order proofreading...')
    fullcond_output_path = compute_conditional_probs(model, param_dict, pdb_file_path, output_dir_path, fs_mask, ylabels, n_decoding_orders, n_dropouts, repack_all)
    if not silent: print(f'Fully-conditional proofreading complete. Output saved to {fullcond_output_path}\n')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Run proofreading on a protein complex.')
    parser.add_argument('pdb_file', type=str, help='Path to the pdb file of the protein complex.')
    parser.add_argument('output_dir', type=str, help='Path to an output directory.')
    parser.add_argument('--device', '-d', type=str, default='cuda:0', help='Device to run the model on.')
    parser.add_argument('--weights', '-w', type=str, default=str(CURR_FILE_DIR_PATH / "model_weights/laser_weights_0p1A_noise_ligandmpnn_split.pt"), help='Path to the weights file.')
    parser.add_argument('--disable_inference_dropout', action='store_true', help='Disable inference dropout.')
    parser.add_argument('--silent', '-s', action='store_true', help='Disable printing.')
    parser.add_argument('--n_decoding_orders', type=int, default=10, help='Number of decoding orders to use.')
    parser.add_argument('--n_dropouts', type=int, default=10, help='Number of dropouts to use.')
    parser.add_argument('--repack_all', action='store_true', help='Repack all residues.')
    parser.add_argument('--selection_string', '-r', type=str, default='', help='A Prody selection string to override the default behavior of proofreading the binding site only.')
    args = parser.parse_args()
    main(**vars(args))