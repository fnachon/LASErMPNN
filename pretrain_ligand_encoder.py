import os
from pathlib import Path
from copy import deepcopy
from typing import Optional
from collections import defaultdict

import wandb
from tqdm import tqdm
from scipy.spatial.transform import Rotation

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_scatter import scatter

from LASErMPNN.utils.model import SpiceDatasetPretrainingModule
from LASErMPNN.utils.spice_dataset import UnclusteredSpiceDataset, SpiceDatasetClusterSampler, SpiceBatchData, collate_spice_data, POSSIBLE_DEGREE_LIST, POSSIBLE_HYBRIDIZATION_LIST, POSSIBLE_NUM_HYDROGENS_LIST, POSSIBLE_FORMAL_CHARGE_LIST, POSSIBLE_IS_AROMATIC_LIST

CURR_FILE_DIR_PATH = Path(__file__).parent


compute_mse_loss = torch.nn.MSELoss()


def train_epoch(model: SpiceDatasetPretrainingModule, optimizer: torch.optim.Adam, train_dataloader: DataLoader, epoch_num: int, params: dict) -> dict:
    model.train()
    epoch_data = process_epoch(model, optimizer, train_dataloader, epoch_num, params)
    return {'train_' + x: y for x,y in epoch_data.items()}


@torch.no_grad()
def test_epoch(model: SpiceDatasetPretrainingModule, test_dataloader: DataLoader, epoch_num: int, params: dict) -> dict:
    model.eval()
    epoch_data = process_epoch(model, None, test_dataloader, epoch_num, params)
    return {'test_' + x: y for x,y in epoch_data.items()}


def process_epoch(model: SpiceDatasetPretrainingModule, optimizer: Optional[torch.optim.Adam], dataloader: DataLoader, epoch_num: int, params: dict) -> dict:
    """
    Process a single epoch.
    """
    output = dict()

    # Whether epoch is for training or testing.
    is_training_mode = False
    if optimizer is not None:
        is_training_mode = True
        output['lr'] = optimizer.param_groups[0]['lr']

    # Initialize epoch data.
    output_ints = defaultdict(int)
    output_floats = defaultdict(float)
    output_lists = defaultdict(list)
    for batch in tqdm(dataloader, total=len(dataloader), desc=f"{'Training' if is_training_mode else 'Testing'} Epoch {epoch_num}", leave=False, dynamic_ncols=True):
        batch: SpiceBatchData

        # Move batch to device.
        batch.to_device(model.device)

        # Construct graphs.
        graph_noise = params['training_noise'] if is_training_mode else 0.0
        batch.construct_graphs(graph_noise, params['model_params']['graph_structure']['lig_lig_knn_graph_k'], model.ligand_featurizer)
        pred_dipoles, pred_partial_charges, pred_mayer_order, pred_hybridization, pred_formal_charge, pred_num_connected_hydrogens, pred_possible_degree, pred_is_aromatic = model(batch)

        # label_sum_charge = scatter(batch.atomic_partial_charges, batch.batch_index, reduce='sum')
        # predicted_sum_charge = scatter(partial_charge_logits.flatten(), batch.batch_index, reduce='sum')
        # sum_charge_loss = compute_mse_loss(predicted_sum_charge, label_sum_charge)

        # pred_rdkit_features = torch.cat([pred_possible_degree, pred_hybridization, pred_num_connected_hydrogens, pred_formal_charge, pred_is_aromatic], dim=-1)
        pred_rdkit_features = torch.cat([pred_possible_degree, pred_hybridization, pred_num_connected_hydrogens, pred_is_aromatic], dim=-1)
        
        offset = 0
        bond_degree_accuracy = (pred_possible_degree.argmax(dim=-1) == torch.narrow(batch.atomic_rdkit_features, 1, offset, len(POSSIBLE_DEGREE_LIST)).argmax(dim=-1)).float().mean().item()
        offset += len(POSSIBLE_DEGREE_LIST)

        hybridization_accuracy = (pred_hybridization.argmax(dim=-1) == torch.narrow(batch.atomic_rdkit_features, 1, offset, len(POSSIBLE_HYBRIDIZATION_LIST)).argmax(dim=-1)).float().mean().item()
        offset += len(POSSIBLE_HYBRIDIZATION_LIST)

        attached_hydrogens_accuracy = (pred_num_connected_hydrogens.argmax(dim=-1) == torch.narrow(batch.atomic_rdkit_features, 1, offset, len(POSSIBLE_NUM_HYDROGENS_LIST)).argmax(dim=-1)).float().mean().item()
        offset += len(POSSIBLE_NUM_HYDROGENS_LIST)

        formal_charge_accuracy = (pred_formal_charge.argmax(dim=-1) == torch.narrow(batch.atomic_rdkit_features, 1, offset, len(POSSIBLE_FORMAL_CHARGE_LIST)).argmax(dim=-1)).float().mean().item()
        nonzero_formal_charge_mask = (torch.narrow(batch.atomic_rdkit_features, 1, offset, len(POSSIBLE_FORMAL_CHARGE_LIST)).argmax(dim=-1) != POSSIBLE_FORMAL_CHARGE_LIST.index(0))
        formal_charge_labels = torch.narrow(batch.atomic_rdkit_features, 1, offset, len(POSSIBLE_FORMAL_CHARGE_LIST))
        nonzero_formal_charge_accuracy = (pred_formal_charge.argmax(dim=-1)[nonzero_formal_charge_mask] == formal_charge_labels[nonzero_formal_charge_mask].argmax(dim=-1)).float().mean().item()
        offset += len(POSSIBLE_FORMAL_CHARGE_LIST)

        adjusted_rdkit_labels = torch.cat([
            torch.narrow(batch.atomic_rdkit_features, 1, 0, len(POSSIBLE_DEGREE_LIST) + len(POSSIBLE_HYBRIDIZATION_LIST) + len(POSSIBLE_NUM_HYDROGENS_LIST)), 
            torch.narrow(batch.atomic_rdkit_features, 1, offset, len(POSSIBLE_IS_AROMATIC_LIST))
        ], dim=-1)

        total_aromatic_accuracy = (pred_is_aromatic.argmax(dim=-1) == torch.narrow(batch.atomic_rdkit_features, 1, offset, len(POSSIBLE_IS_AROMATIC_LIST)).argmax(dim=-1)).float().mean().item()
        is_aromatic_mask = torch.narrow(batch.atomic_rdkit_features, 1, offset, len(POSSIBLE_IS_AROMATIC_LIST)).argmax(dim=-1) == 0
        is_aromatic_accuracy = (pred_is_aromatic.argmax(dim=-1)[is_aromatic_mask] == torch.narrow(batch.atomic_rdkit_features, 1, offset, len(POSSIBLE_IS_AROMATIC_LIST)).argmax(dim=-1)[is_aromatic_mask]).float().mean().item()

        rdkit_loss = params['rdkit_loss_scale_factor'] * F.cross_entropy(pred_rdkit_features, adjusted_rdkit_labels, reduction='mean')

        # Specifically compute the formal charge loss on a subset of nodes with nonzero formal charge.
        num_nonzero_formal_charge = nonzero_formal_charge_mask.sum().item()
        A = (~nonzero_formal_charge_mask).nonzero()
        shuffle = torch.randperm(A.shape[0])
        zero_indices = A[shuffle][:(num_nonzero_formal_charge * 10)].flatten()
        formal_charge_loss = params['formal_charge_loss_scale_factor'] * F.cross_entropy(
            torch.cat([pred_formal_charge[nonzero_formal_charge_mask], pred_formal_charge[zero_indices]]), 
            torch.cat([formal_charge_labels[nonzero_formal_charge_mask], formal_charge_labels[zero_indices]])
        )

        # Compute loss.
        partial_charge_loss = compute_mse_loss(pred_partial_charges.flatten(), batch.atomic_partial_charges)
        mayer_order_loss = torch.tensor(0) # params['mayer_order_loss_scale_factor'] * compute_mse_loss(pred_mayer_order, batch.atomic_mayer_order)

        dipole_loss = params['dipole_loss_scale_factor'] * compute_mse_loss(pred_dipoles.squeeze(1), batch.atomic_dipole_vectors)
        total_loss = partial_charge_loss + dipole_loss + mayer_order_loss + rdkit_loss + formal_charge_loss

        # Backpropagate and update model.
        if is_training_mode:
            total_loss.backward()
            optimizer.step() # type: ignore
            optimizer.zero_grad() # type: ignore

        # Log data.
        output_lists['partial_charge_loss'].append(partial_charge_loss.item())
        output_lists['mayer_order_loss'].append(mayer_order_loss.item())
        output_lists['rdkit_loss'].append(rdkit_loss.item())
        output_lists['formal_charge_loss'].append(formal_charge_loss.item())
        output_lists['dipole_loss'].append(dipole_loss.item())

        output_lists['bond_degree_accuracy'].append(bond_degree_accuracy)
        output_lists['hybridization_accuracy'].append(hybridization_accuracy)
        output_lists['attached_hydrogens_accuracy'].append(attached_hydrogens_accuracy)
        output_lists['formal_charge_accuracy'].append(formal_charge_accuracy)
        output_lists['nonzero_formal_charge_accuracy'].append(nonzero_formal_charge_accuracy)
        output_lists['total_aromatic_accuracy'].append(total_aromatic_accuracy)
        output_lists['is_aromatic_accuracy'].append(is_aromatic_accuracy)

        # output_lists['sum_charge_loss'].append(sum_charge_loss.item())
        output_lists['combination_loss'].append(total_loss.item())
    
    output_lists_normed = {x: sum(y) / max(1, len(y)) for x,y in output_lists.items()}
    output.update(dict(output_ints))
    output.update(dict(output_floats))
    output.update(dict(output_lists_normed))
    return output


def main(params: dict):

    device = torch.device(params['device'])
    model = SpiceDatasetPretrainingModule(**params['model_params'], use_hydrogens=params['use_hydrogens']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    total_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
    print(f'Total parameters to train: {total_params}')

    # Load dataset clusters.
    train_cluster_path = CURR_FILE_DIR_PATH / 'files/spice_clusters/train_clusters.pt'
    test_cluster_path = CURR_FILE_DIR_PATH / 'files/spice_clusters/test_clusters.pt'

    # Load dataset.
    spice_dataset = UnclusteredSpiceDataset(**params)
    if not os.path.exists(train_cluster_path) or not os.path.exists(test_cluster_path):
        train_clusters, test_clusters = spice_dataset.generate_random_split()
        torch.save(train_clusters, train_cluster_path)
        torch.save(test_clusters, test_cluster_path)
    else:
        train_clusters = torch.load(train_cluster_path, weights_only=True)
        test_clusters = torch.load(test_cluster_path, weights_only=True)

    # Initialize dataloaders.
    train_spice_sampler = SpiceDatasetClusterSampler(spice_dataset, train_clusters, **params)
    test_spice_sampler = SpiceDatasetClusterSampler(spice_dataset, test_clusters, **params)
    train_dataloader = DataLoader(spice_dataset, batch_sampler=train_spice_sampler, collate_fn=collate_spice_data, num_workers=params['num_dataloader_workers'], persistent_workers=True)
    test_dataloader = DataLoader(spice_dataset, batch_sampler=test_spice_sampler, collate_fn=collate_spice_data, num_workers=params['num_dataloader_workers'], persistent_workers=True)

    epoch_num = -1
    for epoch_num in range(params['num_epochs']):
        train_epoch_data = train_epoch(model, optimizer, train_dataloader, epoch_num, params)
        test_epoch_data = {}
        if epoch_num % 5 == 0:
            test_epoch_data = test_epoch(model, test_dataloader, epoch_num, params)
            torch.save({'params': params, 'model_state_dict': model.state_dict()}, f"{params['output_path_prefix']}_epoch_{epoch_num}.pt")
        
        epoch_data = {'epoch': epoch_num, **train_epoch_data, **test_epoch_data}

        if params['use_wandb']:
            wandb.log(dict(epoch_data))
        
        out = []
        for key, value in epoch_data.items():
            out.append(f"{key}: {value}")
        print(', '.join(out))

    torch.save({'params': params, 'model_state_dict': model.state_dict()}, f"{params['output_path_prefix']}_epoch_{epoch_num}.pt")


if __name__ == "__main__":
    all_params = {
        'device': 'cuda:0',
        'output_path_prefix': CURR_FILE_DIR_PATH / 'model_weights/retrained_ligand_encoder_',
        'debug': (debug := False),
        'use_wandb': (True and not debug),
        'num_epochs': 100,
        'batch_size': 10_000,
        'learning_rate': 1e-3,
        'training_noise': 0.05,

        'dipole_loss_scale_factor': 10,
        'formal_charge_loss_scale_factor': 1e-2,
        'mayer_order_loss_scale_factor': 1e-3,
        'rdkit_loss_scale_factor': 1e-3,

        'use_hydrogens': True,
        'sample_randomly': True,
        'model_params': {
            'num_ligand_encoder_vectors': 15,
            'node_embedding_dim': 256, 
            'ligand_edge_embedding_dim': 128, 
            'atten_dimension_upscale_factor': None,
            'lig_lig_edge_rbf_params': {
                'num_bins': 75,
                'bin_min': 0.0,
                'bin_max': 15.0,
            }, 
            'graph_structure': {
                'lig_lig_knn_graph_k': 5,
            },
            'num_encoder_layers': 3,
            'num_attention_heads': 3,
            'dropout': 0.1,
            'atten_head_aggr_layers': 0,
        },
        'num_dataloader_workers': 5,
        'path_to_dataset': CURR_FILE_DIR_PATH / 'databases/spice_dataset/SPICE-2.0.1.hdf5',
    }
    if all_params['use_wandb']:
        run = wandb.init(project='LigandEncoderPretraining', entity='benf549', config=all_params)
    main(all_params)
