#!/usr/bin/env python3

"""
Runs tests for the equivariance of the compoenents of the LASErMPNN model.
Execute by running `pytest` command in the root directory of the repository.
"""

import pytest
from copy import deepcopy
from scipy.spatial.transform import Rotation

import torch
import torch.nn as nn
from utils.model import LASErMPNN, SpiceDatasetPretrainingModule
from utils.pdb_dataset import BatchData, LigandData
from utils.spice_dataset import SpiceBatchData, POSSIBLE_DEGREE_LIST, POSSIBLE_FORMAL_CHARGE_LIST, POSSIBLE_HYBRIDIZATION_LIST, POSSIBLE_IS_AROMATIC_LIST, POSSIBLE_NUM_HYDROGENS_LIST
from utils.constants import aa_short_to_idx


@pytest.fixture
def params() -> dict:
    params = {
        'debug': (debug := True), 'use_wandb': True and not debug, 'device': 'cpu',
        'dataset_path': '/scratch/bfry/torch_bioasmb_dataset_chainmasks_dedup_seqres_corrected' + ('/w7' if debug else ''),
        'num_dataloader_workers': 2, 'num_epochs': 100, 'batch_size': 10_000, 'num_residues_per_ligand': 250, 'min_contact_number_for_sampling': 1,
        'max_protein_size': 5_000, 'msa_loss_weight': 0.25, 'protein_training_noise': 0.00, 'ligand_training_noise': 0.00, 'subgraph_only_dropout_rate': 0.00,
        'num_adjacent_residues_to_drop': 7, 'sample_randomly': True, 'autoregressive_test_batch_sample_fraction': 0.1 if not debug else 1.0,
        'model_params': {
            'build_hydrogens': True, 'additional_ligand_mlp': True, 'num_ligand_encoder_vectors': 5, 'num_laser_vectors': 25,
            'node_embedding_dim': 128, 'protein_edge_embedding_dim': 128, 'ligand_edge_embedding_dim': 64,
            'atten_dimension_upscale_factor': 2,
            'dropout': 0.0, 'num_encoder_layers': 3, 'num_decoder_layers': 3, 'chi_angle_rbf_bin_width': 5,
            'num_attention_heads': 3, 'atten_head_aggr_layers': 0,
            'graph_structure': {
                'pr_pr_knn_graph_k': 16, 'lig_pr_distance_cutoff': 15.0, 'lig_pr_knn_graph_k': 16, 'lig_lig_knn_graph_k': 5,
            },
            'prot_prot_edge_rbf_params': {
                'num_bins': 16, 'bin_min': 2, 'bin_max': 22,
            },
            'lig_prot_edge_rbf_params': {
                'num_bins': 75, 'bin_min': 0.0, 'bin_max': 15.0,
            },
            'lig_lig_edge_rbf_params': {
                'num_bins': 75, 'bin_min': 0.0, 'bin_max': 15.0,
            },
        },
        'train_splits_path': ('./files/protein_pretraining_clusters/train_splits_debug.pt' if debug else './files/protein_pretraining_clusters/train_splits.pt'),
        'test_splits_path': ('./files/protein_pretraining_clusters/test_splits_debug.pt' if debug else './files/protein_pretraining_clusters/test_splits.pt'),
        'clustering_output_prefix': 'cluster30test',
        'clustering_output_path': '/scratch/bfry/bioasmb_dataset_sequence_clustering_dedup_seqres/',
    }
    return params

@pytest.fixture
def ligand_encoder_params() -> dict:
    output = {
        'model_params': {
            'atten_dimension_upscale_factor': 2,
            'dropout': 0.1,
            'atten_head_aggr_layers': 0,
            'num_attention_heads': 3,
            'num_ligand_encoder_vectors': 5,
            'ligand_edge_embedding_dim': 128,
            'node_embedding_dim': 128,
            'num_encoder_layers': 3,
            'lig_lig_edge_rbf_params': {
                'num_bins': 75, 'bin_min': 0.0, 'bin_max': 15.0,
            },
        }
    }
    return output


@pytest.fixture
def dummy_batch_data(params: dict) -> BatchData:
    num_residues = 100
    bd_args = {
        'pdb_codes': ['DummyData'],

        # Making everything alanine so we don't have to build sidechains in ligand sampling 
        #   (SVD in sidechain building doesn't converge on dummy data).
        'sequence_indices': torch.full((num_residues,), aa_short_to_idx['A']),
        'chi_angles': torch.full((num_residues, 4), torch.nan),
        'backbone_coords': torch.FloatTensor(num_residues, 5, 3).uniform_(-100, 100),
        'phi_psi_angles': torch.randint(-179, 179, (num_residues, 2)),
        'sidechain_contact_number': torch.randint(0, 10, (num_residues,)),
        'residue_burial_counts': torch.randint(0, 10, (num_residues,)),
        'batch_indices': torch.zeros(num_residues, dtype=torch.int),
        'chain_indices': torch.zeros(num_residues, dtype=torch.int),
        'resnum_indices': torch.arange(num_residues),
        'chain_mask': torch.zeros(num_residues, dtype=torch.bool),
        'sampled_chain_mask': torch.zeros(num_residues, dtype=torch.bool),
        'extra_atom_contact_mask': torch.zeros(num_residues, dtype=torch.bool),
        'msa_data': torch.zeros(num_residues, 21),
        'msa_depth_weight': torch.full((num_residues,), 1),
        'first_shell_ligand_contact_mask': torch.zeros(num_residues, dtype=torch.bool),
        'sc_mediated_hbond_counts': torch.zeros(num_residues, dtype=torch.int),
    }
    dummy_batch_data = BatchData(**bd_args)
    return dummy_batch_data


@pytest.fixture
def model(params: dict, ligand_encoder_params: dict) -> nn.Module:
    model = LASErMPNN(ligand_encoder_params=ligand_encoder_params, **params['model_params'], return_embeddings=True)
    return model


@pytest.fixture
def dummy_spice_batch_data():
    num_dummy_atoms = 100
    sbd_args = {
        'lig_atomic_number': torch.randint(1, 50, (num_dummy_atoms,)),
        'lig_coords': torch.FloatTensor(num_dummy_atoms, 3).uniform_(-100, 100),
        'batch_index': torch.zeros(num_dummy_atoms, dtype=torch.int),
        'atomic_partial_charges': torch.randn(num_dummy_atoms),
        'atomic_dipole_vectors': torch.randn(num_dummy_atoms, 3),
        'atomic_mayer_order': torch.randn(num_dummy_atoms),
        'atomic_rdkit_features': torch.randn(num_dummy_atoms, len(POSSIBLE_DEGREE_LIST) + len(POSSIBLE_FORMAL_CHARGE_LIST) + len(POSSIBLE_HYBRIDIZATION_LIST) + len(POSSIBLE_IS_AROMATIC_LIST) + len(POSSIBLE_NUM_HYDROGENS_LIST)), 

        'ligand_data': None
    }
    return SpiceBatchData(**sbd_args)


@pytest.fixture
def ligand_encoder_model(params: dict):
    model = SpiceDatasetPretrainingModule(**params['model_params'], use_hydrogens=params['model_params']['build_hydrogens'])
    return model


@pytest.fixture
def rotation_matrix():
    return torch.as_tensor(Rotation.random().as_matrix(), dtype=torch.float32)

@pytest.fixture
def linear_offset():
    return 10 * torch.randn(1, 3)

def test_ligand_encoder_equivariance(dummy_spice_batch_data: SpiceBatchData, ligand_encoder_model: SpiceDatasetPretrainingModule, params: dict, rotation_matrix: torch.Tensor, linear_offset: torch.Tensor):
    dummy_spice_batch_data.construct_graphs(0.0, params['model_params']['graph_structure']['lig_lig_knn_graph_k'], ligand_encoder_model.ligand_featurizer)

    # Copy batch clone and rotrate the ligand coordinates.
    batch_clone = deepcopy(dummy_spice_batch_data)
    assert batch_clone.ligand_data is not None, "Need ligand data to run equivariance test."
    batch_clone.ligand_data.lig_coords = (batch_clone.ligand_data.lig_coords @ rotation_matrix) + linear_offset

    ligand_encoder_model.eval()
    with torch.no_grad():
        # Run the model on the original batch data.
        output_vec, partial_charge_logits, *_ = ligand_encoder_model(dummy_spice_batch_data)

        # Run the model on the rotated batch data.
        rotated_output_vec, rotated_partial_charge_logits, *_  = ligand_encoder_model(batch_clone)

        # Test output equivariance.
        assert torch.isclose(partial_charge_logits, rotated_partial_charge_logits, rtol=1e-4, atol=1e-4).all().item(), 'Partial charge logits are not equivariant.'
        assert torch.isclose((output_vec @ rotation_matrix), rotated_output_vec, rtol=1e-4, atol=1e-4).all().item(), 'Output vectors are not equivariant.'


def test_lasermpnn_equivariance(dummy_batch_data: BatchData, model: LASErMPNN, params: dict, rotation_matrix: torch.Tensor, linear_offset: torch.Tensor):
    """
    Test the equivariance of the model by rotating the input coordinates and asserting that the 
        output is the same as the output without rotation. Also tests that embedding vectors are rotated accordingly.
    """
    # Construct the graphs and generate the decoding order for the batch data.
    # Since edges have only invariant representations, we can use the same graph structure for both the original and rotated batch data.
    dummy_batch_data.sample_pseudoligands(params['num_residues_per_ligand'], params['min_contact_number_for_sampling'], model.rotamer_builder)
    dummy_batch_data.construct_graphs(
            model.rotamer_builder, 
            model.ligand_featurizer, 
            **params['model_params']['graph_structure'],
            protein_training_noise=params['protein_training_noise'], 
            ligand_training_noise=params['ligand_training_noise'], 
            subgraph_only_dropout_rate=params['subgraph_only_dropout_rate'],
            num_adjacent_residues_to_drop=params['num_adjacent_residues_to_drop'],
            build_hydrogens=params['model_params']['build_hydrogens'],
    )
    dummy_batch_data.generate_decoding_order()

    # Make a deep copy of the batch data for testing rotations.
    batch_clone = deepcopy(dummy_batch_data)

    # Appease type checker.
    assert batch_clone.ligand_data is not None, "Need ligand data to run equivariance test."

    # Create a rotation matrix and rotate the backbone and ligand coordinates.
    batch_clone.backbone_coords = (batch_clone.backbone_coords @ rotation_matrix) + linear_offset
    batch_clone.ligand_data.lig_coords = (batch_clone.ligand_data.lig_coords @ rotation_matrix) + linear_offset

    model.eval()
    with torch.no_grad():
        # Run the model on the original batch data.
        sequence_logits, output_chi_logits, prot_nodes, lig_nodes = model(dummy_batch_data, return_nodes=True)

        # Run the model on the rotated batch data.
        rotated_sequence_logits, rotated_output_chi_logits, rotated_prot_nodes, rotated_lig_nodes = model(batch_clone, return_nodes=True)

        # Test logit equivariance.
        assert torch.isclose(sequence_logits, rotated_sequence_logits, rtol=1e-4, atol=1e-4).all().item(), 'Sequence logits are not equivariant.'
        assert torch.isclose(output_chi_logits.nan_to_num(), rotated_output_chi_logits.nan_to_num(), rtol=1e-4, atol=1e-4).all().item(), 'Output chi logits are not equivariant.'

        # Test embedding equivariance.
        assert torch.isclose(prot_nodes.scalars, rotated_prot_nodes.scalars, rtol=1e-4, atol=1e-4).all().item(), 'Protein node scalars are not equivariant.'
        assert torch.isclose(prot_nodes.vectors @ rotation_matrix, rotated_prot_nodes.vectors, rtol=1e-4, atol=1e-4).all().item(), 'Protein node vectors are not equivariant.'
        assert torch.isclose(lig_nodes.scalars, rotated_lig_nodes.scalars, rtol=1e-4, atol=1e-4).all().item(), 'Ligand node scalars are not equivariant.'
        assert torch.isclose(lig_nodes.vectors @ rotation_matrix, rotated_lig_nodes.vectors, rtol=1e-4, atol=1e-4).all().item(), 'Ligand node vectors are not equivariant.'


def test_embedding_vector_magnitudes(dummy_batch_data: BatchData, model: LASErMPNN, params: dict):
    """
    Test that the magnitudes of the embedding vectors are the same for the same input data.
    Attempting to catch normalization issues here, this happens when you remove the `self.equivariant_layer_norm` from the HomoGATv2 baseclass.
    """

    # The vector outputs of the model should have a minimum layer mean magnitude of 1e-4.
    min_layer_norm_magnitude = 1e-4

    # Construct the graphs and generate the decoding order for the batch data.
    dummy_batch_data.sample_pseudoligands(params['num_residues_per_ligand'], params['min_contact_number_for_sampling'], model.rotamer_builder)
    dummy_batch_data.construct_graphs(
            model.rotamer_builder, 
            model.ligand_featurizer, 
            **params['model_params']['graph_structure'],
            protein_training_noise=params['protein_training_noise'], 
            ligand_training_noise=params['ligand_training_noise'],
            subgraph_only_dropout_rate=params['subgraph_only_dropout_rate'],
            num_adjacent_residues_to_drop=params['num_adjacent_residues_to_drop'],
            build_hydrogens=params['model_params']['build_hydrogens']
    )
    dummy_batch_data.generate_decoding_order()

    model.eval()
    with torch.no_grad():
        # Run the model on the original batch data.
        _, _, prot_nodes, lig_nodes = model(dummy_batch_data, return_nodes=True)
        assert (torch.linalg.vector_norm(prot_nodes.vectors, dim=-1).mean(dim=-1) > min_layer_norm_magnitude).all().item(), "Protein node vectors are too small"
        if lig_nodes.vectors.numel() > 0:
            assert (torch.linalg.vector_norm(lig_nodes.vectors, dim=-1).mean(dim=-1) > min_layer_norm_magnitude).all().item(), "Ligand node vectors are too small"
