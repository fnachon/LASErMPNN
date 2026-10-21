import os
import h5py
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn.functional as F
from .graph_ops import knn_graph
from torch.utils.data import Dataset, Sampler

from rdkit import Chem
from .ligand_featurization import LigandFeaturizer
from .constants import atomic_number_to_atom, ANGSTROM_PER_BOHR, POSSIBLE_HYBRIDIZATION_LIST, POSSIBLE_FORMAL_CHARGE_LIST, POSSIBLE_NUM_HYDROGENS_LIST, POSSIBLE_DEGREE_LIST, POSSIBLE_IS_AROMATIC_LIST
from .helper_functions import write_xyz_file
from .pdb_dataset import LigandData

from typing import Tuple, List, Optional


@dataclass
class SPICEdatapoint():
    # Atomic numbers of the small molecule.
    atomic_numbers: h5py.Group 
    # Conformations of the small molecule.
    conformations: h5py.Group # (M, N, 3) # Coordinates are in Bohr.
    formation_energy: h5py.Group # (M, 1) # Units are Hartree. 
    # Features for prediction of partial charges.
    mbis_charges: h5py.Group # (N, 1)
    mbis_dipoles: h5py.Group # (N, 3)
    mayer_indices: h5py.Group # (N, 1)
    # Smiles string for the small molecule.
    smiles: h5py.Group
    # The subset of SPICE that this datapoint belongs to (ex: PubChem, dipeptides, ...)
    subset: h5py.Group

    # These are unused.
    dft_total_energy: h5py.Group
    dft_total_gradient: h5py.Group
    formation_energy: h5py.Group
    mbis_quadrupoles: h5py.Group
    mbis_octupoles: h5py.Group
    scf_dipole: h5py.Group
    scf_quadrupole: h5py.Group
    wiberg_lowdin_indices: h5py.Group

    @property
    def num_atoms(self) -> int:
        return len(self.atomic_numbers)

    def to_xyz(self, conformation_index: int, file_path: str):
        """
        Writes the small molecule to an xyz file.
        """
        atomic_symbols = [atomic_number_to_atom[x] for x in self.atomic_numbers] # type: ignore
        angstrom_coords = self.conformations[conformation_index] * ANGSTROM_PER_BOHR # type: ignore
        smiles = self.smiles[0].decode('utf-8') # type: ignore
        write_xyz_file(file_path, atomic_symbols, angstrom_coords, smiles)


@dataclass
class ProcessedSPICEDatapoint():
    atomic_numbers: torch.Tensor
    coords: torch.Tensor
    atomic_partial_charges: torch.Tensor
    conformer_dipoles: torch.Tensor
    rdkit_features: torch.Tensor
    atomic_mayer_order: torch.Tensor
    smiles: str

    @property
    def size(self) -> int:
        return self.atomic_numbers.shape[0]


@dataclass
class SpiceBatchData():
    """
    Attributes:
        lig_nodes, lig_coords, lig_lig_edge_distance, lig_lig_edge_index, atomic_partial_charges
    
    Properties:
        device, num_residues

    Methods:
        construct_graphs
    """
    # NOTE: Attribute names need to match those in LigandEncoderModule forward pass.
    lig_atomic_number: torch.Tensor
    lig_coords: torch.Tensor
    batch_index: torch.Tensor

    atomic_partial_charges: torch.Tensor
    atomic_dipole_vectors: torch.Tensor
    atomic_mayer_order: torch.Tensor
    atomic_rdkit_features: torch.Tensor

    ligand_data: Optional[LigandData] = None

    def construct_graphs(self, noise: float, lig_lig_knn_graph_k: int, ligand_featurizer: LigandFeaturizer):
        # Compute ligand features.
        lig_nodes = ligand_featurizer.encode_ligand_from_atomic_number(self.lig_atomic_number)

        # Compute ligand-ligand edge distance.
        lig_lig_edge_index = knn_graph(self.lig_coords, k=lig_lig_knn_graph_k, batch=self.batch_index, loop=True)
        noised_coords = self.lig_coords + (noise * torch.randn_like(self.lig_coords))
        lig_lig_edge_distance = torch.cdist(noised_coords[lig_lig_edge_index[0]].unsqueeze(1), noised_coords[lig_lig_edge_index[1]].unsqueeze(1)).flatten()

        self.ligand_data = LigandData(lig_nodes, noised_coords, self.batch_index, torch.zeros_like(self.batch_index), lig_lig_edge_index, lig_lig_edge_distance)

    def to_device(self, device: torch.device) -> None:
        """
        Moves all tensors in the SpiceBatchData object to the specified device.
        """
 
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
            if  isinstance(v, LigandData):
                self.__dict__[k] = v.to_device(device)

    @property
    def device(self) -> torch.device:
        """
        Returns the device that the batch tensors are currently on when addressed as model.device
        """
        return self.lig_coords.device

    @property
    def num_residues(self) -> int:
        """
        Returns the number of residues in the batch.
        """
        return self.lig_coords.shape[0]


def safe_index(target_list, value):
    """
    Returns the last index in a list if value is not an element of the list.
    Otherwise returns the index of that value in list.
    """
    try:
        target_index = target_list.index(value)
    except (ValueError, TypeError) as e:
        target_index = len(target_list) - 1

    return target_index


def safe_one_hot(feature_list, feature_value):
    index = safe_index(feature_list, feature_value)
    ohe = F.one_hot(torch.tensor(index), num_classes = len(feature_list))
    return ohe


def extract_features_from_mapped_smiles_string(smiles: str):
    rdmol = Chem.MolFromSmiles(smiles, sanitize=False) # type: ignore
    assert rdmol is not None, "Unable to parse the SMILES string"

    # strip the atom map from the molecule if it has one
    # so we don't affect the sterochemistry tags
    for atom in rdmol.GetAtoms():
        if atom.GetAtomMapNum() != 0:
            # set the map back to zero but hide the index in the atom prop data
            atom.SetProp("_map_idx", str(atom.GetAtomMapNum())) # type: ignore
            # set it back to zero
            atom.SetAtomMapNum(0)

    # Chem.SanitizeMol calls updatePropertyCache so we don't need to call it ourselves
    # https://www.rdkit.org/docs/cppapi/namespaceRDKit_1_1MolOps.html#a8d831787aaf2d65d9920c37b25b476f5
    Chem.SanitizeMol(rdmol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_ADJUSTHS ^ Chem.SANITIZE_SETAROMATICITY) # type: ignore
    Chem.SetAromaticity(rdmol, Chem.AromaticityModel.AROMATICITY_MDL) # type: ignore

    # Chem.MolFromSmiles adds bond directions (i.e. ENDDOWNRIGHT/ENDUPRIGHT), but
    # doesn't set bond.GetStereo(). We need to call AssignStereochemistry for that.
    Chem.AssignStereochemistry(rdmol) # type: ignore

    # Create map dict from rdkit index to spice index.
    rdkit_idx_to_spice_idx = {}
    for atom_idx in range(rdmol.GetNumAtoms()):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        assert atom.GetNumImplicitHs() == 0, "Expected no implicit hydrogens"
        rdkit_idx_to_spice_idx[atom_idx] = int(atom.GetProp("_map_idx")) - 1

    output_rdkit_features_matrix = torch.zeros(
        rdmol.GetNumAtoms(), 
        len(POSSIBLE_HYBRIDIZATION_LIST) + len(POSSIBLE_FORMAL_CHARGE_LIST) + len(POSSIBLE_NUM_HYDROGENS_LIST) + len(POSSIBLE_DEGREE_LIST) + len(POSSIBLE_IS_AROMATIC_LIST)
    )

    for idx, atom in enumerate(rdmol.GetAtoms()):
        spice_idx = rdkit_idx_to_spice_idx[idx]

        # Create a simple fingerprint for each atom (for prediction since we don't want to use RDKit for downstream training.)
        bond_degree_one_hot = safe_one_hot(POSSIBLE_DEGREE_LIST, atom.GetDegree())
        bond_hybridization_one_hot = safe_one_hot(POSSIBLE_HYBRIDIZATION_LIST, str(atom.GetHybridization()))
        total_attached_hydrogens_one_hot = safe_one_hot(POSSIBLE_NUM_HYDROGENS_LIST, atom.GetTotalNumHs(includeNeighbors=True))
        atomic_formal_charges_one_hot = safe_one_hot(POSSIBLE_FORMAL_CHARGE_LIST, atom.GetFormalCharge())
        atom_is_aromatic_one_hot = safe_one_hot(POSSIBLE_IS_AROMATIC_LIST, atom.GetIsAromatic())

        output_rdkit_features_matrix[spice_idx] = torch.cat([
            bond_degree_one_hot, 
            bond_hybridization_one_hot, 
            total_attached_hydrogens_one_hot, 
            atomic_formal_charges_one_hot, 
            atom_is_aromatic_one_hot
        ])
    
    return output_rdkit_features_matrix

class UnclusteredSpiceDataset(Dataset):
    def __init__(self, use_hydrogens: bool, path_to_dataset: str, debug: bool, **kwargs):
        self.use_hydrogens = use_hydrogens
        print('Loading Dataset...')
        self.h5file = h5py.File(path_to_dataset, 'r')
        self.key_list = list(self.h5file.keys())
        
        # Filter key list to only include keys with mbis_charges and mbis_dipoles.
        exclude_list = ['54X VAL']
        new_key_list = []
        size_list = []
        for idx, x in tqdm(enumerate(self.key_list), total=len(self.key_list), desc='Filtering dataset...'):
            if ('mbis_charges' in self.h5file[x].keys()) and ('mbis_dipoles' in self.h5file[x].keys()) and ('atomic_numbers' in self.h5file[x].keys()) and x not in exclude_list:
                new_key_list.append(x)
                size_list.append(len(self.h5file[x]['atomic_numbers']))

            if debug and (idx > 100):
                break

        self.key_list = new_key_list
        self.key_to_index = {j:i for i,j in enumerate(self.key_list)}

        # Compute the number of non-hydrogen atoms in each molecule.
        if not self.use_hydrogens:
            self.index_to_size = [] 
            for key in self.key_list:
                curr_data = SPICEdatapoint(**self.h5file[key]) # type: ignore
                # TODO: This can definitely be optimized.
                hydrogen_mask = torch.tensor(curr_data.atomic_numbers) == 1
                self.index_to_size.append((~hydrogen_mask).sum().item())
        else:
            # self.index_to_size = [len(self.h5file[key]['atomic_numbers']) for key in self.key_list] # type: ignore
            self.index_to_size = size_list

    def __len__(self) -> int:
        return len(self.key_list)
    
    def __getitem__(self, index):
        spice_datapoint = SPICEdatapoint(**self.h5file[self.key_list[index]]) # type: ignore
        smiles = spice_datapoint.smiles[0].decode('utf-8') # type: ignore

        # Extract useful rdkit features for additional predictions.
        rdkit_feature_matrix = extract_features_from_mapped_smiles_string(smiles) # type: ignore

        # Extract atomic numbers.
        atomic_numbers = torch.tensor(spice_datapoint.atomic_numbers, dtype=torch.long)
        hydrogen_mask = atomic_numbers == 1

        # Sample a conformer.
        num_conformations = len(spice_datapoint.conformations)
        assert num_conformations > 0, f"datapoint: {spice_datapoint}, {self.key_list[index]}, {spice_datapoint.conformations}"
        sampled_conformer_idx = torch.randint(0, num_conformations, (1,)).item()
        conformer_coords = torch.tensor(spice_datapoint.conformations[sampled_conformer_idx])

        # Computes a mayer 'order' by summing over mayer indices of all bonds to a given atom.
        atomic_mayer_order = torch.tensor(spice_datapoint.mayer_indices[sampled_conformer_idx]).sum(dim=-1, keepdim=True)

        # Extract node charges.
        conformer_charges = torch.tensor(spice_datapoint.mbis_charges[sampled_conformer_idx]).flatten()
        conformer_dipoles = torch.tensor(spice_datapoint.mbis_dipoles[sampled_conformer_idx])

        # Drop hydrogen nodes if necessary.
        if not self.use_hydrogens:
            atomic_numbers = atomic_numbers[~hydrogen_mask]
            conformer_coords = conformer_coords[~hydrogen_mask]
            conformer_charges = conformer_charges[~hydrogen_mask]
            conformer_dipoles = conformer_dipoles[~hydrogen_mask]

        return ProcessedSPICEDatapoint(atomic_numbers, conformer_coords, conformer_charges, conformer_dipoles, rdkit_feature_matrix, atomic_mayer_order, smiles)
    
    def generate_random_split(self, train_fraction: float = 0.8) -> Tuple[list, list]:
        """
        Returns a random split of the dataset into train and test keys.
        """
        # Shuffle key list in place.
        key_list_copy = deepcopy(self.key_list)
        np.random.shuffle(key_list_copy)

        # Split key list into train and test keys.
        num_train_indices = int(len(key_list_copy) * train_fraction)
        train_keys = key_list_copy[:num_train_indices]
        test_keys = key_list_copy[num_train_indices:]

        return train_keys, test_keys


class SpiceDatasetClusterSampler(Sampler):
    def __init__(self, dataset: UnclusteredSpiceDataset, clusters: list, batch_size: int, sample_randomly: bool, **kwargs):
        self.spice_dataset = dataset
        self.batch_size = batch_size
        self.sample_randomly = sample_randomly

        self.split_indices = [self.spice_dataset.key_to_index[x] for x in clusters if x in self.spice_dataset.key_to_index]
        self.split_sizes = [self.spice_dataset.index_to_size[x] for x in self.split_indices]

        self.batch_list = []
        self.construct_batches()

    def __len__(self):
        return len(self.batch_list)
    
    def construct_batches(self):
        self.batch_list = []
        curr_samples_indices = torch.tensor(self.split_indices)
        curr_samples_sizes = torch.tensor(self.split_sizes)
        size_sort_indices = torch.argsort(curr_samples_sizes)

        counter = 0
        curr_list_sample_indices, curr_list_sizes = [], []
        while counter < len(size_sort_indices):
            while sum(curr_list_sizes) < self.batch_size and counter < len(size_sort_indices):
                curr_size_sort_index = size_sort_indices[counter]
                curr_sample_index = curr_samples_indices[curr_size_sort_index].item()
                curr_size = curr_samples_sizes[curr_size_sort_index].item()

                # Record the current sample and its size.
                curr_list_sample_indices.append(curr_sample_index)
                curr_list_sizes.append(curr_size)
                counter += 1

            # Add the current batch to the list of batches.
            self.batch_list.append(curr_list_sample_indices)

            # Reset the current batch.
            curr_list_sample_indices, curr_list_sizes = [], []
        
        # Shuffle the batches.
        if self.sample_randomly:
            np.random.shuffle(self.batch_list)
        
    def __iter__(self):
        # Yield batches constructed from construct_batches call.
        for batch in self.batch_list:
            yield batch
        
        # Construct new batches.
        self.construct_batches()


def collate_spice_data(batch_list: List[ProcessedSPICEDatapoint]) -> SpiceBatchData:
    """
    Convert a list of ProcessedSPICEDatapoint objects into a BatchData object for use in torch.utils.DataLoader.
    """

    output_tensor_dict = defaultdict(list)
    for mol_idx, molecule in enumerate(batch_list):
        output_tensor_dict['lig_atomic_number'].append(molecule.atomic_numbers)
        output_tensor_dict['lig_coords'].append(molecule.coords * ANGSTROM_PER_BOHR) # Convert from Bohr to Angstrom.
        output_tensor_dict['batch_index'].append(torch.full((molecule.size,), mol_idx, dtype=torch.long))
        output_tensor_dict['atomic_partial_charges'].append(molecule.atomic_partial_charges)
        output_tensor_dict['atomic_dipole_vectors'].append(molecule.conformer_dipoles * ANGSTROM_PER_BOHR)
        output_tensor_dict['atomic_mayer_order'].append(molecule.atomic_mayer_order)
        output_tensor_dict['atomic_rdkit_features'].append(molecule.rdkit_features)
    
    outputs = {}
    for i,j in output_tensor_dict.items():
        if None in j:
            assert False, f"Missing data for {i}"
        outputs[i] = torch.cat(j, dim=0)

    return SpiceBatchData(**outputs)
