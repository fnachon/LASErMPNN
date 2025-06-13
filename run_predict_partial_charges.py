import torch
import argparse
import numpy as np
import prody as pr
from pathlib import Path

from LASErMPNN.utils.model import SpiceDatasetPretrainingModule
from LASErMPNN.utils.spice_dataset import SpiceBatchData
from LASErMPNN.utils.constants import atom_to_atomic_number, POSSIBLE_FORMAL_CHARGE_LIST

CURR_FILE_DIR_PATH = Path(__file__).parent

def main(checkpoint_path: str, inference_molecule_path: str, output_path: str, inference_device_str: str = 'cpu'):

    # Extract model params and state_dict
    param_state_dict = torch.load(checkpoint_path, map_location=inference_device_str, weights_only=True)
    params = param_state_dict['params']

    # Load from model checkpoint
    model = SpiceDatasetPretrainingModule(**params['model_params'], use_hydrogens=params['use_hydrogens'])
    model.load_state_dict(param_state_dict['model_state_dict'])
    model.eval()

    # Load ligand from PDB file. Should have hydrogens if model was trained with hydrogens.
    ligand = pr.parsePDB(inference_molecule_path)
    assert isinstance(ligand, pr.AtomGroup), 'unreachable.'
    ligand = ligand.select('not water').copy()
    coords_list = []
    atoms_list = []
    for atom in ligand:
        coords_list.append(atom.getCoords())
        atoms_list.append(atom.getElement()) # type: ignore
    all_coords = torch.tensor(np.array(coords_list)).float()
    all_atomic_numbers = torch.tensor([atom_to_atomic_number[atom.capitalize()] for atom in atoms_list])

    # Construct SpiceBatchData object
    data = SpiceBatchData(
        lig_atomic_number=all_atomic_numbers,
        lig_coords=all_coords,
        batch_index=torch.zeros_like(all_atomic_numbers),
        atomic_partial_charges=torch.empty((0,)),
        atomic_dipole_vectors=torch.empty((0, 3)),
        atomic_mayer_order=torch.empty((0,)),
        atomic_rdkit_features=torch.empty((0,))
    )
    data.construct_graphs(noise=0.0, **params['model_params']['graph_structure'], ligand_featurizer=model.ligand_featurizer)

    output_ligand = pr.AtomGroup()
    with torch.no_grad():
        # Run inference
        pred_dipoles, pred_partial_charges, pred_mayer_order, pred_hybridization, pred_formal_charge, pred_num_connected_hydrogens, pred_possible_degree, pred_is_aromatic = model(data)

        print('Sum of Partial Charges: ', pred_partial_charges.sum().item())
        # Write output into a prody AtomGroup
        output_ligand.setCoords(all_coords.cpu().numpy())
        output_ligand.setNames(atoms_list) # type: ignore
        output_ligand.setResnames(['LIG'] * len(atoms_list)) # type: ignore
        output_ligand.setResnums(np.zeros(len(atoms_list))) # type: ignore
        output_ligand.setChids(['X'] * len(atoms_list)) # type: ignore
        output_ligand.setOccupancies(np.ones(len(atoms_list))) # type: ignore

        # You could predict other features as well...
        # output_ligand.setBetas([POSSIBLE_FORMAL_CHARGE_LIST[x] for x in pred_formal_charge.argmax(dim=-1)]) # type: ignore
        output_ligand.setBetas(pred_partial_charges.flatten().cpu().numpy()) # type: ignore

    pr.writePDB(output_path, output_ligand)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    default_weights_path = str(CURR_FILE_DIR_PATH / 'model_weights/pretrained_ligand_encoder_weights.pt')

    parser = argparse.ArgumentParser(description='Run Ligand Encoder inference on a PDB file containing only ligand atoms. Writes predicted partial charges to b-factors.')
    parser.add_argument('input_pdb', type=str, help='Path to input PDB file containing ligand coordinates.') 
    parser.add_argument('output_pdb', type=str, help='Output PDB path with predicted partial charges as b-factors.')
    parser.add_argument('--model_weights', '-w', type=str, help='Path to ligand encoder weights.', default=default_weights_path)
    output = vars(parser.parse_args())

    checkpoint_path, inference_molecule_path, output_path = output['model_weights'], output['input_pdb'], output['output_pdb']

    if not Path(inference_molecule_path).exists():
        raise FileNotFoundError(f'{inference_molecule_path} does not exist.')

    main(checkpoint_path, inference_molecule_path, output_path)