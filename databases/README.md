### Training Databases.

- Download by running `download_ligand_encoder_training_dataset.sh` and `download_protonated_pdb_trianing_dataset.sh` in the root directory.

- `./spice_dataset` contains SPICE dataset for ligand encoder pretraining.

- `./pdb_dataset` contains our protonated processed PDB database stored in a python [shelve](https://docs.python.org/3/library/shelve.html) object. See `./utils/pdb_dataset.py` for how to work with the data. 


### Train / Validation Split Info:

> [!IMPORTANT]
> The chain and segment IDs have been remapped in the preprocessed LASErMPNN training dataset sequentially starting with A, B, C... to ensure large proteins in the dataset could be passed through Reduce.
> The full training dataset with remapped chains in PDB format [is uploaded to Zenodo here](www.google.com).

The LASErMPNN train/test splits are stored in `json` files zipped in `./dataset_split_info.zip`.

There are 2 different splits reported:

- One is our reconstruction of the LigandMPNN training data, we did not reproduce the validation/test data since the model was already hyperparameter tuned.
We provide two `.json` files corresponding to the coarse 30% sequence identity clusters and finer 70% sequence identity subclusters which were each sampled once per cluster/subcluster respectively during the training process.

- The second split is our streptavidin held-out split where all streptavidin and related proteins were placed in the validation set such that the train set should be free of contamination. We provide four `.json` files corresponding to the train/validation data for both the 30% sequence identity clusters and 70% sequence identity subclusters for each set. We don't report a validation set since the trained models were validated with de novo design.
