### Training Databases.

- Download by running `download_ligand_encoder_training_dataset.sh` and `download_protonated_pdb_trianing_dataset.sh` in the root directory.

- `./spice_dataset` contains SPICE dataset for ligand encoder pretraining.

- `./pdb_dataset` contains our protonated processed PDB database stored in a python [shelve](https://docs.python.org/3/library/shelve.html) object. See `./utils/pdb_dataset.py` for how to work with the data. 


### Train / Test Splits:

> [!IMPORTANT]
> The chain and segment IDs have been remapped in the preprocessed LASErMPNN training dataset sequentially starting with A, B, C... to ensure large proteins in the dataset could be passed through Reduce.
> The full training dataset with remapped chains in PDB format is uploaded to Zenodo in two chunks [Chunk 1](https://zenodo.org/records/17990180) and [Chunk 2](https://zenodo.org/records/17990253) which can be reconstructed into a single `.zip` file following the instructions in the zenodo description which are copied here below:

```text
Chunk 1/2 of dataset of PDB files processed from the PDB, remapped chain and segment IDs correspond to those reported in the dataset split. Proton positions placed by REDUCE.

Chunk 1 is 10.5281/zenodo.17990180
Chunk 2 is 10.5281/zenodo.17990253

To reassemble the chunks into the full zip file, cat them together with the following command:

cat reduce_filtered_pdb_bioasmb_two_letter_bug_fixed.zip.part.aa \
    reduce_filtered_pdb_bioasmb_two_letter_bug_fixed.zip.part.ab \
  > reconstructed.zip
 
The output of md5sum reconstructed.zip should be c9418cb9368c8068a6053feebbff5fda
The cat command assembles the reconstructed.zip file which is around 50 GB in total.
The full uncompressed dataset is around 210 GB and can be produced by unzipping the reconstructed.zip file.
```



The LASErMPNN train/test splits are stored in `json` files zipped in `./dataset_split_info.zip`.

There are 2 different splits reported:

- One is our reconstruction of the LigandMPNN training data, we did not reproduce the validation/test data since the model was already hyperparameter tuned.
We provide two `.json` files corresponding to the coarse 30% sequence identity clusters and finer 70% sequence identity subclusters which were each sampled once per cluster/subcluster respectively during the training process.

- The second split is our streptavidin held-out split where all streptavidin and related proteins were placed in the validation set such that the train set should be free of contamination. We provide four `.json` files corresponding to the train/validation data for both the 30% sequence identity clusters and 70% sequence identity subclusters for each set. 

The 6w70 structures and monomeric streptavidin fold were used as a small validation set in addition to the de novo design of novel binders.
