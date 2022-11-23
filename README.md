# MdDTI
MdDTI: Multi-dimensional drug-target interaction prediction by preserving the consistency of attention distribution.

## Dependencies
Environments:
* Windows 10. 
* GPU: NVIDIA RTX 3090 Ti and NVIDIA Quadro P4000 (for KIBA dataset).

Dependencies:
* python 3.8
* pytorch >=1.2
* numpy
* sklearn
* tqdm
* rdkit
* subword-nmt 0.3.8

## Resources
* RawData:
    - interaction: The five interaction datasets used in the paper. Including the raw data and random shuffling data (random seed: 1234).
    - affinity: The two affinity datasets used in the paper. Including the raw data and random shuffling data (random seed: 1234).
* handle_interaction: Preprocessing the interaction datasets.
    - data_shaffle.py
    - extract_smiles_and_sdf.py
    - extract_drug_adjacency.py
    - extract_atomic_coordinate.py
    - encode.py: Drug and target encoding representation.
    - extract_icmf.py: Decompose drugs by ICMF method.
* handle_dude: Preprocessing the balanced and unbalanced datasets.
    - Reference handle_interaction.
* handle_affinity: Preprocessing the affinity datasets.
    - Reference handle_interaction
* datasets: Input data of the model.
* query_network: Target query network.
    - encoder_residues.py: Encoder module.
    - decoder_2D_skeletons.py: 2D Decoder module.
    - decoder_3D_skeletons.py: 3D Decoder module.
* config.py: Part of the hyperparameters.
* model.py: MdDTI model architecture.
* train_affinity.py: Train and test the model on the Kd and EC50 datasets.
* train_Davis_KIBA.py: Train and test the model on the Davis and KIBA datasets.
* train_interaction.py: Train and test the model on the HUMAN and C.ELEGANS datasets.
* train_unbalanced_datasets.py: Train and test the model on the dude(1:1), dude(1:3) and dude(1:5) datasets.

## Run
    for HUMAN and C.ELEGANS datasets:
        - run train_interaction.py
        (Note: There is a parameter 'methods_components' in train_interaction.py. 
            if methods_components == 'icmf': the model is MdDTI-ICMF.
            else: the model is MdDTI-BPE.)
    for KIBA and Davis datasets:
        - step 1: Unzip the file “RawData/interaction/KIBA/*.zip”.
        - step 2: run handle_interaction/extract_smiles_and_sdf.py
        - step 3: run handle_interaction/extract_drug_adjacency.py
        - step 4: run handle_interaction/extract_atomic_coordinate.py
        - step 5: run handle_interaction/encode.py: Drug and target encoding representation. (Note: Run once for each dataset.)
        - step 6: run handle_interaction/extract_icmf.py: Decompose drugs by ICMF method. (Note: Run once for each dataset.)
        - step 7: run train_Davis_KIBA.py
    for Kd and EC50 datasets:
        - step 1: run handle_affinity/extract_smiles_and_sdf.py
        - step 2: run handle_affinity/extract_drug_adjacency.py
        - step 3: run handle_affinity/extract_atomic_coordinate.py
        - step 4: run handle_affinity/encode.py: Drug and target encoding representation. (Note: Run once for each dataset.)
        - step 5: run handle_affinity/extract_icmf.py: Decompose drugs by ICMF method. (Note: Run once for each dataset.)
        - step 6: run train_affinity.py
    for dude(1:1), dude(1:3) and dude(1:5) datasets:
        - step 1: run handle_dude/extract_smiles_and_sdf.py
        - step 2: run handle_dude/extract_drug_adjacency.py
        - step 3: run handle_dude/extract_atomic_coordinate.py
        - step 4: run handle_dude/encode.py: Drug and target encoding representation. (Note: Run once for each dataset.)
        - step 5: run handle_dude/extract_icmf.py: Decompose drugs by ICMF method. (Note: Run once for each dataset.)
        - step 6: run train_unbalanced_datasets.py
