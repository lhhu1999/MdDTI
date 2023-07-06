import json
import sys
from rdkit import Chem
import numpy as np
from tqdm import tqdm
import torch


def read_file(path):
    with open(path, 'r') as f:
        data = f.read().strip().split('\n')
    f.close()
    return data


def read_file_by_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    f.close()
    return data


def load_common_data(subs):
    print("Loading common dataset (about 2 minutes)...")
    # load drug and targets
    all_smiles = read_file('./common/all_smiles.txt')
    all_smiles_new = read_file('./common/all_smiles_new.txt')
    all_targets = read_file('./common/all_target.txt')

    # The features of the target.
    residues = read_file_by_json('./common/residues_id.txt')

    # The features of the drug 2D substructure.
    subs = subs.upper()
    if subs == 'ICMF':
        substructures = read_file_by_json('./common/substructure_icmf_id.txt')
    elif subs == 'ESPF':
        substructures = read_file_by_json('./common/substructure_espf_id.txt')
    else:
        print('The parameter of drug 2D substructure is wrong!!! (Please choose for ICMF and ESFP)')
        sys.exit(1)

    # The features of the drug 3D.
    skeletons = read_file_by_json('./common/skeletons_id.txt')
    mols = list(Chem.SDMolSupplier('./common/all_mols.sdf'))
    atoms_idx = read_file_by_json('./common/atoms_idx.txt')
    marks = read_file_by_json('./common/marks.txt')
    return [all_smiles, all_smiles_new, all_targets, residues, substructures, skeletons, mols, atoms_idx, marks]


def shuffle_dataset(data):
    np.random.seed(1234)
    np.random.shuffle(data)
    return data


def load_train_data_DTI(dataset):
    drugs, targets, labels = [], [], []
    fpath = './RawData/interaction/{}.txt'.format(dataset)
    train_data = shuffle_dataset(read_file(fpath))

    print("Loading train dataset...")
    if dataset == 'human' or dataset == 'celegans':
        for item in tqdm(train_data):
            data = str(item).split(' ')
            drugs.append(data[0])
            targets.append(data[1])
            labels.append(int(data[2]))
    elif dataset == 'Davis' or dataset == 'KIBA':
        for item in tqdm(train_data):
            data = str(item).split(' ')
            drugs.append(data[2])
            targets.append(data[3])
            labels.append(int(data[4]))
    return [drugs, targets, labels]


def load_train_data_CPA(dataset):
    drugs_train, targets_train, labels_train = [], [], []
    drugs_test, targets_test, labels_test = [], [], []
    fpath_train = './RawData/affinity/{}/train.txt'.format(dataset)
    fpath_test = './RawData/affinity/{}/test.txt'.format(dataset)
    train_data = shuffle_dataset(read_file(fpath_train))
    test_data = shuffle_dataset(read_file(fpath_test))

    print("Loading train dataset...")
    for item in tqdm(train_data):
        data = str(item).split(',')
        drugs_train.append(data[0])
        targets_train.append(data[1])
        labels_train.append(float(data[2]))

    print("Loading test dataset...")
    for item in tqdm(test_data):
        data = str(item).split(',')
        drugs_test.append(data[0])
        targets_test.append(data[1])
        labels_test.append(float(data[2]))
    return [drugs_train, targets_train, labels_train], [drugs_test, targets_test, labels_test]


def get_train_data(all_data, all_data_ids, train_data_ids):
    all_data_dict = {value: idx for idx, value in enumerate(all_data_ids)}

    train_data = []
    for v in train_data_ids:
        idx = all_data_dict[v]
        train_data.append(all_data[idx])
    return train_data


def get_positions(mols):
    all_pos_idx = []
    all_positions = []
    print("Loading positions...")
    for mol in tqdm(mols):
        conformer = mol.GetConformer()

        # extract atomic space coordinates
        atoms = mol.GetAtoms()
        atoms_idx = [atom.GetIdx() for atom in atoms]
        positions = [conformer.GetAtomPosition(idx) for idx in atoms_idx]
        all_pos_idx.append(atoms_idx)
        all_positions.append(positions)
    return all_pos_idx, all_positions


def get_edge_index(mols):
    edge_index = []
    print("Loading edge index...")
    for mol in tqdm(mols):
        bonds = mol.GetBonds()

        e1, e2 = [], []
        for bond in bonds:
            e1.append(bond.GetBeginAtomIdx())
            e1.append(bond.GetEndAtomIdx())
            e2.append(bond.GetEndAtomIdx())
            e2.append(bond.GetBeginAtomIdx())
        edge_index.append([e1,e2])
    return edge_index


def get_all_adjs(mols):
    adjs = []
    print("Loading adjacency...")
    for mol in tqdm(mols):
        mol = Chem.AddHs(mol)
        adjs.append(Chem.GetAdjacencyMatrix(mol))
    return adjs


def batch_pad(datas, N):
    data = np.zeros((len(datas), N, N))
    for i, a in enumerate(datas):
        n = a.shape[0]
        data[i, :n, :n] = a
    return data


def to_float_tensor(data):
    return torch.FloatTensor(data)


def to_long_tensor(data):
    return torch.LongTensor(data)


def positions_match(atoms_ix, marks, pos_idx, positions):
    pos_new = []
    for i, idx in enumerate(atoms_ix):
        if marks[i] == 1:
            pos_new.append(positions[pos_idx.index(idx)])
    return pos_new


def get_five_fold_idx(lens, fold_i, K_FOLD):
    idx = [id for id in range(lens)]

    fold_size = lens // K_FOLD
    val_start = fold_i * fold_size
    val_end = (fold_i + 1) * fold_size

    train_idx = idx[:val_start] + idx[val_end:]
    test_idx = idx[val_start:val_end]

    return train_idx, test_idx


def get_test_idx(lens, ratio, batchs):
    idx = np.arange(lens)
    num = int(lens/batchs * (1-ratio))
    num_train = num * batchs
    idx_train, idx_test = idx[:num_train], idx[num_train:]
    return idx_train, idx_test