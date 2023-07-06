import numpy as np
from torch.utils.data import Dataset
from utils import get_train_data, get_positions, get_all_adjs, to_float_tensor, to_long_tensor, positions_match, batch_pad
from torch.nn.utils.rnn import pad_sequence


class MyDataset(Dataset):
    def __init__(self, common_data, train_data, args):
        self.args = args
        self.all_smiles_idx = common_data[0]
        self.all_smiles_new = common_data[1]
        self.all_targets_idx = common_data[2]
        self.all_residues = common_data[3]
        self.all_substructures = common_data[4]
        self.all_skeletons = common_data[5]
        self.all_mols = common_data[6]
        self.all_atoms_idx = common_data[7]
        self.all_marks = common_data[8]
        self.all_pos_idx, self.all_positions = get_positions(self.all_mols)
        self.all_adjs = get_all_adjs(self.all_mols)

        self.train_drugs = train_data[0]
        self.train_targets = train_data[1]
        self.train_labels = train_data[2]

        # target data
        self.train_residues = get_train_data(self.all_residues, self.all_targets_idx, self.train_targets)
        # drug data
        self.train_substructures = get_train_data(self.all_substructures, self.all_smiles_idx, self.train_drugs)
        self.train_skeletons = get_train_data(self.all_skeletons, self.all_smiles_idx, self.train_drugs)
        self.train_atoms_idx = get_train_data(self.all_atoms_idx, self.all_smiles_idx, self.train_drugs)
        self.train_marks = get_train_data(self.all_marks, self.all_smiles_idx, self.train_drugs)
        self.train_pos_idx = get_train_data(self.all_pos_idx, self.all_smiles_idx, self.train_drugs)
        self.train_positions = get_train_data(self.all_positions, self.all_smiles_idx, self.train_drugs)
        self.train_adjs = get_train_data(self.all_adjs, self.all_smiles_idx, self.train_drugs)

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, item):
        residues_f = np.array(self.train_residues[item][:self.args.max_target])
        substructures_f = np.array(self.train_substructures[item])
        skeletons_f = np.array(self.train_skeletons[item])
        atoms_idx = self.train_atoms_idx[item]
        marks = self.train_marks[item]
        pos_idx = self.train_pos_idx[item]
        positions = self.train_positions[item]
        adjs = self.train_adjs[item]
        label = self.train_labels[item]
        return residues_f, substructures_f, skeletons_f, atoms_idx, marks, pos_idx, positions, adjs, label


def collate_fn(batch, device):
    drug, target = {}, {}

    batch_residues = []
    batch_substructures, batch_skeletons, batch_atoms_idx, batch_marks = [], [], [], []
    batch_positions, batch_adjs, len_adjs = [], [], []
    batch_labels = []

    len_residues, len_substructures, len_skeletons = [], [], []

    for item in batch:
        batch_residues.append(to_long_tensor(item[0]))
        len_residues.append(len(item[0]))
        batch_substructures.append(to_long_tensor(item[1]))
        len_substructures.append(len(item[1]))
        batch_skeletons.append(to_long_tensor(item[2]))
        len_skeletons.append(len(item[2]))
        batch_atoms_idx.append(item[3])
        batch_marks.append(to_long_tensor(item[4]))
        positions_new = positions_match(item[3], item[4], item[5], item[6])
        batch_positions.append(to_float_tensor(positions_new))
        len_adjs.append(len(item[7]))
        batch_adjs.append(item[7])

        batch_labels.append(item[8])

    min_skeleton_idx = len_skeletons.index(min(len_skeletons))

    batch_adjs = to_long_tensor(batch_pad(batch_adjs, max(len_adjs))).to(device)

    batch_residues = pad_sequence(batch_residues, batch_first=True, padding_value=0).to(device)
    batch_substructures = pad_sequence(batch_substructures, batch_first=True, padding_value=0).to(device)
    batch_skeletons = pad_sequence(batch_skeletons, batch_first=True, padding_value=0).to(device)
    batch_positions = pad_sequence(batch_positions).permute(1, 0, 2).to(device)

    batch_residues_masks = to_long_tensor(np.zeros((batch_residues.shape[0], batch_residues.shape[1]))).to(device)
    batch_substructures_masks = to_long_tensor(np.zeros((batch_substructures.shape[0], batch_substructures.shape[1]))).to(device)
    for i in range(len(len_residues)):
        batch_residues_masks[i, :len_residues[i]] = 1
        batch_substructures_masks[i, :len_substructures[i]] = 1

    target['residues'] = batch_residues
    target['residues_masks'] = batch_residues_masks
    drug['substructures'] = batch_substructures
    drug['substructures_masks'] = batch_substructures_masks
    drug['skeletons'] = batch_skeletons
    drug['marks'] = batch_marks
    drug['positions'] = batch_positions
    drug['adjs'] = batch_adjs
    drug['padding_tensor_idx'] = min_skeleton_idx

    return target, drug, batch_labels
