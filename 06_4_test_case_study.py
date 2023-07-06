import argparse
import os
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from dataloader import collate_fn
from functools import partial
from torch.utils.data import DataLoader
import json
from model import MdDTI
from utils import get_positions, get_all_adjs
from rdkit import Chem
from torch.utils.data import Dataset

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

l_drug_dict = len(pickle.load(open('./case_study/data/atom_r_dict', 'rb')))
l_ICMF_dict = len(pickle.load(open('./case_study/data/icmf_dict', 'rb')))
l_target_dict = len(pickle.load(open('./case_study/data/target_dict', 'rb')))

parser = argparse.ArgumentParser()
parser.add_argument('--substructure', default='ESPF', help='Select the drug 2D substructure. (ICMF or ESPF)')
parser.add_argument('--device', default='cuda:0', help='Disables CUDA training.')
parser.add_argument('--batch-size', type=int, default=16, help='Number of batch_size')
parser.add_argument('--max-target', type=int, default=1000, help='The maximum length of the target.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr-step-size', type=int, default=20, help='Period of learning rate decay.')
parser.add_argument('--lr-gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay.')
parser.add_argument('--count-decline', default=30, help='Early stoping.')
parser.add_argument('--task', default='CPA', help='Train CPA datasets.')
parser.add_argument('--drug-emb-dim', type=int, default=64, help='The embedding dimension of drug')
parser.add_argument('--target-emb-dim', type=int, default=64, help='The embedding dimension of target')
parser.add_argument('--epochs', type=int, default=110, help='Number of epochs to train.')
parser.add_argument('--p', type=int, default=10, help='The scale factor of coordinates.')

args = parser.parse_args()

if args.device == 'cuda:0':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("cuda is not available!!!")
        device = torch.device('cpu')
else:
    device = torch.device('cpu')
print('>>>>>> The code run on the ', device)
criterion = nn.MSELoss()


def read_file_by_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    f.close()
    return data


class Case_Study_Dataset(Dataset):
    def __init__(self, train_data, args):
        self.args = args

        if args.substructure.upper() == 'ICMF':
            self.train_substructures = read_file_by_json('./case_study/data/substructure_icmf_id.txt')
        else:
            self.train_substructures = read_file_by_json('./case_study/data/substructure_espf_id.txt')
        self.train_residues = read_file_by_json('./case_study/data/residues_id.txt')
        self.train_skeletons = read_file_by_json('./case_study/data/skeletons_id.txt')
        self.train_mols = list(Chem.SDMolSupplier('./case_study/data/all_mols.sdf'))[-96:]
        self.train_pos_idx, self.train_positions = get_positions(self.train_mols)
        self.train_adjs = get_all_adjs(self.train_mols)
        self.train_atoms_idx = read_file_by_json('./case_study/data/atoms_idx.txt')
        self.train_marks = read_file_by_json('./case_study/data/marks.txt')
        self.train_labels = train_data[2]

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


def test(model, data_loader):
    model.eval()
    pres = []
    with torch.no_grad():
        for target, drug, labels in tqdm(data_loader, colour='blue'):
            out, loss2 = model(drug, target)
            pres = pres + out.cpu().detach().numpy().reshape(-1).tolist()

    cs_pres = []
    for i in pres[:87]:
        cs_pres.append(1/(10 ** i))

    cs_pres = [i * 1000000000.0 for i in cs_pres] # change to nmol/L

    sorted_pres = sorted(cs_pres)
    print(sorted_pres)

    for i,p in enumerate(sorted_pres):
        j = cs_pres.index(p)
        print("Index %d---> Top %d: Ki:%f"%(j, i+1, cs_pres[j]))

def run(model, test_loader):
    if args.substructure.upper() == 'ICMF':
        # model_path = './case_study/save_model/Ki_ICMF_valid_best_checkpoint91.pth'
        model_path = './case_study/save_model/Kd_ICMF_valid_best_checkpoint28.pth'
    else:
        # model_path = './case_study/save_model/Ki_ESPF_valid_best_checkpoint90.pth'
        model_path = './case_study/save_model/Kd_ESPF_valid_best_checkpoint22.pth'
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print(">>>Load successful")
    else:
        print("Model does not exist")
    test(model, test_loader)


def load_test_data_CS():
    drugs_test, targets_test, labels_test = [], [], []
    fpath_test = './case_study/DTI.txt'
    with open(fpath_test, 'r') as f:
        test_data = f.read().strip().split('\n')
    f.close()

    print("Loading test dataset...")
    for item in tqdm(test_data):
        data = str(item).split(' ')
        drugs_test.append(data[0])
        targets_test.append(data[1])
        labels_test.append(float(data[2]))
    return [drugs_test, targets_test, labels_test]


if __name__ == '__main__':
    test_data = load_test_data_CS()
    if args.substructure.upper() == 'ICMF':
        l_substructures_dict = l_ICMF_dict
    else:
        l_substructures_dict = 23533

    my_collate_fn = partial(collate_fn, device=args.device)
    test_iter = Case_Study_Dataset(test_data, args)
    test_loader = DataLoader(test_iter, batch_size=args.batch_size, collate_fn=my_collate_fn, drop_last=True)

    model = MdDTI(l_drug_dict, l_target_dict, l_substructures_dict, args)
    model.to(args.device)

    run(model, test_loader)
