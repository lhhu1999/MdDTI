from tqdm import tqdm
from rdkit import Chem, RDLogger
from collections import defaultdict
import numpy as np
import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
substructure_dict = defaultdict(lambda: len(substructure_dict))


def create_atoms(mol):     # 把不同类型原子改为数字序列替代
    atoms_ids = [atom.GetIdx() for atom in mol.GetAtoms()]
    atom = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in atoms_ids]
    mark = [0] * len(atom)                # 记录非氢原子位置
    j = 0
    for c in atom:
        if c != 'H' and c != 'h':
            mark[j] = 1
        j = j + 1
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atom[i] = (atom[i], 'aromatic')
    atom = [atom_dict[a] for a in atom]
    if sum(mark) == 0:
        mark = [1 for i in mark]
    return np.array(atom), atoms_ids, mark


def create_ijbonddict(mol):     # 生成每个点其对应点和化学键类型数字(j,bond)组合的字典 eg:{0:[(1,0),(35,0),(46,0)],1:[(3,0),[5,0]],2:[..]}
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))

    atoms_set = set(range(mol.GetNumAtoms()))       # 孤立原子边设为nan
    isolate_atoms = atoms_set - set(i_jbond_dict.keys())
    bond = bond_dict['nan']
    for a in isolate_atoms:
        i_jbond_dict[a].append((a, bond))
    return i_jbond_dict


def atom_features(atoms, i_jbond_dict, r=2):   # return the subgraph of each atom
    if len(atoms) == 1:
        substructures = [substructure_dict[a] + 1 for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        substructures = []
        for _ in range(r):
            substructures = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                substructure = (nodes[i], tuple(sorted(neighbors)))
                substructures.append(substructure_dict[substructure] + 1)

            nodes = substructures
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict
    return substructures


if __name__ == '__main__':
    all_atoms_idx_h = []
    all_marks_h = []
    all_skeletons = []

    print("Loading mols file (about 2 minutes)...")
    mols = list(Chem.SDMolSupplier('./common/all_mols.sdf'))

    for m in tqdm(mols):
        mol = Chem.AddHs(m)
        atoms, atoms_idx, marks = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        all_atoms_idx_h.append(atoms_idx)
        all_marks_h.append(marks)
        all_skeletons.append(atom_features(atoms, i_jbond_dict))  # 返回每个点r范围内子图类型组成的序列 eg:[24,25,26,27,...]

    os.makedirs('./common/dict', exist_ok=True)

    with open('./common/skeletons_id.txt', 'w') as f:
        json.dump(all_skeletons, f)
        f.close()

    with open('./common/atoms_idx.txt', 'w') as f:
        json.dump(all_atoms_idx_h, f)
        f.close()

    with open('./common/marks.txt', 'w') as f:
        json.dump(all_marks_h, f)
        f.close()

    with open('./common/dict/atom_dict', 'wb') as f:       # 保存字典
        pickle.dump(dict(atom_dict), f)
        f.close()
    with open('./common/dict/bond_dict', 'wb') as f:
        pickle.dump(dict(bond_dict), f)
        f.close()
    with open('./common/dict/edge_dict', 'wb') as f:
        pickle.dump(dict(edge_dict), f)
        f.close()
    with open('./common/dict/atom_r_dict', 'wb') as f:
        pickle.dump(dict(substructure_dict), f)
        f.close()

    print("Successfully encoding drug 3D... ")
