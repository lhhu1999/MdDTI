from rdkit import Chem
import os
import numpy as np
import json
import pandas as pd
import pickle
import codecs
from rdkit.Chem import BRICS
from collections import defaultdict
import copy
from rdkit.Chem import AllChem
from subword_nmt.apply_bpe import BPE
import warnings
warnings.filterwarnings("ignore")

atom_dict1 = pickle.load(open('../common/dict/atom_dict', 'rb'))
bond_dict1 = pickle.load(open('../common/dict/bond_dict', 'rb'))
edge_dict1 = pickle.load(open('../common/dict/edge_dict', 'rb'))
atom_r_dict1 = pickle.load(open('../common/dict/atom_r_dict', 'rb'))
icmf_dict1 = pickle.load(open('../common/dict/icmf_dict', 'rb'))
target_dict1 = pickle.load(open('../common/dict/target_dict', 'rb'))

drug_vocab_path = '../RawData/ESPF/drug_codes_chembl.txt'  #已经得到的药物词表
ESPF_codes_drug = codecs.open(drug_vocab_path)
drugESPF = BPE(ESPF_codes_drug, merges=-1, separator='')
sub_csv_d = pd.read_csv('../RawData/ESPF/subword_units_map_chembl.csv') #用于匹配分后的词用索引替代
idx2word_d = sub_csv_d['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

#####################################################
atom_dict = atom_dict1
atom_dict = defaultdict(lambda: len(atom_dict), atom_dict)
bond_dict = bond_dict1
bond_dict = defaultdict(lambda: len(bond_dict), bond_dict)
edge_dict = edge_dict1
edge_dict = defaultdict(lambda: len(edge_dict), edge_dict)
atom_r_dict = atom_r_dict1
atom_r_dict = defaultdict(lambda: len(atom_r_dict), atom_r_dict)
icmf_dict = icmf_dict1
icmf_dict = defaultdict(lambda: len(icmf_dict), icmf_dict)
target_dict = target_dict1
target_dict = defaultdict(lambda: len(target_dict), target_dict)

def drug_espf_split(drug):  #得到药物子序列表示，eg:[7,23,643,...,343,0,0,0..0] [1,1,1,...,1,0,0,0..0]
    d = drugESPF.process_line(drug).split()  # 根据已有的词表划分smiles
    s = [words2idx_d.get(i, 23532) + 1 for i in d]
    return s


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def brics_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]]

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) < 0:
        return [list(range(n_atoms))]
    else:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # break bonds between rings and non-ring atoms
    for c in cliques:
        if len(c) > 1:
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                cliques.remove(c)
                cliques.append([c[1]])
                breaks.append(c)
            if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                cliques.remove(c)
                cliques.append([c[0]])
                breaks.append(c)

    # select atoms at intersections as motif
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
            has_no_single = False
            for nei in atom.GetNeighbors():
                if mol.GetBondBetweenAtoms(nei.GetIdx(), atom.GetIdx()).GetBondType().name != 'SINGLE':
                    has_no_single = True
                    break
            if has_no_single is False:
                continue

            for nei in atom.GetNeighbors():
                if mol.GetBondBetweenAtoms(nei.GetIdx(), atom.GetIdx()).GetBondType().name == 'SINGLE':
                    if [nei.GetIdx(), atom.GetIdx()] in cliques:
                        cliques.remove([nei.GetIdx(), atom.GetIdx()])
                        breaks.append([nei.GetIdx(), atom.GetIdx()])
                    elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                        cliques.remove([atom.GetIdx(), nei.GetIdx()])
                        breaks.append([atom.GetIdx(), nei.GetIdx()])
                    cliques.append([nei.GetIdx()])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if len(c) > 0]
    return cliques


def sanitize(mol):
    try:
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


def get_clique_mol(mol, atoms):
    try:
        smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    except:
        return None
    else:
        new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        new_mol = copy_edit_mol(new_mol).GetMol()
        new_mol = sanitize(new_mol)  # We assume this is not None
        return new_mol


def get_PO4_and_PO3(mol):
    PO4 = Chem.MolFromSmarts('OP(=O)(O)O')
    PO3 = Chem.MolFromSmarts('P(=O)(O)O')
    res_PO4 = list(mol.GetSubstructMatches(PO4))
    res_PO3 = list(mol.GetSubstructMatches(PO3))
    len_PO4 = len(res_PO4)
    len_PO3 = len(res_PO3)
    if len_PO4 == len_PO3:
        return res_PO4
    elif len_PO4 == 0:
        return res_PO3
    else:
        new_res_PO3 = []
        for res in res_PO3:
            for idx in res:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetAtomicNum() == 15:
                    neighbors = atom.GetNeighbors()
                    oxygens_count = sum([1 for nei in neighbors if nei.GetAtomicNum() == 8])
                    if oxygens_count == 3:
                        new_res_PO3.append(res)
                    break
        return new_res_PO3 + res_PO4


def get_F3_and_CN(mol):
    F3 = Chem.MolFromSmarts('*(F)(F)F')
    res_F3 = list(mol.GetSubstructMatches(F3))
    CN = Chem.MolFromSmarts('C#N')
    res_CN = list(mol.GetSubstructMatches(CN))
    return res_F3 + res_CN


def get_CO_and_COO(mol):
    COO = Chem.MolFromSmarts('C(=O)O')
    CO = Chem.MolFromSmarts('C(=O)')
    res_COO = list(mol.GetSubstructMatches(COO))
    res_CO = list(mol.GetSubstructMatches(CO))
    len_COO = len(res_COO)
    len_CO = len(res_CO)
    if len_COO == len_CO:
        return res_COO
    elif len_COO == 0:
        return res_CO
    else:
        new_res_CO = []
        for res in res_CO:
            for idx in res:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetAtomicNum() == 6:
                    neighbors = atom.GetNeighbors()
                    oxygens_count = sum([1 for nei in neighbors if nei.GetAtomicNum() == 8])
                    if oxygens_count == 1:
                        new_res_CO.append(res)
                    break
        return new_res_CO + res_COO


def get_SO2_and_NO2(mol):
    SO2 = Chem.MolFromSmarts('S(=O)(=O)')
    res_SO2 = list(mol.GetSubstructMatches(SO2))

    NO2_1 = Chem.MolFromSmarts('O=[N+]([O-])')
    NO2_2 = Chem.MolFromSmarts('[N+](=O)[O-]')
    res_NO2 = list(mol.GetSubstructMatches(NO2_1)) + list(mol.GetSubstructMatches(NO2_2))
    return res_SO2 + res_NO2


def get_func_grops(smiles):
    mol = Chem.MolFromSmiles(smiles)
    PO4_PO3 = get_PO4_and_PO3(mol)
    F3_CN = get_F3_and_CN(mol)
    CO_COO = get_CO_and_COO(mol)
    SO2_NO2 = get_SO2_and_NO2(mol)
    func_grops = PO4_PO3 + F3_CN + CO_COO + SO2_NO2
    func_grops = [list(fc) for fc in func_grops]
    return func_grops


def get_substructure_icmf(smiles):
    mol = get_mol(smiles)

    cliques = brics_decomp(mol)
    func_grops = get_func_grops(smiles)

    # Remove the small fragments contained in the functional groups.
    cliques_copy = copy.deepcopy(cliques)
    for atoms in cliques_copy:
        for func_grop in func_grops:
            flag = 0
            for atom in atoms:
                if atom not in func_grop:
                    flag = 1
                    break
            if flag == 0:
                cliques.remove(atoms)
                break

    cliques = cliques + func_grops
    sorted_cliques = sorted(cliques, key=lambda x: min(x))

    cliques_smiles = []
    for i, c in enumerate(sorted_cliques):
        cmol = get_clique_mol(mol, c)
        if cmol is None:
            continue
        else:
            cliques_smiles.append(Chem.MolToSmiles(cmol))
    if len(cliques_smiles) == 0:
        cliques_smiles.append(smiles)
    return cliques_smiles


def read_DTI(path):
    all_smiles = []
    all_targets = []
    with open(path, 'r') as f:
        data = f.read().strip().split('\n')
        f.close()
    for item in data:
        smiles = str(item).split(' ')[0]
        target = str(item).split(' ')[1]
        all_smiles.append(smiles)
        all_targets.append(target)
    return all_smiles, all_targets


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [target_dict[sequence[i:i + ngram]] + 1
             for i in range(len(sequence) - ngram + 1)]
    return words

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
        substructures = [atom_r_dict[a] + 1 for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        substructures = []
        for _ in range(r):
            substructures = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                substructure = (nodes[i], tuple(sorted(neighbors)))
                substructures.append(atom_r_dict[substructure] + 1)

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
    os.makedirs('./data', exist_ok=True)
    all_smiles, all_targets = read_DTI('DTI.txt')

    all_substructure_espf = []
    all_substructure_icmf = []
    all_residues = []
    all_mols = []
    all_atoms_idx_h = []
    all_marks_h = []
    all_skeletons = []
    for i in range(len(all_smiles)):
        print(i)
        try:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(all_smiles[i]))  # 统一格式
            target = all_targets[i]

            # Step 1: ESPF
            all_substructure_espf.append(drug_espf_split(smiles))

            # Step 2: ICMF
            substructure_icmf_smiles = get_substructure_icmf(smiles)
            all_substructure_icmf.append([icmf_dict[item] + 1 for item in substructure_icmf_smiles])

            # Step 3: encoding target sequences
            residues = split_sequence(target, 3)
            all_residues.append(residues)

            # Step 4
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            AllChem.EmbedMolecule(mol, randomSeed=1234)
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except ValueError:
                AllChem.EmbedMultipleConfs(mol, numConfs=3)
                pass
            all_mols.append(Chem.RemoveHs(mol))

            mol = Chem.AddHs(mol)
            atoms, atoms_idx, marks = create_atoms(mol)
            i_jbond_dict = create_ijbonddict(mol)

            all_atoms_idx_h.append(atoms_idx)
            all_marks_h.append(marks)
            all_skeletons.append(atom_features(atoms, i_jbond_dict))  # 返回每个点r范围内子图类型组成的序列 eg:[24,25,26,27,...]

            print(i)
        except KeyError:
            pass

    w = Chem.SDWriter('./data/all_mols.sdf')
    for m in all_mols:
        w.write(m)

    with open('./data/substructure_espf_id.txt', 'w') as f:
        json.dump(all_substructure_espf, f)
    f.close()

    with open('./data/substructure_icmf_id.txt', 'w') as f:
        json.dump(all_substructure_icmf, f)
    f.close()

    with open('./data/residues_id.txt', 'w') as f:
        json.dump(all_residues, f)
    f.close()

    with open('./data/skeletons_id.txt', 'w') as f:
        json.dump(all_skeletons, f)
        f.close()

    with open('./data/atoms_idx.txt', 'w') as f:
        json.dump(all_atoms_idx_h, f)
        f.close()

    with open('./data/marks.txt', 'w') as f:
        json.dump(all_marks_h, f)
        f.close()

    with open('./data/icmf_dict', 'wb') as f:
        pickle.dump(dict(icmf_dict), f)
        f.close()

    with open('./data/atom_r_dict', 'wb') as f:
        pickle.dump(dict(atom_r_dict), f)
        f.close()

    with open('./data/target_dict', 'wb') as f:       # 保存子图字典
        pickle.dump(dict(target_dict), f)
        f.close()
