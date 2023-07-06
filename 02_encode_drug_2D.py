from tqdm import tqdm
import pandas as pd
import pickle
import copy
import codecs
import os
import json
from subword_nmt.apply_bpe import BPE
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

drug_vocab_path = './RawData/ESPF/drug_codes_chembl.txt'  #已经得到的药物词表
ESPF_codes_drug = codecs.open(drug_vocab_path)
drugESPF = BPE(ESPF_codes_drug, merges=-1, separator='')
sub_csv_d = pd.read_csv('./RawData/ESPF/subword_units_map_chembl.csv') #用于匹配分后的词用索引替代
idx2word_d = sub_csv_d['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

icmf_dict = defaultdict(lambda: len(icmf_dict))


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


if __name__ == '__main__':
    all_substructure_espf = []
    all_substructure_icmf = []

    with open('./common/all_smiles_new.txt', 'r') as f:
        smiles_no_h = f.read().strip().split('\n')
        f.close()

    for smiles in tqdm(smiles_no_h):
        # Step 1: ESPF
        all_substructure_espf.append(drug_espf_split(smiles))

        # Step 2: ICMF
        substructure_icmf_smiles = get_substructure_icmf(smiles)
        all_substructure_icmf.append([icmf_dict[item] + 1 for item in substructure_icmf_smiles])

    with open('./common/substructure_espf_id.txt', 'w') as f:
        json.dump(all_substructure_espf, f)
    f.close()

    with open('./common/substructure_icmf_id.txt', 'w') as f:
        json.dump(all_substructure_icmf, f)
    f.close()

    os.makedirs('./common/dict', exist_ok=True)

    with open('./common/dict/icmf_dict', 'wb') as f:
        pickle.dump(dict(icmf_dict), f)
        f.close()

    print("Successfully encoding drug 2D (espf and icmf)... ")
