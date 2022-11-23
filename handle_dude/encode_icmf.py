import rdkit.Chem as Chem
from rdkit.Chem import BRICS
from collections import defaultdict
import numpy as np
import os
from tqdm import tqdm
import pickle

cliques_dict = defaultdict(lambda: len(cliques_dict))


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def tree_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])

    # get rings
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Merge Rings with intersection > 2 atoms
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2: continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2: continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []

    cliques = [c for c in cliques if len(c) > 0]
    return cliques


def brics_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) == 0:
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
            flag = 0
            for nei in atom.GetNeighbors():
                if mol.GetBondBetweenAtoms(nei.GetIdx(),
                                           atom.GetIdx()).GetBondType().name == 'SINGLE' and nei.GetIsAromatic() is False:
                    if [nei.GetIdx(), atom.GetIdx()] in cliques:
                        cliques.remove([nei.GetIdx(), atom.GetIdx()])
                        breaks.append([nei.GetIdx(), atom.GetIdx()])
                    elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                        cliques.remove([atom.GetIdx(), nei.GetIdx()])
                        breaks.append([atom.GetIdx(), nei.GetIdx()])
                    cliques.append([nei.GetIdx()])
                else:
                    flag = 1
            if flag == 0:
                cliques.append([atom.GetIdx()])

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


def get_cliques(smiles):
    mol = get_mol(smiles)

    cliques = brics_decomp(mol)
    if len(cliques) <= 1:
        cliques = tree_decomp(mol)

    cliques_smiles = []
    for i, c in enumerate(cliques):
        cmol = get_clique_mol(mol, c)
        if cmol is None:
            continue
        else:
            cliques_smiles.append(Chem.MolToSmiles(cmol))
    if len(cliques_smiles) == 0:
        cliques_smiles.append(smiles)
    return cliques_smiles


if __name__ == "__main__":
    components_cmf = []
    path_output = "../datasets/interaction/dude_1_5"
    with open("../datasets/interaction/dude_1_5/smiles.txt", "r") as f:
        for line in tqdm(f.readlines()):
            smiles = line.strip('\n')
            cliques = get_cliques(smiles)
            components_cmf.append(np.array([cliques_dict[item] for item in cliques]))
    np.save(os.path.join(path_output, 'components_cmf'), components_cmf)

    with open('../datasets/interaction/dude_1_5/drugs_cmf_dict', 'wb') as f:  # 保存子图字典
        pickle.dump(dict(cliques_dict), f)
