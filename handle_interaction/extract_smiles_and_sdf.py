import os
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # 'human', 'celegans', 'Davis', 'KIBA'
    datasets = ['human', 'celegans', 'Davis', 'KIBA']

    for dataset in datasets:
        os.makedirs('../datasets/interaction/' + dataset, exist_ok=True)

        smiles = []
        mols = []
        filename = "../RawData/interaction/{}/data_shuffle.txt".format(dataset)
        with open(filename, 'r') as f:
            data = f.read().strip().split('\n')
            f.close()

        for item in tqdm(data):
            if dataset == 'Davis' or dataset == 'KIBA':
                smile = Chem.MolToSmiles(Chem.MolFromSmiles(str(item).split(' ')[2]))  # 统一格式
            else:
                smile = Chem.MolToSmiles(Chem.MolFromSmiles(str(item).split(' ')[0]))
            if smile not in smiles:
                smiles.append(smile)
                mol = AllChem.AddHs(Chem.MolFromSmiles(smile))
                AllChem.EmbedMolecule(mol, randomSeed=1234)
                mols.append(mol)
        with open('../datasets/interaction/'+ dataset +'/smiles.txt', 'w') as f:
            for smile in smiles:
                f.write(str(smile) + '\n')
            f.close()

        # the mols.sdf contains atomic space coordinates
        w = Chem.SDWriter('../datasets/interaction/' + dataset + '/mols.sdf')
        for m in mols:
            w.write(m)

        print("extract succeeded in " + dataset)
    print("All succeeded !!!")
