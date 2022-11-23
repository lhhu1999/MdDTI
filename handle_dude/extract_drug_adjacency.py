import numpy as np
from tqdm import tqdm
from rdkit import Chem
import warnings
warnings.filterwarnings("ignore")


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    adjacency = np.array(adjacency)
    adjacency += np.eye(adjacency.shape[0], dtype=int)    # change the diagonal to 1
    return adjacency


def save_adjacency(smiles, data):
    adjacency = []
    for smile in smiles:
        mol = Chem.AddHs(Chem.MolFromSmiles(smile))
        adjacency.append(create_adjacency(mol))

    filename_output = "../datasets/interaction/{}/adjacency".format(data)
    np.save(filename_output, adjacency)


if __name__ == '__main__':
    datasets = ['dude_1_1', 'dude_1_3', 'dude_1_5']

    for dataset in datasets:
        filename = "../datasets/interaction/{}/smiles.txt".format(dataset)
        with open(filename, "r") as f:
            data_list = f.read().strip().split('\n')

        smiles = []
        for j in tqdm(data_list):
            smile = j.strip().split(' ')[0]
            smiles.append(smile)

        save_adjacency(smiles, dataset)
        print("extract succeeded in " + dataset)
    print("All succeeded !!!")
