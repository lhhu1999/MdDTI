from rdkit import Chem
import numpy as np
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    datasets = ['Kd', 'EC50']
    for dataset in datasets:
        positions = []
        filename = "../datasets/affinity/{}/mols.sdf".format(dataset)
        mols = Chem.SDMolSupplier(filename)
        for mol in mols:
            position = np.array(mol.GetConformer().GetPositions())
            positions.append(position)
        filename_output = "../datasets/affinity/{}/positions".format(dataset)
        np.save(filename_output, positions)
        print("extract succeeded in " + dataset)
    print("All succeeded !!!")
