from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')


if __name__ == '__main__':
    all_smiles_new = []
    all_mols = []

    # Step 1:
    with open('./common/all_smiles.txt', "r") as f:
        all_smiles = f.read().strip().split('\n')
    f.close()

    with open('./common/all_smiles_new.txt', 'w') as f:
        for smiles in tqdm(all_smiles):
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            all_smiles_new.append(smiles)
            f.write(str(smiles) + '\n')
        f.close()
    print("Successfully extracting SMILES... ")

    # Step 2:
    for smiles in tqdm(all_smiles_new):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=1234)
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except ValueError:
            AllChem.EmbedMultipleConfs(mol, numConfs=3)
            pass
        all_mols.append(Chem.RemoveHs(mol))

    w1 = Chem.SDWriter('./common/all_mols.sdf')
    for m in all_mols:
        w1.write(m)

    print("Successfully extracting Coordinates... ")
