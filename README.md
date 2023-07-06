# MdDTI

Note that the input SMILES is unified using RDKit (by converting SMILES to a mol object using RDKit, and then converting the mol object back to unified SMILES). Make sure to use the same version of RDKit for both training and testing to avoid potential errors.

```python
smiles = 'C1=COC(=C1)C(=O)C(=O)O'
smiles_new = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
print(smiles_new)   # O=C(O)C(=O)c1ccco1
```

### 1. Preprocessing
$\qquad$ **Step 1:** 00_extract_all_smiles.py  
$\qquad$ **Step 2:** 01_create_smiles_and_sdf.py  
$\qquad$ **Step 3:** 02_encode_drug_2D.py  
$\qquad$ **Step 4:** 03_encode_drug_3D.py  
$\qquad$ **Step 5:** 04_encode_target.py  

### 2. Train
$\qquad$ **Step 1:** 05_1_train_Human_Celegans.py  
$\qquad$ **Step 2:** 05_2_train_Davis_KIBA.py  
$\qquad$ **Step 3:** 05_3_train_Kd_EC50.py  
$\qquad$ **Step 4:** 05_4_train_Ki.py 

### 3. Case study
$\qquad$ **Step 1:** case_study/06_1_encode.py  
$\qquad$ **Step 2:** 06_2_train_Ki_case_study.py  
$\qquad$ **Step 3:** 06_3_train_Kd_case_study.py  
$\qquad$ **Step 4:** 06_4_test_case_study.py

For case_study/DTI.txt: The first 86 entries in the DTI.txt file constitute a case study dataset consisting of known drugs (82 antiviral drugs, 1 antiparasitic drug and 3 unrelated drugs) and the target SARS-CoV2 3C-like Protease.
