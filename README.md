# MdDTI

Note that the input SMILES is unified using RDKit (by converting SMILES to a mol object using RDKit, and then converting the mol object back to unified SMILES).
'''python
smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

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
$\qquad$ **Step 1:** 06_1_encode.py  
$\qquad$ **Step 2:** 06_2_train_Ki_case_study.py  
$\qquad$ **Step 3:** 06_3_train_Kd_case_study.py  
$\qquad$ **Step 4:** 06_4_test_case_study.py
