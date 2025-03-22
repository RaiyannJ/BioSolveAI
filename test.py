import pandas as pd
import numpy as np

# RDkit Library
import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.GraphDescriptors import BertzCT, BalabanJ
import torch
from torch_geometric.data import Data

# Yeo-Johnson Trasnformation
from sklearn.preprocessing import PowerTransformer

def standardize_smiles(smiles):
  '''
  This function standardizes all smiles in the dataset
  to ensure consistency in the data.
  '''

  try:
      mol = Chem.MolFromSmiles(smiles)
      if mol:
          return Chem.MolToSmiles(mol)
      else:
          return None
  except:
      return None

def preproccess_data(file_path):
  '''
  This function loads the data as a pandas df, extracts the
  mol object from the smiles and then removes outliers and 
  transforms data to be more normalized.
  '''

  # load data set
  df = pd.read_csv(file_path)

  # obtain standardized SMILES for consistency (drop if invalid)
  df["SMILES"] = df["SMILES"].apply(standardize_smiles)
  df = df.dropna(subset=["SMILES"]).reset_index(drop=True)

  # obtain "mol" objects
  df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)

  # 3 std deviation range as obtained from EDA
  lower_threshold = -9.99
  upper_threshold = 4.21

  # filtering out values outside of this range (0.21% of data)
  df = df[(df['Solubility'] > lower_threshold) & (df['Solubility'] < upper_threshold)].copy()
  df = df.reset_index(drop=True)

  # Yeo-Johnson transformation to normalizae data
  normalizer = PowerTransformer(method='yeo-johnson')
  df['Solubility'] = normalizer.fit_transform(df[['Solubility']])

  assert df.shape == (9959, 27), "df shape is incorrect"
  
  return df

def mol_to_graph(mol):
  '''
  Convert each mol object into GraphTensors (lecture 13).
  So we extract Node features (X) which are atomic features,
  Adjacency matrix (A) for connectivity, 
  Edge features (E) which represent bond features and
  Global tensors (U) which are molecular properties.
  '''
    
  # Node features (X)
  atom_features = []
  for atom in mol.GetAtoms():
    atom_features.append([
        atom.GetAtomicNum(),
        atom.GetIsAromatic(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.IsInRing()
      ])
  x = torch.tensor(atom_features, dtype=torch.float)
  assert x.shape[0] == mol.GetNumAtoms(), "node features shape[0] incorrect"
  assert x.shape[1] == 6, "node features shape[1] is incorrect"

  # Adjacency Matrix (A)
  adj = rdmolops.GetAdjacencyMatrix(mol)
  adj = torch.tensor(adj, dtype=torch.long)
  edge_index = adj.nonzero(as_tuple=False).t().contiguous()
  
  assert edge_index.shape[1] == adj.sum() == 2 * mol.GetNumBonds(), "edge_index.shape[1] is incorrect"
  assert edge_index.shape[0] == 2, "edge_index.shape[0] is incorrect"
  assert adj.shape[0] == adj.shape[1] == mol.GetNumAtoms(), "adj matrix shape is incorrect"

  # Edge features (E)
  bond_type_to_idx = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
  }

  edge_attr = []
  for bond in mol.GetBonds():
    i = bond.GetBeginAtomIdx()
    j = bond.GetEndAtomIdx()
    bond_type = bond.GetBondType()
    bond_feature = [0] * len(bond_type_to_idx)
    bond_feature[bond_type_to_idx[bond_type]] = 1

    # Add edge in both directions (since PyG treats edges as undirected by default)
    edge_attr.append(bond_feature)
    edge_attr.append(bond_feature)

  edge_attr = torch.tensor(edge_attr, dtype=torch.float)
  
  assert edge_attr.shape[0] == 2 * mol.GetNumBonds(), "bond attributes shape[0] is incorrect"
  assert edge_attr.shape[1] == len(bond_type_to_idx), "bond attributes shape[1] is incorrect"
  assert torch.all(edge_attr.sum(dim=1) == 1.0), "bond attributes sums are not 1"

  # Global features (U), these were the 6 most
  # important features obtained from XGBoost.
  mol_wt = Descriptors.MolWt(mol)
  mol_logP = Crippen.MolLogP(mol)
  tpsa = rdMolDescriptors.CalcTPSA(mol)
  balabanJ = float(BalabanJ(mol))
  mol_mr = Crippen.MolMR(mol)
  bertzCT = BertzCT(mol)
  global_features = [mol_wt, mol_logP, tpsa, balabanJ, mol_mr, bertzCT]
  u = torch.tensor(global_features, dtype=torch.float)

  # create Pytorch geometric Data Object
  data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u = u)
  assert u.shape[0] == 6, "Global feature vector should have 6 features"

  return data
