import pandas as pd
import numpy as np

# RDkit Library
import rdkit
import rdkit.Chem.Descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

# Yeo-Johnson Trasnformation
from sklearn.preprocessing import PowerTransformer

def standardize_smiles(smiles):
  '''
  smiles: smiles column from dataset

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
  file_path: file path to the dataset

  This function loads the data as a pandas df, extracts the
  mol object from the smiles and then removes outliers and 
  transforms data to be more normalized.

  This stems from EDA done in the EDA.ipynb file!

  Output is the preprocessed df, df
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

  return df

