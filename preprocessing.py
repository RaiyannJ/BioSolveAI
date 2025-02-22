'''
This script downloads the AqsolDB data sett



Source: https://www.kaggle.com/datasets/sorkun/aqsoldb-a-curated-aqueous-solubility-dataset

Paper citation: https://www.nature.com/articles/s41597-019-0151-1

'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw

# load dataset
file_path = "/Users/raiyannjacob/Desktop/ECE324/BioSolveAI/data/curated-solubility-dataset.csv"
df = pd.read_csv(file_path)

def get_information(dataset):
    df = dataset
    
    # Display basic information
    print(df.info())
    print(df.head())

    # Check missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print(f"There are no missing values.")
    
    # Check duplicates    
    duplicate_values = df.duplicated().sum()
    if  duplicate_values == 0:
        print(f"There are no duplicates.")
    
    # Display basic stats
    stats = df.describe().loc[["mean", "std", "min", "max"]]
    print(f"Mean, Std Dev, Min and Max Values of the dataset:\n{stats}")
    
    #mean = stats.loc["mean", "MolWt"]
    #print(mean)
    
'''# Distirbutions
plt.hist(df["Solubility"], bins=30, edgecolor="black")
plt.xlabel("Log Solubility (LogS)")
plt.ylabel("Frequency")
plt.title("Distribution of Solubility")
plt.show()

# Correlation Map
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()'''


df["mol"] = df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x) if pd.notnull(x) else None)
mol = df["mol"].dropna().iloc[0]
Draw.MolToImage(mol)

'''if __name__ == '__main__':
    get_information(df)'''