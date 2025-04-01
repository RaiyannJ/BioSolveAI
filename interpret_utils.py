import torch
import torch.nn.functional as F
from gcn import GCN 
from torch_geometric.data import Data

import matplotlib.cm as cmaps
# from rdkit.Chem import rdMolDraw2D -> broken, spent many hours cause of this
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

#pytorch tensor to numpy array
def cast_tensor(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t) # if alr np array just return

def display_header(text):
    print("\n====", text, "====\n")


def visualize_attribution(smiles, attribution_scores, molecule_name="Molecule", cmap_name='viridis'):
    """
    Visualizes node attribution scores on a 2D molecule depiction using RDKit.

    Args:
        smiles (str): SMILES string of the molecule.
        attribution_scores (torch.Tensor or np.ndarray): Node attribution scores.
        molecule_name (str): Name of the molecule for the title.
        cmap_name (str): Matplotlib colormap name (e.g. 'viridis').
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    attribution_scores_np = cast_tensor(attribution_scores)
    num_atoms = mol.GetNumAtoms()
    if len(attribution_scores_np) != num_atoms:
        raise ValueError(f"Number of attribution scores ({len(attribution_scores_np)}) must match number of atoms ({num_atoms})")

    # Normalize to [0, 1]
    min_score = np.min(attribution_scores_np)
    max_score = np.max(attribution_scores_np)
    normalized = (attribution_scores_np - min_score) / (max_score - min_score) if max_score > min_score else np.zeros_like(attribution_scores_np)

    cmap = cmaps.get_cmap(cmap_name)
    atom_colors = {i: tuple(cmap(score)[:3]) for i, score in enumerate(normalized)}

    drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=range(num_atoms), highlightAtomColors=atom_colors)
    drawer.FinishDrawing()

    display_header(f"Node Attribution Heatmap - {molecule_name}")
    ipd.display(ipd.SVG(drawer.GetDrawingText()))
    

def load_trained_model(model_path, model_args=None, device="cpu"):
    if model_args is None:
        raise ValueError("You must provide model_args to load the model architecture.")

    model = GCN(**model_args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_node_attribution_gnn(model, data_loader, sample_index=0, device="cpu"):
    model.eval()
    sample_data = list(data_loader)[sample_index].to(device)
    sample_data.x.requires_grad_(True)

    # Forward pass
    output = model(sample_data)
    prediction = output.item()

    # Backward pass for gradients
    model.zero_grad()
    output.backward()

    # Get gradients w.r.t. input node features
    gradients = sample_data.x.grad  # shape: [num_nodes, num_features]

    # Attribution: sum of absolute gradients per node
    node_attribution_scores = gradients.abs().sum(dim=1).detach().cpu()

    return node_attribution_scores, sample_data, prediction, sample_data.y.item()


def summarize_attributions(model, data_loader, num_samples=10, device="cpu"):
    summary = []

    for idx in range(min(num_samples, len(data_loader))):
        scores, data, pred, target = get_node_attribution_gnn(model, data_loader, idx, device)
        atom_idx = scores.argmax().item()
        mol = Chem.MolFromSmiles(data.smiles)
        atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol() if mol else "?"
        summary.append((pred, target, atom_symbol, scores.numpy()))

    return summary


def plot_top_atoms_histogram(summary):
    atoms = [entry[2] for entry in summary]
    counts = Counter(atoms)
    plt.figure(figsize=(6,4))
    plt.bar(counts.keys(), counts.values(), color='teal')
    plt.title("Top Attributed Atom Symbols")
    plt.xlabel("Atom")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()


def plot_prediction_vs_target(summary):
    preds = [entry[0] for entry in summary]
    targets = [entry[1] for entry in summary]
    plt.figure(figsize=(5,5))
    plt.scatter(targets, preds, c='darkorange')
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'k--')
    plt.xlabel("True Solubility")
    plt.ylabel("Predicted Solubility")
    plt.title("Predicted vs. Actual")
    plt.grid(True)
    plt.show()


def plot_mean_attribution_by_element(summary):
    element_scores = defaultdict(list)
    for (_, _, atom_symbol, scores) in summary:
        element_scores[atom_symbol].append(np.max(scores))

    elements = list(element_scores.keys())
    means = [np.mean(element_scores[el]) for el in elements]

    plt.figure(figsize=(6,4))
    plt.bar(elements, means, color='slateblue')
    plt.title("Mean Attribution by Atom Type")
    plt.xlabel("Atom Symbol")
    plt.ylabel("Mean Max Attribution")
    plt.grid(True)
    plt.show()