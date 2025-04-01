import torch
import torch.nn.functional as F
from gcn import GCN 
from torch_geometric.data import Data

import matplotlib.cm as cmaps
from rdkit.Chem import rdMolDraw2D
import IPython.display as ipd
import numpy as np
from rdkit import Chem

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