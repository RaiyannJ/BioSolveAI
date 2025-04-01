import torch
import torch.nn.functional as F
from gcn import GCN 
from torch_geometric.data import Data

def load_trained_model(model_path, model_args=None, device="cpu"):
    if model_args is None:
        raise ValueError("You must provide model_args to load the model architecture.")

    model = GCN(**model_args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

