import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_dim=64, output_dim=1):
        super(GCN, self).__init__()
        # 1st layer 3 parallel GCNconv layer
        self.conv1a = GCNConv(num_node_features, hidden_dim)
        self.conv1b = GCNConv(num_node_features, hidden_dim)
        self.conv1c = GCNConv(num_node_features, hidden_dim)
        # Combine 3 outputs with a linear transformation
        self.lin1 = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # 2nd layer 3 parallel GCNConv layers on transformed features
        self.conv2a = GCNConv(hidden_dim, hidden_dim)
        self.conv2b = GCNConv(hidden_dim, hidden_dim)
        self.conv2c = GCNConv(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Final fully connected layer to produce 1 output (solubility prediction)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1st layer 3 parallel GCNs
        x1a = F.relu(self.conv1a(x, edge_index))
        x1b = F.relu(self.conv1b(x, edge_index))
        x1c = F.relu(self.conv1c(x, edge_index))
        