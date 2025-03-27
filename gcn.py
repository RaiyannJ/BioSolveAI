import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU
class GCN1(nn.Module):
    def __init__(self, num_node_features, edge_attr_dim, u_dim, hidden_dim=64, output_dim=1):
        super(GCN1, self).__init__()

        # MLPs to preprocess edge attributes before passing to GINEConv
        self.edge_mlp1 = Sequential(Linear(edge_attr_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.edge_mlp2 = Sequential(Linear(edge_attr_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))

        # 1st layer: 3 parallel GINEConv layers
        self.conv1a = GINEConv(nn=Sequential(Linear(num_node_features, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)),
                               edge_dim=hidden_dim)
        self.conv1b = GINEConv(nn=Sequential(Linear(num_node_features, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)),
                               edge_dim=hidden_dim)
        self.conv1c = GINEConv(nn=Sequential(Linear(num_node_features, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)),
                               edge_dim=hidden_dim)

        self.lin1 = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # 2nd layer 3 parallel GCNConv layers on transformed features
        self.conv2a = GINEConv(nn=Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)),
                               edge_dim=hidden_dim)
        self.conv2b = GINEConv(nn=Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)),
                               edge_dim=hidden_dim)
        self.conv2c = GINEConv(nn=Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)),
                               edge_dim=hidden_dim)

        self.lin2 = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Final fully connected layer to produce 1 output (solubility prediction)
        self.fc = Linear(hidden_dim + u_dim, output_dim) #add the global feature 
        
def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch

        # Preprocess edge attributes for first GINE layer
        edge_attr1 = self.edge_mlp1(edge_attr)
        
       # 1st GINE layer (3 parallel paths)
        x1a = F.relu(self.conv1a(x, edge_index, edge_attr1))
        x1b = F.relu(self.conv1b(x, edge_index, edge_attr1))
        x1c = F.relu(self.conv1c(x, edge_index, edge_attr1))
        x1 = torch.cat([x1a, x1b, x1c], dim=1)
        x1 = F.relu(self.lin1(x1))
        
        # 2nd layer 3 parallel GCNs and combine
        x2a = F.relu(self.conv2a(x1, edge_index))
        x2b = F.relu(self.conv2b(x1, edge_index))
        x2c = F.relu(self.conv2c(x1, edge_index))
        x2 = torch.cat([x2a, x2b, x2c], dim=1)
        x2 = F.relu(self.lin2(x2))
        
        # Global mean pooling aggregates node features into a graph-level representation
        x_pool = global_mean_pool(x2, batch)
        
        # Final prediction layer
        out = self.fc(x_pool)
        return out
        