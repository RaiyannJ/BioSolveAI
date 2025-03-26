#this is the GNN architecture 
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader

class GNN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=64, output_dim=1):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = torch.nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Graph-level readout (pooling)
        x = global_mean_pool(x, batch)

        # Fully connected layers for final prediction
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Instantiate the model
node_feature_dim = 6  # as in your data loader (atomic features)
edge_feature_dim = 4  # as you defined (bond features)
model = GNN(node_feature_dim, edge_feature_dim)

# Example training loop (pseudo-code, replace placeholders accordingly):
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out.squeeze(), data.y.float())  # Assuming regression task
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# Example usage (with your DataLoader):
# dataset is your PyG dataset created using your existing preprocessing pipeline
loader = DataLoader(dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # Assuming regression for solubility

for epoch in range(1, 101):
    loss = train(model, loader, optimizer, criterion)
    print(f'Epoch {epoch}, Loss: {loss:.4f}')
