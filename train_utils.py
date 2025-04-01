import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import scipy.stats as stats
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, num_epochs):
    """plot training and validation loss over epochs."""
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()
    
    
def train(model, train_loader, val_loader, num_epochs, learning_rate, weight_decay, device):
    """Train the GCN model and evaluate its performance over epochs."""
    
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Define loss criterion, optimizer, and learning rate scheduler
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            target = data.y.view(data.num_graphs, -1).to(device)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.num_graphs
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        all_preds, all_targets = [], []
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data)
                target = data.y.view(data.num_graphs, -1).to(device)
                loss = criterion(output, target)
                val_loss += loss.item() * data.num_graphs
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step()

    # save the trained model
    torch.save(model.state_dict(), 'best_model.pth')

    # Plot training and validation loss
    plot_losses(train_losses, val_losses, num_epochs)
    
    return model

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data and compute RMSE and R² with 95% confidence intervals.
    """
    model.eval()
    all_preds, all_targets = [], []
    
    # get predictions and targets from test set
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            target = data.y.view(data.num_graphs, -1).to(device)
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # calculate RMSE and R²
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    
    # compute 95% Confidence Interval for RMSE
    squared_errors = (all_preds - all_targets) ** 2
    mean_se = np.mean(squared_errors)
    se = stats.sem(squared_errors)
    ci_mse = stats.t.interval(0.95, len(squared_errors)-1, loc=mean_se, scale=se)
    ci_rmse = (np.sqrt(ci_mse[0]), np.sqrt(ci_mse[1]))
    
    # bootstrapping for R^2 CI
    bootstrap_r2 = []
    n = len(all_targets)
    for _ in range(1000):
        indices = np.random.choice(n, n, replace=True)
        boot_preds = all_preds[indices]
        boot_targets = all_targets[indices]
        bootstrap_r2.append(r2_score(boot_targets, boot_preds))
    ci_r2 = np.percentile(bootstrap_r2, [2.5, 97.5])
    
    print(f"Test RMSE: {rmse:.4f} with 95% CI: [{ci_rmse[0]:.4f}, {ci_rmse[1]:.4f}]")
    print(f"Test R²: {r2:.4f} with 95% CI: [{ci_r2[0]:.4f}, {ci_r2[1]:.4f}]")
    
    return rmse, ci_rmse, r2, ci_r2

