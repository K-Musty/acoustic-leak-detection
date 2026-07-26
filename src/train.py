# src/train.py
import sys
import os
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.unified_dataset import UnifiedAcousticDataset
from models.prototypical import PrototypicalNetwork

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch(model, dataset, optimizer, num_episodes=100, device='cuda'):
    model.train()
    total_loss = 0
    for _ in range(num_episodes):
        support_X, support_y, query_X, query_y = dataset.get_episode()
        support_X = support_X.to(device)
        support_y = support_y.to(device)
        query_X = query_X.to(device)
        query_y = query_y.to(device)
        distances = model(support_X, support_y, query_X)
        loss = nn.CrossEntropyLoss()(-distances, query_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / num_episodes

def evaluate(model, dataset, num_episodes=50, device='cuda'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(num_episodes):
            support_X, support_y, query_X, query_y = dataset.get_episode()
            support_X = support_X.to(device)
            support_y = support_y.to(device)
            query_X = query_X.to(device)
            query_y = query_y.to(device)
            distances = model(support_X, support_y, query_X)
            preds = torch.argmin(distances, dim=1)
            correct += (preds == query_y).sum().item()
            total += query_y.size(0)
    return 100 * correct / total

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    
    set_seed(config.get('seed', 42))
    
    # Determine input dimension
    input_dim = 2000 if config.get('dataset') == 'mimii' else 888
    
    # Dataset
    dataset = UnifiedAcousticDataset(
        dataset=config['dataset'],
        machine=config.get('machine', 'fan'),
        split='train',
        shot=config['shot'],
        seed=config.get('seed', 42),
        snr=config.get('snr'),  # Pass SNR for noise experiments
        data_root=config.get('data_root', 'data/processed')
    )
    val_dataset = UnifiedAcousticDataset(
        dataset=config['dataset'],
        machine=config.get('machine', 'fan'),
        split='val',
        shot=config['shot'],
        seed=config.get('seed', 42),
        snr=config.get('snr'),  # Same noise for validation
        data_root=config.get('data_root', 'data/processed')
    )
    
    # Model
    model = PrototypicalNetwork(
        input_dim=input_dim,
        embed_dim=config.get('embed_dim', 128),
        use_self_attention=config.get('use_self_attention', False),
        use_cross_attention=config.get('use_cross_attention', False),
        conformer_kwargs=config.get('conformer_kwargs', {})
    )
    model = model.to(device)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-4))
    
    best_val_acc = 0
    epochs = config.get('epochs', 50)
    
    print(f"\n🚀 Starting {epochs} epochs...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, dataset, optimizer, 
                                num_episodes=config.get('episodes_per_epoch', 100),
                                device=device)
        val_acc = evaluate(model, val_dataset, num_episodes=50, device=device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('experiments', exist_ok=True)
            torch.save(model.state_dict(), f'experiments/best_{config["name"]}.pt')
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    print(f"\n✅ Best val acc: {best_val_acc:.2f}%")
    return best_val_acc

if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/baseline.yaml'
    train(config_path)
