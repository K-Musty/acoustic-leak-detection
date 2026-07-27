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

def load_pretrained(model, pretrained_path):
    """Load pretrained weights from MIMII training."""
    if not os.path.exists(pretrained_path):
        print(f"⚠️ Pretrained weights not found: {pretrained_path}")
        return model
    
    try:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        # Load only encoder weights (ignore classifier head)
        model.encoder.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded pretrained weights from {pretrained_path}")
    except Exception as e:
        print(f"⚠️ Failed to load pretrained weights: {e}")
    return model

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    
    set_seed(config.get('seed', 42))
    
    # Determine input dimension
    input_dim = 2000 if config.get('dataset') == 'mimii' else 888
    
    # Create dataset split (from config, supports 'test' for MIMII pretraining)
    dataset = UnifiedAcousticDataset(
        dataset=config['dataset'],
        machine=config.get('machine', 'fan'),
        split=config.get('split', 'train'),   # ← SUPPORTS 'test' NOW
        shot=config['shot'],
        seed=config.get('seed', 42),
        snr=config.get('snr'),
        data_root=config.get('data_root', 'data/processed')
    )
    val_dataset = UnifiedAcousticDataset(
        dataset=config['dataset'],
        machine=config.get('machine', 'fan'),
        split=config.get('split', 'val'),     # ← SUPPORTS 'test' NOW
        shot=config['shot'],
        seed=config.get('seed', 42),
        snr=config.get('snr'),
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
    
    # Load pretrained weights if specified
    if config.get('pretrained_path'):
        model = load_pretrained(model, config['pretrained_path'])
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 0.0)
    )
    
    best_val_acc = 0
    patience_counter = 0
    patience = config.get('patience', 20)
    epochs = config.get('epochs', 50)
    
    print(f"\n🚀 Starting {epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(epochs):
        train_loss = train_epoch(
            model, dataset, optimizer,
            num_episodes=config.get('episodes_per_epoch', 100),
            device=device
        )
        val_acc = evaluate(model, val_dataset, num_episodes=50, device=device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs('experiments', exist_ok=True)
            torch.save(model.state_dict(), f'experiments/best_{config["name"]}.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    
    print("=" * 60)
    print(f"✅ Best val acc: {best_val_acc:.2f}%")
    return best_val_acc

if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/baseline.yaml'
    train(config_path)
