import sys
import os
import numpy as np
import torch

# Add project root and src to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import the unified dataset loader (now works because src is in path)
from data.unified_dataset import UnifiedAcousticDataset

print("=" * 60)
print("TESTING UNIFIED DATASET LOADER")
print("=" * 60)

# 1. Test GPLA
print("\n1️⃣ Testing GPLA (train split):")
dataset_gpla = UnifiedAcousticDataset(
    dataset='gpla',
    split='train',
    shot=5,
    seed=42
)
support_X, support_y, query_X, query_y = dataset_gpla.get_episode()
print(f"  Support X: {support_X.shape}")
print(f"  Support y: {support_y.shape}")
print(f"  Query X:   {query_X.shape}")
print(f"  Query y:   {query_y.shape}")
print(f"  Classes:   {dataset_gpla.classes}")

# 2. Test GPLA with noise
print("\n2️⃣ Testing GPLA (SNR=5dB):")
dataset_noisy = UnifiedAcousticDataset(
    dataset='gpla',
    split='train',
    shot=3,
    seed=42,
    snr=5
)
support_X, support_y, query_X, query_y = dataset_noisy.get_episode()
print(f"  Support X: {support_X.shape}")
print(f"  Query X:   {query_X.shape}")
print(f"  Signal range: [{support_X.min():.3f}, {support_X.max():.3f}]")

# 3. Test MIMII (Fan) - test split
print("\n3️⃣ Testing MIMII (Fan - TEST split):")
dataset_fan = UnifiedAcousticDataset(
    dataset='mimii',
    machine='fan',
    split='test',
    shot=5,
    seed=42
)
support_X, support_y, query_X, query_y = dataset_fan.get_episode()
print(f"  Support X: {support_X.shape}")
print(f"  Support y: {support_y.shape}")
print(f"  Query X:   {query_X.shape}")
print(f"  Query y:   {query_y.shape}")
print(f"  Classes:   {dataset_fan.classes}")
print(f"  Normal: {np.sum(dataset_fan.y==0)}, Anomaly: {np.sum(dataset_fan.y==1)}")

# 4. Test MIMII (Pump)
print("\n4️⃣ Testing MIMII (Pump - TEST split):")
dataset_pump = UnifiedAcousticDataset(
    dataset='mimii',
    machine='pump',
    split='test',
    shot=5,
    seed=42
)
support_X, support_y, query_X, query_y = dataset_pump.get_episode()
print(f"  Support X: {support_X.shape}")
print(f"  Classes:   {dataset_pump.classes}")
print(f"  Normal: {np.sum(dataset_pump.y==0)}, Anomaly: {np.sum(dataset_pump.y==1)}")

# 5. Test MIMII with noise
print("\n5️⃣ Testing MIMII (Fan - TEST split, SNR=5dB):")
dataset_fan_noisy = UnifiedAcousticDataset(
    dataset='mimii',
    machine='fan',
    split='test',
    shot=3,
    seed=42,
    snr=5
)
support_X, support_y, query_X, query_y = dataset_fan_noisy.get_episode()
print(f"  Support X: {support_X.shape}")
print(f"  Query X:   {query_X.shape}")
print(f"  Signal range: [{support_X.min():.3f}, {support_X.max():.3f}]")

print("\n✅ All tests passed! Unified dataset loader is ready.")
