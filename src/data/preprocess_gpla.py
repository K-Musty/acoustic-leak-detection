# src/data/preprocess_gpla.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_gpla():
    """Convert GPLA raw XLSX to processed NPY arrays."""
    
    raw_dir = 'data/raw/gpla_v2'
    proc_dir = 'data/processed/gpla_v2'
    
    os.makedirs(proc_dir, exist_ok=True)
    
    # Load data
    print("📂 Loading GPLA data...")
    X = pd.read_excel(f'{raw_dir}/data.xlsx', header=None).values.astype(np.float32)
    y = pd.read_excel(f'{raw_dir}/lable.xlsx', header=None).values.flatten().astype(np.int64)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    if X.shape[0] > y.shape[0]:
        X = X[:y.shape[0]]
    elif y.shape[0] > X.shape[0]:
        y = y[:X.shape[0]]
    print(f"Aligned: X {X.shape}, y {y.shape}")

    # FIX: Convert labels 1-12 to 0-11 for PyTorch
    y = y - 1

    print(f"✅ Loaded {len(X)} samples, {len(np.unique(y))} classes")
    print(f"   Labels: {np.unique(y)} (0-indexed)")
    
    # Normalize 12-bit ADC to [-1, 1]
    X = (X / 2048.0) - 1.0
    
    # Split: 60% train, 20% val, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    # Save
    np.save(f'{proc_dir}/X_train.npy', X_train)
    np.save(f'{proc_dir}/y_train.npy', y_train)
    np.save(f'{proc_dir}/X_val.npy', X_val)
    np.save(f'{proc_dir}/y_val.npy', y_val)
    np.save(f'{proc_dir}/X_test.npy', X_test)
    np.save(f'{proc_dir}/y_test.npy', y_test)
    
    # Metadata
    metadata = {
        'num_samples': len(X),
        'num_classes': len(np.unique(y)),
        'classes': np.unique(y).tolist(),
        'seq_len': X.shape[1],
        'sample_rate': 1000,
        'train': len(X_train),
        'val': len(X_val),
        'test': len(X_test)
    }
    with open(f'{proc_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved to {proc_dir}")
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

if __name__ == '__main__':
    preprocess_gpla()
