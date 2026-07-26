import os
import json
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_mimii_files(machine_dir, split='train', class_type='normal', max_len=2000, target_sr=1000):
    """
    Load WAV files for a given machine, split, and class type.
    
    Args:
        machine_dir: Path to machine directory
        split: 'train' or 'test'
        class_type: 'normal' or 'anomaly'
        max_len: Fixed sequence length
        target_sr: Target sampling rate
    """
    split_dir = os.path.join(machine_dir, split)
    
    # Find matching files
    pattern = f"{class_type}_id_*" if class_type == 'normal' else f"{class_type}_id_*"
    import glob
    files = glob.glob(os.path.join(split_dir, f"{class_type}_id_*.wav"))
    
    X = []
    for filepath in tqdm(files, desc=f"{split}/{class_type}"):
        signal, sr = librosa.load(filepath, sr=target_sr)
        
        # Trim/pad to fixed length
        if len(signal) > max_len:
            signal = signal[:max_len]
        else:
            signal = np.pad(signal, (0, max_len - len(signal)))
        
        X.append(signal)
    
    return np.array(X, dtype=np.float32)

def preprocess_mimii(raw_dir='data/raw/mimii', processed_dir='data/processed/mimii', max_len=2000):
    """
    Preprocess all MIMII machines with train/test splits.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    machines = ['fan', 'pump', 'slider', 'valve']
    all_metadata = {}
    
    for machine in machines:
        print(f"\n{'='*50}")
        print(f"🔧 Processing {machine.upper()}")
        print(f"{'='*50}")
        
        machine_dir = os.path.join(raw_dir, machine)
        
        # Load normal and anomaly separately
        # Train split usually contains normal only, test split contains anomaly only
        X_train_normal = load_mimii_files(machine_dir, 'train', 'normal', max_len)
        X_train_anomaly = load_mimii_files(machine_dir, 'train', 'anomaly', max_len)
        
        X_test_normal = load_mimii_files(machine_dir, 'test', 'normal', max_len)
        X_test_anomaly = load_mimii_files(machine_dir, 'test', 'anomaly', max_len)
        
        # Combine
        X_train = np.vstack([X_train_normal, X_train_anomaly]) if len(X_train_anomaly) > 0 else X_train_normal
        y_train = np.concatenate([
            np.zeros(len(X_train_normal)),
            np.ones(len(X_train_anomaly))
        ]) if len(X_train_anomaly) > 0 else np.zeros(len(X_train_normal))
        
        X_test = np.vstack([X_test_normal, X_test_anomaly]) if len(X_test_anomaly) > 0 else X_test_normal
        y_test = np.concatenate([
            np.zeros(len(X_test_normal)),
            np.ones(len(X_test_anomaly))
        ]) if len(X_test_anomaly) > 0 else np.zeros(len(X_test_normal))
        
        print(f"\n📊 {machine} Statistics:")
        print(f"  Train: {len(X_train)} samples (normal: {np.sum(y_train==0)}, anomaly: {np.sum(y_train==1)})")
        print(f"  Test:  {len(X_test)} samples (normal: {np.sum(y_test==0)}, anomaly: {np.sum(y_test==1)})")
        
        # Split train into train/val (80/20)
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        # Save
        machine_dir_out = os.path.join(processed_dir, machine)
        os.makedirs(machine_dir_out, exist_ok=True)
        
        np.save(f'{machine_dir_out}/X_train.npy', X_train_split)
        np.save(f'{machine_dir_out}/y_train.npy', y_train_split)
        np.save(f'{machine_dir_out}/X_val.npy', X_val)
        np.save(f'{machine_dir_out}/y_val.npy', y_val)
        np.save(f'{machine_dir_out}/X_test.npy', X_test)
        np.save(f'{machine_dir_out}/y_test.npy', y_test)
        
        # Metadata
        metadata = {
            'machine': machine,
            'total_samples': len(X_train) + len(X_test),
            'num_classes': 2,
            'classes': ['normal', 'anomaly'],
            'sequence_length': max_len,
            'sample_rate': 1000,
            'train_size': len(X_train_split),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'normal_train': int(np.sum(y_train_split == 0)),
            'anomaly_train': int(np.sum(y_train_split == 1)),
            'normal_val': int(np.sum(y_val == 0)),
            'anomaly_val': int(np.sum(y_val == 1)),
            'normal_test': int(np.sum(y_test == 0)),
            'anomaly_test': int(np.sum(y_test == 1))
        }
        
        with open(f'{machine_dir_out}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        all_metadata[machine] = metadata
        print(f"  ✅ Saved to {machine_dir_out}")
    
    # Save overall metadata
    with open(f'{processed_dir}/all_metadata.json', 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\n{'='*50}")
    print("✅ All MIMII machines processed!")
    
    return all_metadata

if __name__ == '__main__':
    preprocess_mimii()
