# src/data/unified_dataset.py (CORRECTED VERSION)
import os
import json
import numpy as np
import torch

class UnifiedAcousticDataset:
    def __init__(self, dataset='gpla', machine='fan', split='train', 
                 shot=5, seed=42, snr=None, data_root='data/processed'):
        self.dataset = dataset
        self.machine = machine
        self.split = split
        self.shot = shot
        self.seed = seed
        self.snr = snr
        self.data_root = data_root
        self.rng = np.random.RandomState(seed)
        self._load_data()

    def _load_data(self):
        if self.dataset == 'gpla':
            self._load_gpla()
        elif self.dataset == 'mimii':
            self._load_mimii()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def _load_gpla(self):
        path = f'{self.data_root}/gpla_v2/'
        self.X = np.load(f'{path}/X_{self.split}.npy')
        self.y = np.load(f'{path}/y_{self.split}.npy')
        
        with open(f'{path}/metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # ============================================================
        # FIX: Convert 1-indexed labels to 0-indexed
        # ============================================================
        self.classes = [c - 1 for c in self.metadata['classes']]  # 1-12 → 0-11
        self.num_classes = len(self.classes)
        
        # Map 0-indexed class to indices in data (which are 1-indexed)
        self.class_to_idx = {c: np.where(self.y == c + 1)[0].tolist() for c in self.classes}
        
        # Shift stored labels to 0-indexed
        self.y = self.y - 1
        
        print(f"✅ Loaded GPLA ({self.split}): {len(self.X)} samples, {self.num_classes} classes")
        print(f"   Classes: {self.classes} (0-indexed)")

    def _load_mimii(self):
        path = f'{self.data_root}/mimii/{self.machine}/'
        self.X = np.load(f'{path}/X_{self.split}.npy')
        self.y = np.load(f'{path}/y_{self.split}.npy')
        
        with open(f'{path}/metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.classes = [0, 1]
        self.num_classes = len(self.classes)
        self.class_to_idx = {
            0: np.where(self.y == 0)[0].tolist(),
            1: np.where(self.y == 1)[0].tolist()
        }
        
        print(f"✅ Loaded MIMII ({self.machine}, {self.split}): {len(self.X)} samples, {self.num_classes} classes")
        for c in self.classes:
            print(f"   Class {c}: {len(self.class_to_idx[c])} samples")

    def _add_noise(self, signal):
        if self.snr is None:
            return signal
        noise = np.random.normal(0, 1, len(signal))
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        scale = np.sqrt(signal_power / (noise_power * 10 ** (self.snr / 10)))
        return signal + scale * noise

    def get_episode(self):
        available_classes = []
        for cls in self.classes:
            cnt = len(self.class_to_idx[cls])
            if cnt >= self.shot * 2:
                available_classes.append(cls)
            else:
                if cnt > 0:
                    print(f"⚠️ Class {cls} has only {cnt} samples (need {self.shot*2}) - skipping")
                else:
                    print(f"⚠️ Class {cls} has 0 samples - skipping")
        
        if not available_classes:
            raise ValueError(f"No class has enough samples for shot={self.shot}. "
                             f"Counts: {[(c, len(self.class_to_idx[c])) for c in self.classes]}")
        
        num_to_sample = min(self.num_classes, len(available_classes))
        classes = self.rng.choice(available_classes, num_to_sample, replace=False)
        
        support_X, support_y, query_X, query_y = [], [], [], []
        for cls in classes:
            idx = self.class_to_idx[cls]
            if len(idx) < self.shot * 2:
                selected = self.rng.choice(idx, self.shot * 2, replace=True)
            else:
                selected = self.rng.choice(idx, self.shot * 2, replace=False)
            
            for i in range(self.shot):
                sig = self.X[selected[i]].copy()
                sig = self._add_noise(sig)
                support_X.append(sig)
                support_y.append(cls)
            
            for i in range(self.shot, self.shot * 2):
                sig = self.X[selected[i]].copy()
                sig = self._add_noise(sig)
                query_X.append(sig)
                query_y.append(cls)
        
        return (
            torch.FloatTensor(np.array(support_X)),
            torch.LongTensor(np.array(support_y)),
            torch.FloatTensor(np.array(query_X)),
            torch.LongTensor(np.array(query_y))
        )
