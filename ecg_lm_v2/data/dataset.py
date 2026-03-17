import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import wfdb
import pywt


class ECGDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_leads: int = 12,
        sampling_rate: int = 500,
        sequence_length: int = 5000,
        transform=None,
        target_transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_leads = num_leads
        self.sampling_rate = sampling_rate
        self.sequence_length = sequence_length
        self.transform = transform
        self.target_transform = target_transform
        
        self.data_files = self._load_file_list()
        
    def _load_file_list(self) -> List[str]:
        split_file = self.data_dir / f"{self.split}_files.txt"
        if split_file.exists():
            with open(split_file, 'r') as f:
                files = [line.strip() for line in f.readlines()]
        else:
            files = []
            for ext in ['*.mat', '*.hea', '*.csv']:
                files.extend([str(f.stem) for f in self.data_dir.glob(ext) 
                             if f.is_file()])
            files = list(set(files))
        return files
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def _load_mat_file(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        mat_data = sio.loadmat(str(file_path))
        if 'val' in mat_data:
            ecg_signal = mat_data['val']
        elif 'ecg' in mat_data:
            ecg_signal = mat_data['ecg']
        else:
            ecg_signal = list(mat_data.values())[0]
            
        if 'rhythm' in mat_data:
            label = mat_data['rhythm']
        elif 'label' in mat_data:
            label = mat_data['label']
        else:
            label = "Unknown"
            
        return ecg_signal, label
    
    def _load_wfdb_file(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        record = wfdb.rdrecord(str(file_path.with_suffix('')))
        ecg_signal = record.p_signal
        annotation = wfdb.rdann(str(file_path.with_suffix('')), 'atr')
        
        metadata = {
            'fs': record.fs,
            'units': record.units,
            'comments': record.comments,
        }
        
        return ecg_signal, annotation.symbol
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_name = self.data_files[idx]
        
        file_path = self.data_dir / f"{file_name}.mat"
        if not file_path.exists():
            file_path = self.data_dir / f"{file_name}.hea"
            
        if file_path.suffix == '.mat':
            ecg_signal, label = self._load_file_list(file_path)
        else:
            ecg_signal, label = self._load_wfdb_file(file_path)
        
        ecg_signal = self._preprocess_signal(ecg_signal)
        
        if self.transform:
            ecg_signal = self.transform(ecg_signal)
            
        label = self._encode_label(label)
        
        if self.target_transform:
            label = self.target_transform(label)
            
        return ecg_signal, label
    
    def _preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T
            
        if signal.shape[0] < self.num_leads:
            padding = np.zeros((self.num_leads - signal.shape[0], signal.shape[1]))
            signal = np.vstack([signal, padding])
        elif signal.shape[0] > self.num_leads:
            signal = signal[:self.num_leads, :]
        
        if signal.shape[1] > self.sequence_length:
            start_idx = np.random.randint(0, signal.shape[1] - self.sequence_length)
            signal = signal[:, start_idx:start_idx + self.sequence_length]
        elif signal.shape[1] < self.sequence_length:
            pad_len = self.sequence_length - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_len)), mode='constant')
            
        signal = signal.astype(np.float32)
        
        return signal
    
    def _encode_label(self, label) -> int:
        label_map = {
            "Normal": 0, "NORM": 0,
            "AFib": 1, "AF": 1, "AFIB": 1,
            "AFL": 2, "AFLU": 2,
            "SVT": 3, "SVTACH": 3,
            "VT": 4, "VTA": 4,
            "PVC": 5, "PVCs": 5,
            "PAC": 6, "PACs": 6,
            "LBBB": 7,
            "RBBB": 8,
            "Brady": 9, "BRADY": 9,
            "Tachy": 10, "TACHY": 10,
            "MI": 11, "MI-": 11, "MI+": 11,
        }
        
        if isinstance(label, (list, np.ndarray)):
            label = label[0] if len(label) > 0 else "Unknown"
            
        return label_map.get(str(label), 0)


class SyntheticECGDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 1000,
        num_leads: int = 12,
        sampling_rate: int = 500,
        sequence_length: int = 5000,
        num_classes: int = 12,
    ):
        self.num_samples = num_samples
        self.num_leads = num_leads
        self.sampling_rate = sampling_rate
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
    def __len__(self) -> int:
        return self.num_samples
    
    def _generate_heartbeat(self, heart_rate: int = 70) -> np.ndarray:
        t = np.linspace(0, 1.0, int(self.sampling_rate))
        
        p_wave = 0.15 * np.sin(2 * np.pi * 1.2 * t)
        qrs_complex = 1.0 * np.sin(2 * np.pi * 2.5 * t) - 0.3 * np.sin(2 * np.pi * 5 * t)
        t_wave = 0.25 * np.sin(2 * np.pi * 1.5 * t)
        
        pr_interval = int(0.16 * self.sampling_rate)
        qrs_duration = int(0.08 * self.sampling_rate)
        st_interval = int(0.26 * self.sampling_rate)
        
        heartbeat = np.zeros(int(self.sampling_rate))
        
        heartbeat[:pr_interval] = p_wave[:pr_interval]
        heartbeat[pr_interval:pr_interval+qrs_duration] = qrs_complex[:qrs_duration]
        heartbeat[pr_interval+qrs_duration:pr_interval+qrs_duration+st_interval] = \
            t_wave[pr_interval:pr_interval+qrs_duration+st_interval]
            
        return heartbeat
    
    def _add_noise(self, signal: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        noise = np.random.randn(*signal.shape) * noise_level
        return signal + noise
    
    def _baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        t = np.linspace(0, 1, signal.shape[1])
        baseline = 0.1 * np.sin(2 * np.pi * 0.5 * t)
        baseline = np.tile(baseline, (signal.shape[0], 1))
        return signal + baseline
    
    def _add_arrhythmia(self, signal: np.ndarray, label: int) -> np.ndarray:
        if label in [5, 6]:
            num_extra = np.random.randint(1, 4)
            for _ in range(num_extra):
                pos = np.random.randint(100, signal.shape[1] - 100)
                heartbeat = self._generate_heartbeat()
                signal[:, pos:pos+len(heartbeat)] += heartbeat * 1.5
                
        elif label in [1, 2]:
            t = np.linspace(0, 1, signal.shape[1])
            irregular = 0.3 * np.sin(2 * np.pi * 5 * t) * np.random.rand(signal.shape[1])
            irregular = np.tile(irregular, (signal.shape[0], 1))
            signal = signal + irregular
            
        return signal
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = idx % self.num_classes
        
        num_beats = int(self.sequence_length / (self.sampling_rate / (70 + label * 5)))
        
        signal = np.zeros((self.num_leads, self.sequence_length), dtype=np.float32)
        
        for lead in range(self.num_leads):
            beat = self._generate_heartbeat(heart_rate=70 + label * 5)
            
            for i in range(num_beats):
                pos = i * len(beat)
                if pos + len(beat) <= self.sequence_length:
                    signal[lead, pos:pos+len(beat)] = beat
                    
            signal[lead] = self._add_noise(signal[lead], noise_level=0.02 + label * 0.005)
            signal[lead] = self._baseline_wander(signal[lead].reshape(1, -1)).flatten()
            
        signal = self._add_arrhythmia(signal, label)
        
        signal = (signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-8)
        
        return torch.tensor(signal, dtype=torch.float32), label


def get_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    use_synthetic: bool = False,
    synthetic_num_samples: int = 1000,
    **kwargs
) -> DataLoader:
    
    if use_synthetic:
        dataset = SyntheticECGDataset(
            num_samples=synthetic_num_samples,
            num_leads=kwargs.get('num_leads', 12),
            sampling_rate=kwargs.get('sampling_rate', 500),
            sequence_length=kwargs.get('sequence_length', 5000),
            num_classes=kwargs.get('num_classes', 12),
        )
    else:
        dataset = ECGDataset(
            data_dir=data_dir,
            split=split,
            num_leads=kwargs.get('num_leads', 12),
            sampling_rate=kwargs.get('sampling_rate', 500),
            sequence_length=kwargs.get('sequence_length', 5000),
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=kwargs.get('pin_memory', True),
    )
