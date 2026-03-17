import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy.interpolate import interp1d
import pywt
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


class ECGPreprocessor:
    def __init__(
        self,
        sampling_rate: int = 500,
        lowcut: float = 0.5,
        highcut: float = 50.0,
        notch_freq: float = 50.0,
        normalize_method: str = "zscore",
    ):
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.normalize_method = normalize_method
        
    def bandpass_filter(
        self, 
        ecg: np.ndarray, 
        lowcut: Optional[float] = None,
        highcut: Optional[float] = None
    ) -> np.ndarray:
        if lowcut is None:
            lowcut = self.lowcut
        if highcut is None:
            highcut = self.highcut
            
        nyquist = self.sampling_rate / 2
        low = max(0.001, lowcut / nyquist)
        high = min(0.999, highcut / nyquist)
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        if ecg.ndim == 1:
            return signal.filtfilt(b, a, ecg)
        else:
            filtered = np.zeros_like(ecg)
            for i in range(ecg.shape[0]):
                filtered[i] = signal.filtfilt(b, a, ecg[i])
            return filtered
    
    def notch_filter(self, ecg: np.ndarray, freq: Optional[float] = None) -> np.ndarray:
        if freq is None:
            freq = self.notch_freq
            
        nyquist = self.sampling_rate / 2
        notch_freq_norm = freq / nyquist
        
        b, a = signal.iirnotch(notch_freq_norm, 30, self.sampling_rate)
        
        if ecg.ndim == 1:
            return signal.filtfilt(b, a, ecg)
        else:
            filtered = np.zeros_like(ecg)
            for i in range(ecg.shape[0]):
                filtered[i] = signal.filtfilt(b, a, ecg[i])
            return filtered
    
    def remove_baseline_wander(self, ecg: np.ndarray) -> np.ndarray:
        wavelet = 'db4'
        level = 8
        
        if ecg.ndim == 1:
            coeffs = pywt.wavedec(ecg, wavelet, level=level)
            coeffs[-1] = np.zeros_like(coeffs[-1])
            coeffs[-2] = np.zeros_like(coeffs[-2])
            reconstructed = pywt.waverec(coeffs, wavelet)
            return ecg - reconstructed[:len(ecg)]
        else:
            denoised = np.zeros_like(ecg)
            for i in range(ecg.shape[0]):
                c = pywt.wavedec(ecg[i], wavelet, level=level)
                c[-1] = np.zeros_like(c[-1])
                c[-2] = np.zeros_like(c[-2])
                reconstructed = pywt.waverec(c, wavelet)
                denoised[i] = ecg[i] - reconstructed[:len(ecg[i])]
            return denoised
    
    def denoise_wavelet(self, ecg: np.ndarray, wavelet: str = 'db4') -> np.ndarray:
        if ecg.ndim == 1:
            coeffs = pywt.wavedec(ecg, wavelet, level=5)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(ecg)))
            denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            return pywt.waverec(denoised_coeffs, wavelet)[:len(ecg)]
        else:
            denoised = np.zeros_like(ecg)
            for i in range(ecg.shape[0]):
                c = pywt.wavedec(ecg[i], wavelet, level=5)
                sigma = np.median(np.abs(c[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(ecg[i])))
                dc = [pywt.threshold(coef, threshold, mode='soft') for coef in c]
                denoised[i] = pywt.waverec(dc, wavelet)[:len(ecg[i])]
            return denoised
    
    def normalize(
        self, 
        ecg: np.ndarray, 
        method: Optional[str] = None
    ) -> np.ndarray:
        if method is None:
            method = self.normalize_method
            
        if method == "zscore":
            if ecg.ndim == 1:
                return (ecg - ecg.mean()) / (ecg.std() + 1e-8)
            else:
                mean = ecg.mean(axis=1, keepdims=True)
                std = ecg.std(axis=1, keepdims=True)
                return (ecg - mean) / (std + 1e-8)
        elif method == "minmax":
            if ecg.ndim == 1:
                min_val = ecg.min()
                max_val = ecg.max()
                return (ecg - min_val) / (max_val - min_val + 1e-8)
            else:
                min_val = ecg.min(axis=1, keepdims=True)
                max_val = ecg.max(axis=1, keepdims=True)
                return (ecg - min_val) / (max_val - min_val + 1e-8)
        return ecg
    
    def resample(self, ecg: np.ndarray, target_rate: int) -> np.ndarray:
        if target_rate == self.sampling_rate:
            return ecg
            
        ratio = target_rate / self.sampling_rate
        new_length = int(ecg.shape[-1] * ratio)
        
        if ecg.ndim == 1:
            return signal.resample(ecg, new_length)
        else:
            resampled = np.zeros((ecg.shape[0], new_length))
            for i in range(ecg.shape[0]):
                resampled[i] = signal.resample(ecg[i], new_length)
            return resampled
    
    def __call__(self, ecg: np.ndarray) -> np.ndarray:
        ecg = self.notch_filter(ecg)
        ecg = self.bandpass_filter(ecg)
        ecg = self.remove_baseline_wander(ecg)
        ecg = self.normalize(ecg)
        return ecg


class ECGDenoiser(nn.Module):
    def __init__(
        self,
        num_leads: int = 12,
        hidden_channels: int = 64,
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(num_leads, hidden_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels * 2, hidden_channels, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, num_leads, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded + x


class AdaptiveECGPreprocessor:
    def __init__(
        self,
        sampling_rate: int = 500,
        device: str = "cuda",
    ):
        self.sampling_rate = sampling_rate
        self.device = device
        self.denoiser = ECGDenoiser().to(device)
        self.optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=1e-4)
        self.scaler = torch.cuda.amp.GradScaler()
        
    def denoise(self, ecg: np.ndarray) -> np.ndarray:
        self.denoiser.eval()
        with torch.no_grad():
            x = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0).to(self.device)
            denoised = self.denoiser(x).cpu().numpy().squeeze(0)
        return denoised
    
    def train_denoiser(self, noisy_ecg: np.ndarray, clean_ecg: np.ndarray, epochs: int = 10):
        self.denoiser.train()
        
        noisy = torch.tensor(noisy_ecg, dtype=torch.float32).unsqueeze(0).to(self.device)
        clean = torch.tensor(clean_ecg, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                output = self.denoiser(noisy)
                loss = F.mse_loss(output, clean)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
        self.denoiser.eval()


class ECGQualityAssessment(nn.Module):
    def __init__(self, num_leads: int = 12):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(num_leads, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.features(x)
        features = features.squeeze(-1)
        quality = self.classifier(features)
        return quality
    
    def assess_quality(self, ecg: np.ndarray) -> float:
        self.eval()
        with torch.no_grad():
            if ecg.ndim == 1:
                ecg = ecg.reshape(1, -1)
            x = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0)
            quality = self.forward(x)
            return quality.item()


class ECGQualityMetrics:
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
    
    def compute_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)
    
    def compute_snr_estimation(self, ecg: np.ndarray) -> float:
        if ecg.ndim == 1:
            ecg = ecg.reshape(1, -1)
        
        snr_values = []
        for lead in ecg:
            qrs_mask = self._detect_qrs_region(lead)
            
            signal_region = lead[qrs_mask]
            noise_region = lead[~qrs_mask]
            
            signal_power = np.mean(signal_region ** 2)
            noise_power = np.var(noise_region)
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                snr_values.append(snr)
        
        return np.mean(snr_values) if snr_values else 0.0
    
    def _detect_qrs_region(self, lead: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        abs_lead = np.abs(lead)
        threshold_val = threshold * np.max(abs_lead)
        return abs_lead > threshold_val
    
    def compute_correlation(self, ecg1: np.ndarray, ecg2: np.ndarray) -> float:
        if ecg1.ndim > 1:
            correlations = [np.corrcoef(ecg1[i], ecg2[i])[0, 1] for i in range(ecg1.shape[0])]
            return np.mean([c for c in correlations if not np.isnan(c)])
        return np.corrcoef(ecg1, ecg2)[0, 1]
    
    def compute_kurtosis(self, ecg: np.ndarray) -> float:
        if ecg.ndim == 1:
            ecg = ecg.reshape(1, -1)
        
        kurtosis_values = []
        for lead in ecg:
            mean = np.mean(lead)
            std = np.std(lead)
            if std > 0:
                k = np.mean(((lead - mean) / std) ** 4) - 3
                kurtosis_values.append(k)
        
        return np.mean(kurtosis_values)
    
    def assess_quality_all(self, ecg: np.ndarray) -> Dict[str, float]:
        snr = self.compute_snr_estimation(ecg)
        kurtosis = self.compute_kurtosis(ecg)
        
        baseline_std = np.std(self._estimate_baseline(ecg))
        
        metrics = {
            'snr_db': snr,
            'kurtosis': kurtosis,
            'baseline_stability': baseline_std,
        }
        
        metrics['quality_score'] = self._compute_quality_score(metrics)
        
        return metrics
    
    def _estimate_baseline(self, ecg: np.ndarray) -> np.ndarray:
        if ecg.ndim == 1:
            return signal.medfilt(ecg, kernel_size=int(0.2 * self.sampling_rate))
        else:
            baseline = np.zeros_like(ecg)
            for i in range(ecg.shape[0]):
                baseline[i] = signal.medfilt(ecg[i], kernel_size=int(0.2 * self.sampling_rate))
            return baseline
    
    def _compute_quality_score(self, metrics: Dict[str, float]) -> float:
        snr_score = min(max((metrics['snr_db'] + 20) / 40, 0), 1) * 50
        
        kurtosis_diff = abs(metrics['kurtosis'])
        kurtosis_score = max(0, 25 - kurtosis_diff * 5)
        
        baseline_score = max(0, 25 - metrics['baseline_stability'] * 50)
        
        return snr_score + kurtosis_score + baseline_score


class ECGAugmentation:
    def __init__(
        self,
        noise_level: float = 0.05,
        shift_max: int = 100,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        dropout_prob: float = 0.1,
    ):
        self.noise_level = noise_level
        self.shift_max = shift_max
        self.scale_range = scale_range
        self.dropout_prob = dropout_prob
        
    def add_noise(self, ecg: np.ndarray, noise_level: Optional[float] = None) -> np.ndarray:
        if noise_level is None:
            noise_level = self.noise_level
        noise = np.random.randn(*ecg.shape) * noise_level
        return ecg + noise
    
    def random_shift(self, ecg: np.ndarray) -> np.ndarray:
        shift = np.random.randint(-self.shift_max, self.shift_max)
        return np.roll(ecg, shift, axis=-1)
    
    def random_scale(self, ecg: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(*self.scale_range)
        return ecg * scale
    
    def lead_dropout(self, ecg: np.ndarray) -> np.ndarray:
        if np.random.random() < self.dropout_prob:
            lead_idx = np.random.randint(0, ecg.shape[0])
            ecg = ecg.copy()
            ecg[lead_idx] = 0
        return ecg
    
    def time_mask(self, ecg: np.ndarray, max_mask_width: int = 100) -> np.ndarray:
        if np.random.random() < self.dropout_prob:
            ecg = ecg.copy()
            mask_width = np.random.randint(1, max_mask_width)
            mask_start = np.random.randint(0, ecg.shape[-1] - mask_width)
            ecg[..., mask_start:mask_start + mask_width] = 0
        return ecg
    
    def amplitude_warp(self, ecg: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        warp_factor = 1 + np.random.randn(*ecg.shape) * sigma
        return ecg * warp_factor
    
    def time_warp(self, ecg: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        if ecg.ndim == 1:
            return self._warp_1d(ecg, sigma)
        else:
            warped = np.zeros_like(ecg)
            for i in range(ecg.shape[0]):
                warped[i] = self._warp_1d(ecg[i], sigma)
            return warped
    
    def _warp_1d(self, x: np.ndarray, sigma: float) -> np.ndarray:
        orig_steps = np.arange(x.shape[0])
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(4,))
        warp_steps = np.linspace(0, x.shape[0] - 1, num=4 + 1)
        
        warper = interp1d(orig_steps, x, kind='cubic', fill_value='extrapolate')
        x_warped = warper(warp_steps * random_warps)
        
        return x_warped
    
    def mixup(self, ecg1: np.ndarray, ecg2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        lam = np.random.beta(alpha, alpha)
        return lam * ecg1 + (1 - lam) * ecg2
    
    def __call__(self, ecg: np.ndarray) -> np.ndarray:
        if np.random.random() < 0.5:
            ecg = self.add_noise(ecg)
        if np.random.random() < 0.5:
            ecg = self.random_shift(ecg)
        if np.random.random() < 0.3:
            ecg = self.random_scale(ecg)
        if np.random.random() < 0.3:
            ecg = self.lead_dropout(ecg)
        if np.random.random() < 0.2:
            ecg = self.time_mask(ecg)
        return ecg


def preprocess_ecg_batch(
    ecg_batch: torch.Tensor,
    preprocessing_config: Optional[Dict] = None
) -> torch.Tensor:
    if preprocessing_config is None:
        preprocessing_config = {
            'sampling_rate': 500,
            'lowcut': 0.5,
            'highcut': 50.0,
            'notch_freq': 50.0,
            'normalize_method': 'zscore'
        }
    
    preprocessor = ECGPreprocessor(**preprocessing_config)
    
    processed = np.zeros_like(ecg_batch.numpy())
    for i in range(ecg_batch.shape[0]):
        processed[i] = preprocessor(ecg_batch[i].numpy())
        
    return torch.tensor(processed, dtype=ecg_batch.dtype)
