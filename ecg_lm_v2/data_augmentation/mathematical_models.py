import numpy as np
import torch
from typing import Dict, Tuple, Optional
from scipy.integrate import odeint


class CoupledODEModel:
    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        coupling_strength: float = 0.5,
    ):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.coupling_strength = coupling_strength
    
    def lorenz_system(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, z = state
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return np.array([dxdt, dydt, dzdt])
    
    def coupled_lorenz(self, state: np.ndarray, t: float) -> np.ndarray:
        n = len(state) // 3
        state = state.reshape(n, 3)
        
        derivatives = np.zeros_like(state)
        
        for i in range(n):
            x, y, z = state[i]
            dxdt = self.sigma * (y - x)
            dydt = x * (self.rho - z) - y
            dzdt = x * y - self.beta * z
            derivatives[i] = [dxdt, dydt, dzdt]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    derivatives[i] += self.coupling_strength * (state[j] - state[i])
        
        return derivatives.flatten()
    
    def generate_signal(
        self,
        duration: float = 10.0,
        dt: float = 0.001,
        initial_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        t = np.arange(0, duration, dt)
        
        if initial_state is None:
            initial_state = np.random.randn(6) + np.array([1, 1, 1, -1, -1, -1])
        
        states = odeint(self.coupled_lorenz, initial_state, t)
        
        return states[:, 0] + states[:, 3]
    
    def generate_ecg_like(
        self,
        num_samples: int = 5000,
        heart_rate: int = 70,
    ) -> np.ndarray:
        duration = num_samples / 360.0
        dt = 1.0 / 360.0
        
        t = np.linspace(0, duration, num_samples)
        
        heart_period = 60.0 / heart_rate
        num_beats = int(duration / heart_period)
        
        signal = np.zeros(num_samples)
        
        for beat in range(num_beats):
            beat_start = int(beat * heart_period * 360)
            if beat_start >= num_samples:
                break
            
            t_beat = np.linspace(0, heart_period, int(heart_period * 360))
            
            p_wave = 0.15 * np.sin(2 * np.pi * 1.2 * t_beat) * np.exp(-((t_beat - 0.15) ** 2) / 0.005)
            qrs_complex = 1.0 * np.sin(2 * np.pi * 2.5 * t_beat) * np.exp(-((t_beat - 0.35) ** 2) / 0.002)
            qrs_complex += -0.3 * np.sin(2 * np.pi * 5 * t_beat) * np.exp(-((t_beat - 0.35) ** 2) / 0.002)
            t_wave = 0.25 * np.sin(2 * np.pi * 1.5 * t_beat) * np.exp(-((t_beat - 0.55) ** 2) / 0.01)
            
            heartbeat = p_wave + qrs_complex + t_wave
            
            end_idx = min(beat_start + len(heartbeat), num_samples)
            signal[beat_start:end_idx] += heartbeat[:end_idx - beat_start]
        
        return signal


class ECGWaveformGenerator:
    def __init__(
        self,
        sampling_rate: int = 500,
        heart_rate: int = 70,
    ):
        self.sampling_rate = sampling_rate
        self.heart_rate = heart_rate
    
    def generate_p_wave(self, t: np.ndarray, amplitude: float = 0.15, duration: float = 0.08) -> np.ndarray:
        center = 0.15
        gaussian = np.exp(-((t - center) ** 2) / (2 * (duration / 3) ** 2))
        oscillation = np.sin(2 * np.pi * 1.2 * t)
        return amplitude * gaussian * oscillation
    
    def generate_qrs_complex(
        self,
        t: np.ndarray,
        q_amplitude: float = 0.15,
        r_amplitude: float = 1.0,
        s_amplitude: float = 0.25,
    ) -> np.ndarray:
        q_wave = -q_amplitude * np.exp(-((t - 0.32) ** 2) / 0.0005)
        r_wave = r_amplitude * np.exp(-((t - 0.35) ** 2) / 0.0003)
        s_wave = -s_amplitude * np.exp(-((t - 0.38) ** 2) / 0.0005)
        return q_wave + r_wave + s_wave
    
    def generate_t_wave(self, t: np.ndarray, amplitude: float = 0.25, duration: float = 0.16) -> np.ndarray:
        center = 0.55
        gaussian = np.exp(-((t - center) ** 2) / (2 * (duration / 3) ** 2))
        oscillation = np.sin(2 * np.pi * 1.5 * t)
        return amplitude * gaussian * oscillation
    
    def generate_heartbeat(
        self,
        pr_interval: float = 0.16,
        qrs_duration: float = 0.08,
        st_duration: float = 0.12,
        qt_interval: float = 0.4,
    ) -> np.ndarray:
        duration = pr_interval + qrs_duration + st_duration + (qt_interval - 0.2)
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        p_wave = self.generate_p_wave(t)
        
        qrs_complex = self.generate_qrs_complex(t)
        
        t_wave = self.generate_t_wave(t)
        
        heartbeat = np.zeros_like(t)
        
        p_end = int(pr_interval * self.sampling_rate)
        qrs_end = p_end + int(qrs_duration * self.sampling_rate)
        
        heartbeat[:p_end] = p_wave[:p_end]
        heartbeat[p_end:qrs_end] = qrs_complex[p_end:qrs_end]
        heartbeat[qrs_end:] = t_wave[qrs_end:]
        
        return heartbeat
    
    def generate_full_ecg(
        self,
        num_seconds: float = 10.0,
        noise_level: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_samples = int(num_seconds * self.sampling_rate)
        
        heart_period = 60.0 / self.heart_rate
        num_beats = int(num_seconds / heart_period)
        
        heartbeat = self.generate_heartbeat()
        
        signal = np.zeros(num_samples)
        time = np.arange(num_samples) / self.sampling_rate
        
        for i in range(num_beats):
            beat_start = int(i * heart_period * self.sampling_rate)
            beat_end = min(beat_start + len(heartbeat), num_samples)
            
            signal[beat_start:beat_end] += heartbeat[:beat_end - beat_start]
        
        noise = np.random.randn(num_samples) * noise_level
        signal += noise
        
        baseline_wander = 0.1 * np.sin(2 * np.pi * 0.5 * time)
        signal += baseline_wander
        
        return signal, time


class TemporalAutomataModel:
    def __init__(
        self,
        sampling_rate: int = 500,
        heart_rate: int = 70,
    ):
        self.sampling_rate = sampling_rate
        self.heart_rate = heart_rate
        
        self.states = ['baseline', 'p_wave', 'pr_segment', 'qrs_complex', 'st_segment', 't_wave']
        
        self.transitions = {
            'baseline': 'p_wave',
            'p_wave': 'pr_segment',
            'pr_segment': 'qrs_complex',
            'qrs_complex': 'st_segment',
            'st_segment': 't_wave',
            't_wave': 'baseline',
        }
        
        self.state_durations = {
            'baseline': lambda: np.random.uniform(0.3, 0.5),
            'p_wave': lambda: np.random.uniform(0.08, 0.12),
            'pr_segment': lambda: np.random.uniform(0.05, 0.08),
            'qrs_complex': lambda: np.random.uniform(0.06, 0.10),
            'st_segment': lambda: np.random.uniform(0.05, 0.10),
            't_wave': lambda: np.random.uniform(0.10, 0.18),
        }
    
    def generate_state_signal(self, state: str, duration: float) -> np.ndarray:
        num_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, num_samples)
        
        if state == 'baseline':
            return np.zeros(num_samples)
        elif state == 'p_wave':
            center = duration / 2
            return 0.15 * np.sin(2 * np.pi * 1.2 * t) * np.exp(-((t - center) ** 2) / (2 * (duration / 3) ** 2))
        elif state == 'pr_segment':
            return np.zeros(num_samples)
        elif state == 'qrs_complex':
            center = duration / 2
            r_wave = 1.0 * np.exp(-((t - center) ** 2) / 0.0003)
            q_wave = -0.15 * np.exp(-((t - center * 0.8) ** 2) / 0.0005)
            s_wave = -0.25 * np.exp(-((t - center * 1.2) ** 2) / 0.0005)
            return q_wave + r_wave + s_wave
        elif state == 'st_segment':
            return np.zeros(num_samples)
        elif state == 't_wave':
            center = duration / 2
            return 0.25 * np.sin(2 * np.pi * 1.5 * t) * np.exp(-((t - center) ** 2) / (2 * (duration / 3) ** 2))
        
        return np.zeros(num_samples)
    
    def generate(self, num_seconds: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        num_samples = int(num_seconds * self.sampling_rate)
        
        signal = np.zeros(num_samples)
        time = np.arange(num_samples) / self.sampling_rate
        
        current_state = 'baseline'
        state_start = 0
        sample_idx = 0
        
        while sample_idx < num_samples:
            duration = self.state_durations[current_state]()
            state_signal = self.generate_state_signal(current_state, duration)
            
            end_idx = min(sample_idx + len(state_signal), num_samples)
            signal[sample_idx:end_idx] = state_signal[:end_idx - sample_idx]
            
            sample_idx = end_idx
            
            current_state = self.transitions[current_state]
        
        return signal, time


def generate_ecg_with_ode(
    num_seconds: float = 10.0,
    heart_rate: int = 70,
    sampling_rate: int = 500,
    add_noise: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    model = CoupledODEModel()
    waveform_gen = ECGWaveformGenerator(sampling_rate=sampling_rate, heart_rate=heart_rate)
    
    signal, time = waveform_gen.generate_full_ecg(num_seconds=num_seconds, noise_level=0.02 if add_noise else 0.0)
    
    return signal, time
