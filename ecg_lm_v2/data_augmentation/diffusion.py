import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.time_emb = nn.Linear(time_emb_dim, channels)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        h = h + self.time_emb(time_emb)[:, :, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return x + h


class ECG DiffusionGenerator(nn.Module):
    def __init__(
        self,
        num_leads: int = 12,
        seq_length: int = 5000,
        channels: int = 64,
        time_emb_dim: int = 128,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.seq_length = seq_length
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        
        self.conv_in = nn.Conv1d(num_leads, channels, kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList()
        ch = channels
        for _ in range(3):
            self.downs.append(nn.ModuleList([
                ResidualBlock(ch, time_emb_dim),
                ResidualBlock(ch, time_emb_dim),
            ]))
            self.downs.append(nn.Conv1d(ch, ch * 2, kernel_size=3, stride=2, padding=1))
            ch *= 2
        
        self.mid = nn.ModuleList([
            ResidualBlock(ch, time_emb_dim),
            ResidualBlock(ch, time_emb_dim),
        ])
        
        self.ups = nn.ModuleList()
        for _ in range(3):
            self.ups.append(nn.ConvTranspose1d(ch, ch // 2, kernel_size=4, stride=2, padding=1))
            self.ups.append(nn.ModuleList([
                ResidualBlock(ch, time_emb_dim),
                ResidualBlock(ch, time_emb_dim),
            ]))
            ch //= 2
        
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, num_leads, kernel_size=3, padding=1),
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        
        h = self.conv_in(x)
        
        for i in range(0, len(self.downs), 3):
            for block in self.downs[i:i+2]:
                h = block(h, t_emb)
            h = self.downs[i+2](h)
        
        for block in self.mid:
            h = block(h, t_emb)
        
        for i in range(0, len(self.ups), 3):
            h = self.ups[i](h)
            for block in self.ups[i+1:i+3]:
                h = block(h, t_emb)
        
        h = self.conv_out(h)
        return h


class DiffusionModel:
    def __init__(
        self,
        num_leads: int = 12,
        seq_length: int = 5000,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda",
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        self.model = ECGDiffusionGenerator(
            num_leads=num_leads,
            seq_length=seq_length,
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x0.shape[0]
        alpha_hat_t = self.alpha_hat[t].reshape(batch_size, 1, 1)
        
        noise = torch.randn_like(x0)
        xt = torch.sqrt(alpha_hat_t) * x0 + torch.sqrt(1 - alpha_hat_t) * noise
        
        return xt, noise
    
    def train_step(self, x0: torch.Tensor) -> Dict[str, float]:
        batch_size = x0.shape[0]
        
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        
        xt, noise = self.forward_diffusion(x0, t)
        
        predicted_noise = self.model(xt, t)
        
        loss = F.mse_loss(predicted_noise, noise)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    @torch.no_grad()
    def sample(self, num_samples: int) -> torch.Tensor:
        self.model.eval()
        
        xT = torch.randn(num_samples, self.num_leads, self.seq_length).to(self.device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((num_samples,), t, device=self.device).long()
            
            predicted_noise = self.model(xT, t_batch)
            
            alpha_t = self.alpha[t]
            alpha_hat_t = self.alpha_hat[t]
            alpha_hat_t_1 = self.alpha_hat[t - 1] if t > 0 else torch.tensor(1.0).to(self.device)
            
            beta_t = self.beta[t]
            
            mean = (xT - beta_t / torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            if t > 0:
                noise = torch.randn_like(xT)
                variance = beta_t
                xT = mean + torch.sqrt(variance) * noise
            else:
                xT = mean
        
        self.model.train()
        return xT
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class DDIMScheduler:
    def __init__(
        self,
        num_timesteps: int = 1000,
        num_inference_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_timesteps = num_timesteps
        self.num_inference_steps = num_inference_steps
        
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        timesteps = torch.linspace(num_timesteps - 1, 0, num_inference_steps).long()
        self.timesteps = timesteps
        
    def get_alpha(self, t: int) -> float:
        return self.alpha_hat[t].item()
    
    def set_alpha(self, t: int, alpha: float):
        self.alpha_hat[t] = alpha
