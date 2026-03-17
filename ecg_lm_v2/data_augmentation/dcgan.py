import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DCGANGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        seq_length: int = 5000,
        num_leads: int = 12,
        features_dim: int = 512,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.num_leads = num_leads
        
        self.init_size = seq_length // 32
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size),
            nn.BatchNorm1d(128 * self.init_size),
            nn.ReLU(inplace=True)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            
            nn.Conv1d(64, num_leads, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.init_size)
        out = self.conv_blocks(out)
        
        if out.size(2) < self.seq_length:
            out = F.interpolate(out, size=self.seq_length, mode='linear', align_corners=False)
        elif out.size(2) > self.seq_length:
            out = out[:, :, :self.seq_length]
            
        return out


class DCGANDiscriminator(nn.Module):
    def __init__(
        self,
        seq_length: int = 5000,
        num_leads: int = 12,
    ):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(num_leads, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        feat_size = 512 * (seq_length // 16)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_blocks(x)
        out = self.fc(out)
        return out


class DCGAN:
    def __init__(
        self,
        latent_dim: int = 100,
        seq_length: int = 5000,
        num_leads: int = 12,
        device: str = "cuda",
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
    ):
        self.latent_dim = latent_dim
        self.device = device
        
        self.generator = DCGANGenerator(
            latent_dim=latent_dim,
            seq_length=seq_length,
            num_leads=num_leads,
        ).to(device)
        
        self.discriminator = DCGANDiscriminator(
            seq_length=seq_length,
            num_leads=num_leads,
        ).to(device)
        
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2)
        )
        
        self.adversarial_loss = nn.BCELoss()
        
    def train_step(self, real_ecg: torch.Tensor) -> Dict[str, float]:
        batch_size = real_ecg.size(0)
        valid = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)
        
        self.optimizer_G.zero_grad()
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        gen_ecg = self.generator(noise)
        
        g_loss = self.adversarial_loss(
            self.discriminator(gen_ecg), valid
        )
        
        g_loss.backward()
        self.optimizer_G.step()
        
        self.optimizer_D.zero_grad()
        real_loss = self.adversarial_loss(
            self.discriminator(real_ecg), valid
        )
        fake_loss = self.adversarial_loss(
            self.discriminator(gen_ecg.detach()), fake
        )
        
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        self.optimizer_D.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
        }
    
    def generate(self, num_samples: int) -> torch.Tensor:
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim).to(self.device)
            gen_ecg = self.generator(noise)
        self.generator.train()
        return gen_ecg
    
    def save_checkpoint(self, path: str):
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
