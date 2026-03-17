import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 12,
        seq_length: int = 5000,
        num_leads: int = 12,
        hidden_dims: list = [256, 512, 256],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.num_leads = num_leads
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        input_dim = latent_dim + num_classes
        self.init_size = seq_length // 32
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = hidden_dim
        self.fc = nn.Sequential(*layers)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], num_leads * seq_length),
            nn.Tanh()
        )
        
    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat([noise, label_embedding], dim=1)
        
        out = self.fc(gen_input)
        out = self.output(out)
        out = out.view(out.size(0), self.num_leads, self.seq_length)
        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        num_classes: int = 12,
        seq_length: int = 5000,
        num_leads: int = 12,
        hidden_dims: list = [256, 512, 256],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.num_leads = num_leads
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        input_dim = num_leads + num_classes
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        
        feat_size = 256 * (seq_length // 8)
        
        self.validity = nn.Sequential(
            nn.Linear(feat_size + num_classes, hidden_dims[-1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        label_embedding = self.label_emb(labels)
        label_embedding = label_embedding.unsqueeze(-1).expand(-1, -1, self.seq_length)
        
        d_in = torch.cat([x, label_embedding], dim=1)
        
        features = self.feature_extractor(d_in)
        features = torch.cat([features, label_embedding[:, :self.num_classes]], dim=1)
        
        validity = self.validity(features)
        return validity, features


class CGAN:
    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 12,
        seq_length: int = 5000,
        num_leads: int = 12,
        device: str = "cuda",
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
    ):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device
        
        self.generator = Generator(
            latent_dim=latent_dim,
            num_classes=num_classes,
            seq_length=seq_length,
            num_leads=num_leads,
        ).to(device)
        
        self.discriminator = Discriminator(
            num_classes=num_classes,
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
        
    def train_step(self, real_ecg: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        batch_size = real_ecg.size(0)
        valid = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)
        
        self.optimizer_G.zero_grad()
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        gen_ecg = self.generator(noise, labels)
        
        g_loss = self.adversarial_loss(
            self.discriminator(gen_ecg, labels)[0], valid
        )
        
        g_loss.backward()
        self.optimizer_G.step()
        
        self.optimizer_D.zero_grad()
        real_loss = self.adversarial_loss(
            self.discriminator(real_ecg, labels)[0], valid
        )
        fake_loss = self.adversarial_loss(
            self.discriminator(gen_ecg.detach(), labels)[0], fake
        )
        
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        self.optimizer_D.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
        }
    
    def generate(
        self,
        num_samples: int,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.generator.eval()
        with torch.no_grad():
            if labels is None:
                labels = torch.randint(0, self.num_classes, (num_samples,)).to(self.device)
            
            noise = torch.randn(num_samples, self.latent_dim).to(self.device)
            gen_ecg = self.generator(noise, labels)
            
        self.generator.train()
        return gen_ecg, labels
    
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
