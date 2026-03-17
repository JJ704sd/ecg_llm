import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ECGVAEEncoder(nn.Module):
    def __init__(
        self,
        seq_length: int = 5000,
        num_leads: int = 12,
        latent_dim: int = 64,
        hidden_dims: list = [64, 128, 256],
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(num_leads, hidden_dims[0], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            
            nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            
            nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        intermediate_dim = hidden_dims[2] * (seq_length // 8)
        
        self.fc_mu = nn.Linear(intermediate_dim, latent_dim)
        self.fc_logvar = nn.Linear(intermediate_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ECGVAEDecoder(nn.Module):
    def __init__(
        self,
        seq_length: int = 5000,
        num_leads: int = 12,
        latent_dim: int = 64,
        hidden_dims: list = [64, 128, 256],
    ):
        super().__init__()
        self.seq_length = seq_length
        self.num_leads = num_leads
        
        intermediate_dim = hidden_dims[2] * (seq_length // 8)
        
        self.fc = nn.Linear(latent_dim, intermediate_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dims[2], hidden_dims[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            
            nn.ConvTranspose1d(hidden_dims[1], hidden_dims[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            
            nn.ConvTranspose1d(hidden_dims[0], hidden_dims[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            
            nn.Conv1d(hidden_dims[0], num_leads, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), -1, self.seq_length // 8)
        out = self.decoder(h)
        
        if out.size(2) < self.seq_length:
            out = F.interpolate(out, size=self.seq_length, mode='linear', align_corners=False)
        elif out.size(2) > self.seq_length:
            out = out[:, :, :self.seq_length]
            
        return out


class ECGVAE(nn.Module):
    def __init__(
        self,
        seq_length: int = 5000,
        num_leads: int = 12,
        latent_dim: int = 64,
        hidden_dims: list = [64, 128, 256],
    ):
        super().__init__()
        
        self.encoder = ECGVAEEncoder(
            seq_length=seq_length,
            num_leads=num_leads,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        
        self.decoder = ECGVAEDecoder(
            seq_length=seq_length,
            num_leads=num_leads,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def generate(self, num_samples: int, device: str = "cuda") -> torch.Tensor:
        z = torch.randn(num_samples, self.decoder.fc.in_features).to(device)
        return self.decode(z)


class ECGVAETrainer:
    def __init__(
        self,
        model: ECGVAE,
        device: str = "cuda",
        lr: float = 1e-4,
        beta: float = 1.0,
    ):
        self.model = model
        self.device = device
        self.beta = beta
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    def vae_loss(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        x = x.to(self.device)
        
        recon, mu, logvar = self.model(x)
        
        total_loss, recon_loss, kl_loss = self.vae_loss(recon, x, mu, logvar)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
        }
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class ConditionalECGVAE(nn.Module):
    def __init__(
        self,
        num_classes: int = 12,
        seq_length: int = 5000,
        num_leads: int = 12,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        self.encoder = nn.Sequential(
            nn.Conv1d(num_leads + num_classes, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        intermediate_dim = 256 * (seq_length // 8)
        self.fc_mu = nn.Linear(intermediate_dim, latent_dim)
        self.fc_logvar = nn.Linear(intermediate_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, intermediate_dim),
            nn.Unflatten(1, (256, seq_length // 8)),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, num_leads, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if labels is not None:
            label_onehot = F.one_hot(labels, self.num_classes).float()
            label_onehot = label_onehot.unsqueeze(-1).expand(-1, -1, x.size(2))
            x = torch.cat([x, label_onehot], dim=1)
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        if labels is not None:
            label_onehot = F.one_hot(labels, self.num_classes).float()
            z = torch.cat([z, label_onehot], dim=1)
        
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def generate(
        self,
        labels: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        label_onehot = F.one_hot(labels, self.num_classes).float()
        z = torch.randn(num_samples, self.fc_mu.out_features).to(labels.device)
        z = torch.cat([z, label_onehot], dim=1)
        return self.decoder(z)
