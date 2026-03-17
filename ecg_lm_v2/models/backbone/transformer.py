import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ECGPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        time_positions = torch.arange(max_len).unsqueeze(1).float()
        time_positions = time_positions / max_len
        
        phase_positions = torch.arange(max_len).unsqueeze(1).float()
        phase_positions = (phase_positions % 250) / 250.0
        
        position_emb = torch.zeros(1, max_len, d_model)
        
        for i in range(0, d_model, 4):
            div_term = math.exp(torch.tensor(i // 4) * (-math.log(10000.0) / (d_model // 4)))
            position_emb[0, :, i] = torch.sin(time_positions.squeeze() * div_term)
            position_emb[0, :, i+1] = torch.cos(time_positions.squeeze() * div_term)
            position_emb[0, :, i+2] = torch.sin(phase_positions.squeeze() * div_term)
            position_emb[0, :, i+3] = torch.cos(phase_positions.squeeze() * div_term)
            
        self.register_buffer('position_emb', position_emb)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.position_emb[:, :x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embeddings(positions)
        x = x + position_emb
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.out_linear(attn_output)
        
        return output, attn_weights


class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x
        x = self.norm1(x)
        
        attn_output, attn_weights = self.attention(x, x, x, mask)
        
        x = residual + self.dropout(attn_output)
        
        residual = x
        x = self.norm2(x)
        x = residual + self.feed_forward(x)
        
        return x, attn_weights


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class CNNFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        conv_channels: list = [64, 128, 256],
        kernel_sizes: list = [5, 3, 3],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_conv = ConvBlock(
            input_channels, 
            conv_channels[0], 
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2,
            dropout=dropout
        )
        
        self.conv_layers = nn.ModuleList()
        for i in range(1, len(conv_channels)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        conv_channels[i-1], 
                        conv_channels[i], 
                        kernel_size=kernel_sizes[i],
                        padding=kernel_sizes[i] // 2,
                        stride=2
                    ),
                    nn.BatchNorm1d(conv_channels[i]),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
        
        self.conv_channels = conv_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            
        return x


class ECGTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 12,
        d_ff: int = 1024,
        conv_channels: list = [64, 128, 256],
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_channels = input_channels
        
        self.cnn_extractor = CNNFeatureExtractor(
            input_channels=input_channels,
            conv_channels=conv_channels,
            dropout=dropout
        )
        
        self.input_projection = nn.Linear(conv_channels[-1], d_model)
        
        if use_positional_encoding:
            self.pos_encoder = ECGPositionalEncoding(d_model, max_seq_len, dropout)
        else:
            self.pos_encoder = None
            
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        x = self.cnn_extractor(x)
        
        x = x.transpose(1, 2)
        
        x = self.input_projection(x)
        
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
            
        attn_weights_list = []
        
        for block in self.transformer_blocks:
            x, attn_weights = block(x, mask)
            attn_weights_list.append(attn_weights)
            
        x = self.norm(x)
        
        return x, attn_weights_list


class LeadAttention(nn.Module):
    def __init__(self, d_model: int, num_leads: int = 12):
        super().__init__()
        
        self.lead_attention = nn.Parameter(torch.ones(num_leads))
        
        self.lead_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead_weights = F.softmax(self.lead_attention, dim=0)
        
        lead_weights = lead_weights.unsqueeze(0).unsqueeze(-1)
        
        weighted = x * lead_weights
        
        output = self.lead_projection(weighted)
        
        return output, lead_weights


class CrossLeadTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_leads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.lead_attention = LeadAttention(d_model, num_leads)
        
        self.cross_lead_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 2, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, lead_weights = self.lead_attention(x)
        
        for block in self.cross_lead_blocks:
            x, _ = block(x)
            
        return x, lead_weights


class MaskedSignalModelingHead(nn.Module):
    def __init__(self, d_model: int, input_channels: int = 12):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, input_channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class PretrainECGTransformer(nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 12,
        d_ff: int = 1024,
        conv_channels: list = [64, 128, 256],
        dropout: float = 0.1,
        masked_ratio: float = 0.15,
    ):
        super().__init__()
        self.masked_ratio = masked_ratio
        
        self.encoder = ECGTransformerEncoder(
            input_channels=input_channels,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            conv_channels=conv_channels,
            dropout=dropout,
        )
        
        self.masked_head = MaskedSignalModelingHead(d_model, input_channels)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        encoder_output, attention_weights = self.encoder(x)
        
        pooled = encoder_output.mean(dim=1)
        
        logits = self.classifier(pooled)
        
        outputs = {
            'logits': logits,
            'encoder_output': encoder_output,
            'attention_weights': attention_weights,
        }
        
        if return_features:
            outputs['pooled_features'] = pooled
            
        return outputs
    
    def generate_mask(
        self, 
        seq_len: int, 
        batch_size: int, 
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.rand(batch_size, seq_len, device=device) > self.masked_ratio
        
        return mask
    
    def masked_signal_modeling(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, seq_len = x.shape
        
        mask = self.generate_mask(seq_len, batch_size, x.device)
        
        x_masked = x.clone()
        x_masked = x_masked * mask.unsqueeze(1).float()
        
        encoder_output, _ = self.encoder(x_masked)
        
        predicted_signal = self.masked_head(encoder_output)
        
        return predicted_signal, x, mask


class ContrastiveECGModel(nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 12,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        
        self.encoder = ECGTransformerEncoder(
            input_channels=input_channels,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_output, _ = self.encoder(x)
        
        pooled = encoder_output.mean(dim=1)
        
        projected = self.projection(pooled)
        
        projected = F.normalize(projected, dim=-1)
        
        return projected, encoder_output
    
    def contrastive_loss(
        self, 
        projected1: torch.Tensor, 
        projected2: torch.Tensor
    ) -> torch.Tensor:
        batch_size = projected1.size(0)
        
        projected = torch.cat([projected1, projected2], dim=0)
        
        similarity = torch.matmul(projected, projected.T) / self.temperature
        
        labels = torch.arange(batch_size, device=projected1.device)
        
        loss = F.cross_entropy(similarity[:batch_size], labels) + \
               F.cross_entropy(similarity[batch_size:], labels)
        
        return loss / 2


class ECGTransformerWithPretrain:
    def __init__(
        self,
        model: PretrainECGTransformer,
        device: str = "cuda",
        lr: float = 1e-4,
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
    def train_step(
        self, 
        x: torch.Tensor,
        task: str = "msm"
    ) -> Dict[str, float]:
        self.model.train()
        x = x.to(self.device)
        
        if task == "msm":
            predicted, target, mask = self.model.masked_signal_modeling(x)
            
            loss = F.mse_loss(predicted[:, :, :target.shape[2]], target, reduction='none')
            loss = (loss * mask.unsqueeze(1)).sum() / mask.sum()
            
        elif task == "contrastive":
            projected1, _ = self.model(x)
            
            x_shuffled = x[torch.randperm(x.size(0))]
            projected2, _ = self.model(x_shuffled)
            
            contrastive_module = ContrastiveECGModel(
                input_channels=x.shape[1],
                d_model=self.model.encoder.d_model
            ).to(self.device)
            loss = contrastive_module.contrastive_loss(projected1, projected2)
            
        else:
            outputs = self.model(x)
            loss = F.cross_entropy(outputs['logits'], torch.randint(0, 2, (x.size(0),)).to(self.device))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def pretrain(
        self,
        dataloader,
        num_epochs: int = 100,
        tasks: List[str] = ["msm", "contrastive"],
    ):
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                    
                task = tasks[epoch % len(tasks)]
                loss_dict = self.train_step(x, task)
                total_loss += loss_dict['loss']
            
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
