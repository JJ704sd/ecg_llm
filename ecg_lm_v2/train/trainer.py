import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
import time
from pathlib import Path
import json
from tqdm import tqdm


class ECGTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.model.to(self.device)
        
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-5
            )
        else:
            self.optimizer = optimizer
            
        if scheduler is None:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=100,
                eta_min=1e-6
            )
        else:
            self.scheduler = scheduler
            
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
            
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            
            if isinstance(outputs, dict):
                loss = self.criterion(
                    outputs['classification_logits'],
                    targets
                )
            else:
                loss = self.criterion(outputs, targets)
                
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = outputs['classification_logits'].max(1) if isinstance(outputs, dict) else outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
            
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'acc': epoch_acc}
    
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                if isinstance(outputs, dict):
                    loss = self.criterion(
                        outputs['classification_logits'],
                        targets
                    )
                else:
                    loss = self.criterion(outputs, targets)
                    
                running_loss += loss.item()
                
                _, predicted = outputs['classification_logits'].max(1) if isinstance(outputs, dict) else outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return {'loss': val_loss, 'acc': val_acc}
    
    def save_checkpoint(self, epoch: int, filename: str = "checkpoint.pth"):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        
        return checkpoint['epoch']
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 15,
        save_every: int = 5,
    ):
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['loss'])
            self.train_accs.append(train_metrics['acc'])
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.2f}%")
            
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.val_losses.append(val_metrics['loss'])
                self.val_accs.append(val_metrics['acc'])
                
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.2f}%")
                
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    
                    self.save_checkpoint(epoch, filename="best_model.pth")
                    print("New best model saved!")
                else:
                    patience_counter += 1
                    
            self.scheduler.step()
            
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, filename=f"checkpoint_epoch_{epoch}.pth")
                
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
                
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        self._save_training_history()
        
    def _save_training_history(self):
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }
        
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
            
        print(f"Training history saved to {history_path}")


class ECGPretrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        masked_ratio: float = 0.15,
    ):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.masked_ratio = masked_ratio
        
        self.model.to(self.device)
        
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-5
            )
        else:
            self.optimizer = optimizer
            
        if scheduler is None:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=100,
                eta_min=1e-6
            )
        else:
            self.scheduler = scheduler
            
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_loss = float('inf')
        
    def masked_signal_modeling(self, x: torch.Tensor):
        batch_size, num_leads, seq_len = x.shape
        
        mask = torch.rand(batch_size, seq_len) > self.masked_ratio
        mask = mask.to(x.device)
        
        masked_x = x.clone()
        masked_x = masked_x * mask.unsqueeze(1).float()
        
        return masked_x, mask
    
    def pretrain_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Pretrain Epoch {epoch}")
        
        for batch_idx, (inputs, _) in enumerate(pbar):
            inputs = inputs.to(self.device)
            
            masked_inputs, mask = self.masked_signal_modeling(inputs)
            
            self.optimizer.zero_grad()
            
            encoder_output, _ = self.model.encoder(masked_inputs)
            
            pooled = encoder_output.mean(dim=1)
            
            reconstruction = self.model.classifier_head(pooled)
            
            loss = nn.functional.mse_loss(reconstruction, inputs.mean(dim=1))
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            
            pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
            
        epoch_loss = running_loss / len(self.train_loader)
        
        return {'loss': epoch_loss}
    
    def pretrain(
        self,
        num_epochs: int,
        save_every: int = 10,
    ):
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Pretrain Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            train_metrics = self.pretrain_epoch(epoch)
            
            print(f"Pretrain Loss: {train_metrics['loss']:.4f}")
            
            if train_metrics['loss'] < self.best_loss:
                self.best_loss = train_metrics['loss']
                self._save_checkpoint(epoch, filename="best_pretrain.pth")
                print("New best pretrain model saved!")
                
            if epoch % save_every == 0:
                self._save_checkpoint(epoch, filename=f"pretrain_checkpoint_epoch_{epoch}.pth")
                
            self.scheduler.step()
            
        print("\nPretraining completed!")
        self._save_checkpoint(num_epochs, filename="final_pretrain.pth")
        
    def _save_checkpoint(self, epoch: int, filename: str):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
