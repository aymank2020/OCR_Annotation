"""
Training Module with VideoX Support - FIXED
Complete implementation for Atlas Action Recognition
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # Updated import
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
import json


class ActionRecognitionTrainer:
    """
    Trainer for VideoX/CLIP Action Recognition
    
    Features:
    - Mixed precision training
    - Gradient accumulation
    - Checkpointing
    - Learning rate scheduling
    - Multi-task loss (classification + boundary)
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 config: Dict,
                 device: str = 'cuda'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training config
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.grad_accum_steps = config['training']['gradient_accumulation_steps']
        self.use_fp16 = config['training']['use_fp16']
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Scheduler (Cosine with warmup)
        warmup_steps = len(train_loader) * 5  # 5 epochs warmup
        total_steps = len(train_loader) * self.num_epochs
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_steps // 3,
            T_mult=1,
            eta_min=self.learning_rate * 0.01
        )
        
        # Loss functions - FIXED: Use BCEWithLogitsLoss instead of BCELoss
        self.action_criterion = nn.CrossEntropyLoss()
        self.boundary_criterion = nn.BCEWithLogitsLoss()  # ✅ FIXED
        
        # Mixed precision - Updated for PyTorch 2.x
        self.scaler = GradScaler('cuda') if self.use_fp16 and device == 'cuda' else None
        
        # Checkpointing
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.save_every = config['training']['save_every']
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_history = []
        self.val_history = []
        
        # Backbone unfreezing
        self.freeze_backbone = config['training'].get('freeze_backbone', True)
        self.unfreeze_after_epoch = config['training'].get('unfreeze_after_epoch', 20)
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Mixed precision: {self.use_fp16}")
        print(f"  Gradient accumulation: {self.grad_accum_steps}")
        print(f"  Freeze backbone: {self.freeze_backbone}")
    
    def train_epoch(self) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}")
        
        for i, (frames, actions, metadata) in enumerate(pbar):
            frames = frames.to(self.device)
            actions = actions.to(self.device)
            
            # Forward pass
            if self.use_fp16 and self.device == 'cuda':
                with autocast('cuda'):
                    loss = self._compute_loss(frames, actions)
                    loss = loss / self.grad_accum_steps
                
                self.scaler.scale(loss).backward()
            else:
                loss = self._compute_loss(frames, actions)
                loss = loss / self.grad_accum_steps
                loss.backward()
            
            # Gradient accumulation
            if (i + 1) % self.grad_accum_steps == 0:
                if self.use_fp16 and self.device == 'cuda':
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            total_loss += loss.item() * self.grad_accum_steps
            pbar.set_postfix({'loss': f"{loss.item() * self.grad_accum_steps:.4f}"})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def _compute_loss(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-task loss
        
        Loss = action_loss + boundary_loss
        """
        batch_size, num_frames = frames.size(0), frames.size(1)
        
        # Forward pass
        outputs = self.model(frames)
        
        # Action classification loss
        action_logits = outputs['action_logits']  # [B, T, num_classes]
        
        # Expand actions to all frames
        actions_expanded = actions.unsqueeze(1).expand(-1, num_frames)  # [B, T]
        
        action_loss = self.action_criterion(
            action_logits.reshape(-1, action_logits.size(-1)),
            actions_expanded.reshape(-1)
        )
        
        # Boundary loss - FIXED: Now outputs are logits, not probabilities
        boundary_start = outputs['boundary_start'].squeeze(-1)  # [B, T]
        boundary_end = outputs['boundary_end'].squeeze(-1)      # [B, T]
        
        # Create targets
        start_target = torch.zeros_like(boundary_start)
        start_target[:, 0] = 1.0  # Start at first frame
        
        end_target = torch.zeros_like(boundary_end)
        end_target[:, -1] = 1.0  # End at last frame
        
        # BCEWithLogitsLoss expects logits (no sigmoid applied)
        boundary_loss = (
            self.boundary_criterion(boundary_start, start_target) +
            self.boundary_criterion(boundary_end, end_target)
        ) / 2
        
        # Combined loss
        total_loss = action_loss + 0.5 * boundary_loss
        
        return total_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        
        for frames, actions, metadata in self.val_loader:
            frames = frames.to(self.device)
            actions = actions.to(self.device)
            
            loss = self._compute_loss(frames, actions)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, resume_from: Optional[str] = None):
        """
        Train the model
        
        Args:
            resume_from: Path to checkpoint to resume from
        """
        # Resume from checkpoint
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print(f"\nTraining for {self.num_epochs} epochs...")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Unfreeze backbone after certain epochs
            if self.freeze_backbone and epoch == self.unfreeze_after_epoch:
                print(f"\n🔓 Unfreezing backbone at epoch {epoch + 1}")
                self.model.unfreeze_backbone()
                self.freeze_backbone = False
            
            # Train
            train_loss = self.train_epoch()
            self.train_history.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_history.append(val_loss)
            
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint('best.pth')
                print("✅ Saved best model")
            
            # Save latest
            self.save_checkpoint('latest.pth')
        
        print("\n✅ Training complete!")
        self._save_history()
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def _save_history(self):
        """Save training history"""
        history = {
            'train_loss': self.train_history,
            'val_loss': self.val_history,
            'best_loss': self.best_loss
        }
        
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    print("Trainer module ready!")
    print("Import this module to train your model.")
