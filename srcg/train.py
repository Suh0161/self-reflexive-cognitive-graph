"""
Training script for SRCG
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import yaml
from pathlib import Path
import logging
from tqdm import tqdm

from .model import SRCG
from .sivf import SIVFLogger, graph_entropy, motif_reuse, structural_efficiency


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: Optional[str] = None):
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log') if log_dir else logging.StreamHandler(),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class Trainer:
    """SRCG Trainer"""
    
    def __init__(
        self,
        model: SRCG,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[dict] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        log_dir: Optional[Path] = None,
        dataset_name: str = 'synthetic',
        progressive: bool = False,
        input_dim: int = 32,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_dir = log_dir
        self.dataset_name = dataset_name
        self.progressive = progressive
        self.input_dim = input_dim
        
        # Training config
        self.config = config or {}
        train_cfg = self.config.get('training', {})
        srcg_cfg = self.config.get('SRCG', {})
        
        self.epochs = train_cfg.get('epochs', 50)
        self.loss_weights = srcg_cfg.get('loss_weights', {
            'task': 1.0,
            'energy': 0.1,
            'instability': 0.05
        })
        
        # Optimizer (only for node/head params, not A)
        lr_nodes = srcg_cfg.get('lr_nodes', 1e-3)
        # Convert to float if it's a string (YAML sometimes reads scientific notation as string)
        if isinstance(lr_nodes, str):
            lr_nodes = float(lr_nodes)
        self.optimizer = optim.Adam(
            [
                {'params': self.model.encoder.parameters()},
                {'params': self.model.node_transform.parameters()},
                {'params': self.model.output_head.parameters()},
            ],
            lr=lr_nodes
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'train_task_loss': [],
            'train_energy': [],
            'train_instability': [],
            'val_loss': [],
            'num_edges': [],
        }
        
        # SIVF logging (optional)
        enable_sivf = srcg_cfg.get('enable_sivf', False)
        sivf_dir = log_dir / 'sivf_logs' if log_dir else Path('sivf_logs')
        self.sivf_logger = SIVFLogger(save_dir=sivf_dir, enabled=enable_sivf)
        
        self.logger = setup_logging(log_dir)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_task_loss = 0.0
        total_reward = 0.0
        total_energy = 0.0
        total_instability = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            y_pred, info = self.model(inputs)
            
            # Task loss
            task_loss = self.criterion(y_pred, targets)
            
            # Structure costs
            C_energy, C_inst = self.model.compute_structure_costs(
                info['H_final'],
                info['H_prev_step'],
                info['A']
            )
            
            # Total loss
            loss = (
                self.loss_weights['task'] * task_loss +
                self.loss_weights['energy'] * C_energy +
                self.loss_weights['instability'] * C_inst
            )
            
            # Backward pass (only for node/head params)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Structural plasticity update (no grad)
            with torch.no_grad():
                R_k = self.model.compute_reward(task_loss, C_energy, C_inst)
                self.model.update_structure(info['H_final'], R_k)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_task_loss += task_loss.item()
                total_reward += R_k
                total_energy += C_energy.item()
                total_instability += C_inst.item()
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'task': f'{task_loss.item():.4f}',
                'edges': self.model.get_num_edges()
            })
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_task_loss = total_task_loss / num_batches
        avg_reward = total_reward / num_batches
        avg_energy = total_energy / num_batches
        avg_instability = total_instability / num_batches
        num_edges = self.model.get_num_edges()
        
        # Log SIVF metrics
        sivf_metrics = self.sivf_logger.log_epoch(
            epoch,
            avg_loss,
            avg_task_loss,
            avg_reward,
            avg_energy,
            avg_instability,
            self.model.A.detach(),
        )
        
        return {
            'loss': avg_loss,
            'task_loss': avg_task_loss,
            'reward': avg_reward,
            'energy': avg_energy,
            'instability': avg_instability,
            'num_edges': num_edges,
            **sivf_metrics,  # Include SIVF metrics if enabled
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                y_pred, info = self.model(inputs)
                loss = self.criterion(y_pred, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'loss': avg_loss}
    
    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Dataset: {self.dataset_name}")
        if self.progressive:
            self.logger.info(f"Progressive difficulty: level increases every 10 epochs")
        self.logger.info(f"Initial edges: {self.model.get_num_edges()}")
        
        for epoch in range(1, self.epochs + 1):
            # Progressive difficulty: update dataset level every 10 epochs
            if self.progressive and self.dataset_name == 'hierarchical':
                current_level = min((epoch - 1) // 10 + 1, 4)  # Cap at level 4
                if epoch == 1 or (epoch - 1) % 10 == 0:
                    from .data import create_dataloaders
                    train_cfg = self.config.get('training', {})
                    data_cfg = self.config.get('data', {})
                    self.logger.info(f"[Epoch {epoch}] Switching to difficulty level {current_level}")
                    self.train_loader, self.val_loader = create_dataloaders(
                        dataset_name=self.dataset_name,
                        train_size=data_cfg.get('train_size', 800),
                        val_size=data_cfg.get('val_size', 200),
                        input_dim=self.input_dim,
                        output_dim=1,
                        batch_size=train_cfg.get('batch_size', 8),
                        level=current_level,
                    )
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Log
            log_msg = (
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Task Loss: {train_metrics['task_loss']:.4f} | "
                f"Reward: {train_metrics.get('reward', 0.0):.4f} | "
                f"Energy: {train_metrics['energy']:.4f} | "
                f"Instability: {train_metrics['instability']:.4f} | "
                f"Edges: {train_metrics['num_edges']} | "
                f"Val Loss: {val_metrics.get('loss', 0.0):.4f}"
            )
            
            # Add SIVF metrics if enabled
            if self.sivf_logger.enabled and 'entropy' in train_metrics:
                log_msg += (
                    f" | Entropy: {train_metrics['entropy']:.3f} | "
                    f"Reuse: {train_metrics['reuse']:.3f} | "
                    f"Eff: {train_metrics['efficiency']:.6f}"
                )
            
            self.logger.info(log_msg)
            
            # Save metrics
            self.metrics['train_loss'].append(train_metrics['loss'])
            self.metrics['train_task_loss'].append(train_metrics['task_loss'])
            self.metrics['train_energy'].append(train_metrics['energy'])
            self.metrics['train_instability'].append(train_metrics['instability'])
            self.metrics['num_edges'].append(train_metrics['num_edges'])
            if val_metrics:
                self.metrics['val_loss'].append(val_metrics['loss'])
            
            # Save checkpoint
            if self.log_dir and epoch % 10 == 0:
                self.save_checkpoint(epoch)
            
            # Save SIVF checkpoint
            if self.sivf_logger.enabled:
                self.sivf_logger.save_checkpoint(epoch, eval_every=5)
        
        # Final SIVF plots and logs
        if self.sivf_logger.enabled:
            self.sivf_logger.plot_trends()
            self.sivf_logger.save_final_logs()
            self.logger.info("\n[SIVF] Self-improvement validation complete!")
            self.logger.info("  ✅ Self-improving if: Reward↑, Entropy↓, Reuse↑, Efficiency↑")
            self.logger.info("  ⚠️  Needs tuning if: Metrics plateau or oscillate")
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        if self.log_dir is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'config': self.config,
        }
        
        path = self.log_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

