#!/usr/bin/env python3
"""
Main training script for SRCG
"""

import argparse
import yaml
from pathlib import Path
import torch

from srcg.model import SRCG
from srcg.train import Trainer, load_config
from srcg.data import create_dataloaders


def main():
    parser = argparse.ArgumentParser(description='Train SRCG model')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu), auto-detect if not specified'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset name: "synthetic" or "hierarchical" (overrides config)'
    )
    parser.add_argument(
        '--progressive',
        action='store_true',
        help='Enable progressive difficulty (level increases every 10 epochs)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get config values
    srcg_cfg = config.get('SRCG', {})
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    
    # Dataset selection
    dataset_name = args.dataset or data_cfg.get('name', 'synthetic')
    progressive = args.progressive or data_cfg.get('progressive', False)
    input_dim = data_cfg.get('input_dim', 32)
    
    print(f"[DATA] Using dataset: {dataset_name}")
    if progressive:
        print(f"[DATA] Progressive difficulty enabled (level increases every 10 epochs)")
    
    # Create initial dataloaders
    print("Creating datasets...")
    train_loader, val_loader = create_dataloaders(
        dataset_name=dataset_name,
        train_size=data_cfg.get('train_size', 800),
        val_size=data_cfg.get('val_size', 200),
        input_dim=input_dim,
        output_dim=1,
        batch_size=train_cfg.get('batch_size', 8),
        level=1,  # Start at level 1
    )
    
    # Create model
    print("Creating model...")
    model = SRCG(
        input_dim=input_dim,
        output_dim=1,
        num_nodes=srcg_cfg.get('num_nodes', 100),
        node_dim=srcg_cfg.get('dim', 128),
        reasoning_steps=srcg_cfg.get('reasoning_steps', 20),
        alpha_damping=srcg_cfg.get('alpha_damping', 0.5),
        w_max=srcg_cfg.get('w_max', 0.1),
        prune_threshold=srcg_cfg.get('prune_threshold', 0.02),
        add_threshold=srcg_cfg.get('add_threshold', 0.8),
        max_new_edges=srcg_cfg.get('max_new_edges_per_episode', 10),
        eta_w=float(srcg_cfg.get('lr_edges', 0.01)),
        lambda_energy=srcg_cfg.get('loss_weights', {}).get('energy', 0.1),
        lambda_inst=srcg_cfg.get('loss_weights', {}).get('instability', 0.05),
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Initial graph has {model.get_num_edges()} edges")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        log_dir=output_dir,
        dataset_name=dataset_name,
        progressive=progressive,
        input_dim=input_dim,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    final_path = output_dir / 'model_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': trainer.metrics,
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")


if __name__ == '__main__':
    main()

