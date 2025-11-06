#!/usr/bin/env python3
"""
Visualize SRCG graph evolution across training epochs.

Generates adjacency matrix heatmaps showing how the graph structure
self-organizes during training, providing visual evidence of self-improvement.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

from srcg.model import SRCG

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
SAVE_DIR = Path("graph_visuals")
CHECKPOINTS = [
    "outputs/checkpoint_epoch_10.pt",
    "outputs/checkpoint_epoch_20.pt",
    "outputs/checkpoint_epoch_30.pt",
    "outputs/checkpoint_epoch_40.pt",
    "outputs/checkpoint_epoch_50.pt",
]
SAVE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------

def load_model(checkpoint_path, config=None):
    """Load SRCG model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Get model config from checkpoint or use defaults
    if config is None:
        config = ckpt.get('config', {}).get('SRCG', {})
    
    # Create model with same architecture
    model = SRCG(
        input_dim=config.get('input_dim', 32),
        output_dim=config.get('output_dim', 1),
        num_nodes=config.get('num_nodes', 100),
        node_dim=config.get('dim', 128),
        reasoning_steps=config.get('reasoning_steps', 20),
        alpha_damping=config.get('alpha_damping', 0.5),
        w_max=config.get('w_max', 0.1),
        prune_threshold=config.get('prune_threshold', 0.02),
        add_threshold=config.get('add_threshold', 0.8),
        max_new_edges=config.get('max_new_edges_per_episode', 10),
        eta_w=config.get('lr_edges', 0.01),
        lambda_energy=config.get('loss_weights', {}).get('energy', 0.1),
        lambda_inst=config.get('loss_weights', {}).get('instability', 0.05),
    )
    
    # Load state dict (may not have A if it's not saved)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    
    return model


def plot_adjacency(A, epoch, num_edges, save_path=None):
    """Plot adjacency matrix as heatmap."""
    A_np = A.abs().cpu().numpy() if torch.is_tensor(A) else np.abs(A)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Absolute values (magnitude)
    im1 = axes[0].imshow(A_np, cmap="inferno", interpolation="nearest", aspect="auto")
    axes[0].set_title(f"Adjacency Magnitude | Epoch {epoch}\n(Edges: {num_edges})", fontsize=12)
    axes[0].set_xlabel("Source node j ->")
    axes[0].set_ylabel("Target node i")
    plt.colorbar(im1, ax=axes[0], label="|Weight|")
    
    # Plot 2: Sparsity pattern (binary)
    A_binary = (A_np > 1e-6).astype(float)
    im2 = axes[1].imshow(A_binary, cmap="Greys", interpolation="nearest", aspect="auto")
    axes[1].set_title(f"Sparsity Pattern | Epoch {epoch}\n(Active connections)", fontsize=12)
    axes[1].set_xlabel("Source node j ->")
    axes[1].set_ylabel("Target node i")
    plt.colorbar(im2, ax=axes[1], label="Connected (1) / Disconnected (0)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved {save_path}")
    
    plt.close()


def compute_graph_stats(A):
    """Compute graph statistics."""
    A_np = A.abs().cpu().numpy() if torch.is_tensor(A) else np.abs(A)
    
    num_edges = int((A_np > 1e-6).sum())
    total_possible = A_np.size
    density = num_edges / total_possible if total_possible > 0 else 0
    avg_weight = A_np[A_np > 1e-6].mean() if num_edges > 0 else 0
    
    return {
        'num_edges': num_edges,
        'density': density,
        'avg_weight': avg_weight,
        'max_weight': A_np.max(),
    }


def plot_evolution_comparison(checkpoints_data):
    """Create side-by-side comparison of all epochs."""
    n_epochs = len(checkpoints_data)
    fig, axes = plt.subplots(2, n_epochs, figsize=(5*n_epochs, 10))
    
    if n_epochs == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (epoch, A, stats) in enumerate(checkpoints_data):
        A_np = A.abs().cpu().numpy() if torch.is_tensor(A) else np.abs(A)
        
        # Top row: magnitude
        im1 = axes[0, idx].imshow(A_np, cmap="inferno", interpolation="nearest", aspect="auto")
        axes[0, idx].set_title(f"Epoch {epoch}\nEdges: {stats['num_edges']}", fontsize=10)
        if idx == 0:
            axes[0, idx].set_ylabel("Magnitude")
        
        # Bottom row: sparsity
        A_binary = (A_np > 1e-6).astype(float)
        axes[1, idx].imshow(A_binary, cmap="Greys", interpolation="nearest", aspect="auto")
        if idx == 0:
            axes[1, idx].set_ylabel("Sparsity")
    
    plt.tight_layout()
    save_path = SAVE_DIR / "graph_evolution_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved comparison plot: {save_path}")
    plt.close()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    print("[GRAPH VISUALIZATION] Generating adjacency matrix visualizations...")
    print(f"[INFO] Save directory: {SAVE_DIR}")
    
    checkpoints_data = []
    available_checkpoints = []
    
    for ckpt_path in CHECKPOINTS:
        if not os.path.exists(ckpt_path):
            print(f"[!] Missing checkpoint: {ckpt_path}")
            continue
        
        # Extract epoch number
        try:
            epoch = int(Path(ckpt_path).stem.split("_")[-1])
        except:
            print(f"[!] Could not parse epoch from: {ckpt_path}")
            continue
        
        print(f"\n[LOADING] Epoch {epoch} from {ckpt_path}")
        
        try:
            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location="cpu")
            config = ckpt.get('config', {})
            
            # Create model
            model = load_model(ckpt_path, config.get('SRCG', {}))
            
            # Get adjacency matrix
            A = model.A.detach().cpu()
            
            # Compute stats
            stats = compute_graph_stats(A)
            print(f"  Edges: {stats['num_edges']} | Density: {stats['density']:.4f} | Avg weight: {stats['avg_weight']:.6f}")
            
            # Plot individual
            save_path = SAVE_DIR / f"graph_epoch_{epoch:02d}.png"
            plot_adjacency(A, epoch, stats['num_edges'], save_path)
            
            # Store for comparison
            checkpoints_data.append((epoch, A, stats))
            available_checkpoints.append(epoch)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {ckpt_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison plot if we have multiple epochs
    if len(checkpoints_data) > 1:
        print(f"\n[COMPARISON] Creating side-by-side comparison...")
        checkpoints_data.sort(key=lambda x: x[0])  # Sort by epoch
        plot_evolution_comparison(checkpoints_data)
    
    # Summary
    print(f"\n[OK] Visualization complete!")
    print(f"  Generated {len(checkpoints_data)} graph visualizations")
    print(f"  Epochs: {sorted(available_checkpoints)}")
    print(f"  Output directory: {SAVE_DIR}")
    
    print("\n[INTERPRETATION]")
    print("  - Bright regions = strong connections")
    print("  - Dark regions = weak/no connections")
    print("  - Structured patterns = self-organized motifs")
    print("  - Increasing sparsity + structure = self-improvement")

