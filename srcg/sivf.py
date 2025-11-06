"""
Self-Improvement Verification Framework (SIVF) Metrics

Tracks metrics to validate self-improvement characteristics:
- Graph entropy (structure organization)
- Motif reuse (pattern learning)
- Structural efficiency (reward per edge)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from pathlib import Path
import json


def graph_entropy(A: torch.Tensor) -> float:
    """
    Shannon entropy of absolute edge weights.
    Higher entropy = more random/disorganized graph.
    Lower entropy = more structured/organized graph.
    """
    a = torch.abs(A).flatten()
    p = a / (a.sum() + 1e-9)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * p.log()).sum().item())


def motif_reuse(A_prev: torch.Tensor, A_now: torch.Tensor) -> float:
    """
    Cosine similarity between consecutive adjacency matrices.
    Measures how much the graph structure is being reused/refined.
    Higher = more motif reuse (learning patterns).
    """
    v1 = A_prev.flatten()
    v2 = A_now.flatten()
    num = (v1 * v2).sum()
    den = (v1.norm() * v2.norm() + 1e-9)
    return float((num / den).item())


def structural_efficiency(reward: float, num_edges: int) -> float:
    """
    Reward per edge - measures intelligence per connection.
    Higher = more efficient use of graph structure.
    """
    return reward / (num_edges + 1e-6)


class SIVFLogger:
    """Tracks SIVF metrics during training."""
    
    def __init__(self, save_dir: Optional[Path] = None, enabled: bool = True):
        self.enabled = enabled
        if save_dir is None:
            save_dir = Path("sivf_logs")
        self.save_dir = Path(save_dir)
        if enabled:
            self.save_dir.mkdir(exist_ok=True)
        
        self.logs: Dict[str, list] = {
            "epoch": [],
            "reward": [],
            "loss": [],
            "task_loss": [],
            "entropy": [],
            "reuse": [],
            "edges": [],
            "efficiency": [],
            "energy": [],
            "instability": [],
        }
        
        self.A_prev: Optional[torch.Tensor] = None
    
    def log_epoch(
        self,
        epoch: int,
        avg_loss: float,
        avg_task_loss: float,
        avg_reward: float,
        avg_energy: float,
        avg_instability: float,
        A_now: torch.Tensor,
    ) -> Dict[str, float]:
        """Log metrics for one epoch."""
        if not self.enabled:
            return {}
        
        # Compute SIVF metrics
        ent = graph_entropy(A_now)
        
        if self.A_prev is not None:
            reuse = motif_reuse(self.A_prev, A_now)
        else:
            reuse = 0.0
        
        num_edges = int((A_now.abs() > 1e-6).sum().item())
        eff = structural_efficiency(avg_reward, num_edges)
        
        # Store
        self.logs["epoch"].append(epoch)
        self.logs["loss"].append(avg_loss)
        self.logs["task_loss"].append(avg_task_loss)
        self.logs["reward"].append(avg_reward)
        self.logs["energy"].append(avg_energy)
        self.logs["instability"].append(avg_instability)
        self.logs["entropy"].append(ent)
        self.logs["reuse"].append(reuse)
        self.logs["edges"].append(num_edges)
        self.logs["efficiency"].append(eff)
        
        # Update previous adjacency
        self.A_prev = A_now.detach().clone()
        
        return {
            'entropy': ent,
            'reuse': reuse,
            'edges': num_edges,
            'efficiency': eff,
        }
    
    def save_checkpoint(self, epoch: int, eval_every: int = 5):
        """Save metrics checkpoint."""
        if not self.enabled or len(self.logs["epoch"]) == 0:
            return
        
        if (epoch + 1) % eval_every == 0:
            path = self.save_dir / f"metrics_epoch_{epoch+1}.json"
            # Save last eval_every epochs
            recent_logs = {
                k: v[-eval_every:] if len(v) >= eval_every else v
                for k, v in self.logs.items()
            }
            with open(path, "w") as f:
                json.dump(recent_logs, f, indent=2)
    
    def plot_trends(self, filename: str = "sivf_trends.png"):
        """Plot all SIVF trends."""
        if not self.enabled or len(self.logs["epoch"]) == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Self-Improvement Verification Framework (SIVF) Trends", fontsize=14)
        
        epochs = self.logs["epoch"]
        
        # Row 1: Core metrics
        axes[0, 0].plot(epochs, self.logs["reward"], 'g-', label='Reward', linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].set_title("Reward (UP = good)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].plot(epochs, self.logs["loss"], 'r-', label='Total Loss', linewidth=2)
        axes[0, 1].plot(epochs, self.logs["task_loss"], 'orange', label='Task Loss', linewidth=2)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Loss (DOWN = good)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        axes[0, 2].plot(epochs, self.logs["entropy"], 'b-', label='Entropy', linewidth=2)
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Entropy")
        axes[0, 2].set_title("Graph Entropy (DOWN = structured)")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        
        # Row 2: Structure metrics
        axes[1, 0].plot(epochs, self.logs["reuse"], 'purple', label='Motif Reuse', linewidth=2)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Cosine Similarity")
        axes[1, 0].set_title("Motif Reuse (UP = patterns)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].plot(epochs, self.logs["edges"], 'brown', label='# Edges', linewidth=2)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Number of Edges")
        axes[1, 1].set_title("Active Edges")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        axes[1, 2].plot(epochs, self.logs["efficiency"], 'teal', label='Efficiency', linewidth=2)
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Reward / Edge")
        axes[1, 2].set_title("Structural Efficiency (UP = smart)")
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        
        plt.tight_layout()
        path = self.save_dir / filename
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SIVF] Trends plot saved to {path}")
    
    def save_final_logs(self):
        """Save complete logs."""
        if not self.enabled or len(self.logs["epoch"]) == 0:
            return
        
        path = self.save_dir / "sivf_complete_logs.json"
        with open(path, "w") as f:
            json.dump(self.logs, f, indent=2)
        print(f"[SIVF] Complete logs saved to {path}")

