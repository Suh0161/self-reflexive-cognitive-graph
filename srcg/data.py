"""
SRCG Dataset Utilities

Includes:
- SyntheticDataset: Simple baseline for stability testing
- HierarchicalPatternDataset: Progressive reasoning difficulty for self-improvement validation
"""

import torch
from torch.utils.data import Dataset
import math
import random


# ================================================================
#  SyntheticDataset
#  - Simple nonlinear regression task
#  - Use this first to check that SRCG trains stably
# ================================================================
class SyntheticDataset(Dataset):
    def __init__(self, size=2000, input_dim=32, seed=42):
        self.size = size
        self.input_dim = input_dim
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.rand(self.input_dim, generator=self.rng)
        # Nonlinear mapping (smooth but nontrivial)
        y = (
            torch.sin((x[:16] * math.pi).sum())
            + torch.cos((x[16:] * math.pi / 2).sum())
        ) / 2.0
        y = torch.tensor([y], dtype=torch.float32)
        return x.float(), y.float()


# ================================================================
#  HierarchicalPatternDataset
#  - Progressive reasoning difficulty (sum → alternating sum → parity → logic)
#  - Use this to test SRCG self-improvement and structure adaptation
# ================================================================
class HierarchicalPatternDataset(Dataset):
    def __init__(self, size=2000, input_dim=32, level=1, seed=123):
        self.size = size
        self.input_dim = input_dim
        self.level = level
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.rand(self.input_dim, generator=self.rng)

        if self.level == 1:
            # Simple sum
            y = x.sum()
        elif self.level == 2:
            # Alternating sum: + - + - ...
            y = (x[::2] - x[1::2]).sum()
        elif self.level == 3:
            # Parity: 1 if sum of rounded bits is odd, else 0
            bits = (x > 0.5).float()
            y = bits.sum() % 2
        else:
            # Logical threshold: 1 if > half are > 0.5
            y = torch.tensor([1.0 if (x > 0.5).sum() > self.input_dim / 2 else 0.0])

        y = torch.tensor([float(y)], dtype=torch.float32)
        return x.float(), y.float()


# ================================================================
#  Utility loader
# ================================================================
def load_dataset(name="synthetic", size=2000, input_dim=32, level=1, seed=None):
    """
    Load a dataset by name.
    
    Args:
        name: "synthetic" or "hierarchical"
        size: Number of samples
        input_dim: Input dimension
        level: Difficulty level (used only for hierarchical, 1-4)
        seed: Random seed (None = use default)
    
    Returns:
        Dataset instance
    """
    if seed is None:
        seed = 42 if name == "synthetic" else 123
    
    if name.lower() == "synthetic":
        print(f"[DATA] Loaded SyntheticDataset (size={size}, input_dim={input_dim})")
        return SyntheticDataset(size=size, input_dim=input_dim, seed=seed)
    elif name.lower() == "hierarchical":
        print(f"[DATA] Loaded HierarchicalPatternDataset (size={size}, input_dim={input_dim}, level={level})")
        return HierarchicalPatternDataset(size=size, input_dim=input_dim, level=level, seed=seed)
    else:
        raise ValueError(f"Unknown dataset name: {name}. Choose 'synthetic' or 'hierarchical'")


def create_dataloaders(
    dataset_name="synthetic",
    train_size=800,
    val_size=200,
    input_dim=32,
    output_dim=1,
    batch_size=8,
    level=1,
    seed=None,
):
    """
    Create train and validation dataloaders.
    
    Args:
        dataset_name: "synthetic" or "hierarchical"
        train_size: Training set size
        val_size: Validation set size
        input_dim: Input dimension
        output_dim: Output dimension (for compatibility, actual determined by dataset)
        batch_size: Batch size
        level: Difficulty level for hierarchical dataset
        seed: Random seed
    
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = load_dataset(
        name=dataset_name,
        size=train_size,
        input_dim=input_dim,
        level=level,
        seed=seed,
    )
    
    val_dataset = load_dataset(
        name=dataset_name,
        size=val_size,
        input_dim=input_dim,
        level=level,
        seed=(seed + 1) if seed is not None else None,  # Different seed for validation
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    return train_loader, val_loader


# Backward compatibility alias
SimpleTaskDataset = SyntheticDataset
