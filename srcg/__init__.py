"""SRCG: Self-Reflexive Cognitive Graph"""

__version__ = "1.0.0"

from .model import SRCG, NodeMLP
from .data import (
    SimpleTaskDataset,
    SyntheticDataset,
    HierarchicalPatternDataset,
    load_dataset,
    create_dataloaders,
)

__all__ = [
    "SRCG",
    "NodeMLP",
    "SimpleTaskDataset",
    "SyntheticDataset",
    "HierarchicalPatternDataset",
    "load_dataset",
    "create_dataloaders",
]

