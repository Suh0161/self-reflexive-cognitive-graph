#!/usr/bin/env python3
"""
Interactive testing script for SRCG model.

Test your trained model with custom inputs and see predictions.
"""

import torch
import numpy as np
from pathlib import Path
import argparse

from srcg.model import SRCG


def load_trained_model(checkpoint_path, device='cpu'):
    """Load a trained SRCG model from checkpoint."""
    print(f"\n[LOADING] Loading model from {checkpoint_path}...")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
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
    
    # Load weights
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    epoch = ckpt.get('epoch', 'unknown')
    input_dim = config.get('input_dim', 32)
    print(f"[OK] Model loaded (from epoch {epoch})")
    print(f"  - Nodes: {model.num_nodes}")
    print(f"  - Edges: {model.get_num_edges()}")
    print(f"  - Input dim: {input_dim}")
    
    return model, input_dim


def create_test_input(input_type, input_dim=32):
    """
    Create test input based on type.
    
    Args:
        input_type: "random", "ones", "zeros", "sum_test", "pattern"
        input_dim: Input dimension
    """
    if input_type == "random":
        x = torch.randn(input_dim)
    elif input_type == "ones":
        x = torch.ones(input_dim)
    elif input_type == "zeros":
        x = torch.zeros(input_dim)
    elif input_type == "sum_test":
        # Create a simple pattern: first half sums, second half sums
        x = torch.ones(input_dim) * 0.5
        x[:input_dim//4] = 1.0  # First quarter
        x[input_dim//4:input_dim//2] = 1.0  # Second quarter
    elif input_type == "pattern":
        # Alternating pattern for level 2 task
        x = torch.zeros(input_dim)
        x[::2] = 1.0  # Even indices
        x[1::2] = 0.5  # Odd indices
    else:
        raise ValueError(f"Unknown input type: {input_type}")
    
    return x.unsqueeze(0)  # Add batch dimension


def predict_with_model(model, x, device='cpu'):
    """Get prediction from model."""
    x = x.to(device)
    
    with torch.no_grad():
        y_pred, info = model(x)
        y_pred = y_pred.cpu().item()
    
    return y_pred, info


def compute_expected_output(x, dataset_type='hierarchical', level=1):
    """Compute expected output for comparison."""
    x_flat = x.squeeze(0)
    
    if dataset_type == 'hierarchical':
        if level == 1:
            # Simple sum
            expected = x_flat.sum().item()
        elif level == 2:
            # Alternating sum
            expected = (x_flat[::2] - x_flat[1::2]).sum().item()
        elif level == 3:
            # Parity
            bits = (x_flat > 0.5).float()
            expected = (bits.sum() % 2).item()
        else:
            # Threshold logic
            expected = 1.0 if (x_flat > 0.5).sum() > len(x_flat) / 2 else 0.0
    else:
        # Synthetic dataset
        import math
        expected = (
            torch.sin((x_flat[:16] * math.pi).sum()) +
            torch.cos((x_flat[16:] * math.pi / 2).sum())
        ).item() / 2.0
    
    return expected


def interactive_test(model, device='cpu', input_dim=32, dataset_type='hierarchical', level=1):
    """Interactive testing loop."""
    print("\n" + "="*60)
    print("INTERACTIVE MODEL TESTING")
    print("="*60)
    print("\nAvailable test types:")
    print("  1. random  - Random input")
    print("  2. ones    - All ones (should sum to input_dim for level 1)")
    print("  3. zeros   - All zeros (should be 0)")
    print("  4. sum_test - Simple sum pattern")
    print("  5. pattern - Alternating pattern (for level 2)")
    print("  6. custom  - Enter your own values")
    print("  7. quit    - Exit")
    print("\n" + "-"*60)
    
    # Map numbers to test types
    test_type_map = {
        '1': 'random',
        '2': 'ones',
        '3': 'zeros',
        '4': 'sum_test',
        '5': 'pattern',
        '6': 'custom',
        '7': 'quit',
    }
    
    while True:
        try:
            choice = input("\nEnter test type (1-7 or name, 'quit' to exit): ").strip().lower()
            
            if choice == 'quit' or choice == 'q' or choice == '7':
                break
            
            # Map number to test type
            if choice in test_type_map:
                choice = test_type_map[choice]
            
            if choice == 'custom':
                print(f"\nEnter {input_dim} comma-separated values (or press Enter for random):")
                values_str = input().strip()
                if values_str:
                    values = [float(v.strip()) for v in values_str.split(',')]
                    if len(values) != input_dim:
                        print(f"[ERROR] Need {input_dim} values, got {len(values)}")
                        continue
                    x = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
                else:
                    x = create_test_input("random", input_dim)
            else:
                x = create_test_input(choice, input_dim)
            
            # Get prediction
            y_pred, info = predict_with_model(model, x, device)
            
            # Compute expected (for comparison)
            try:
                expected = compute_expected_output(x, dataset_type, level)
            except:
                expected = None
            
            # Display results
            print("\n" + "-"*60)
            print(f"Input: {x.squeeze(0).tolist()[:10]}... (showing first 10 of {input_dim})")
            print(f"Input sum: {x.squeeze(0).sum().item():.4f}")
            
            if expected is not None:
                print(f"\nExpected output: {expected:.6f}")
                print(f"Model prediction: {y_pred:.6f}")
                error = abs(y_pred - expected)
                print(f"Error: {error:.6f}")
                
                if error < 0.1:
                    print("[OK] Good prediction!")
                elif error < 0.5:
                    print("[WARN] Moderate error")
                else:
                    print("[ERROR] Large error")
            else:
                print(f"\nModel prediction: {y_pred:.6f}")
            
            print(f"\nModel info:")
            print(f"  - Active edges: {model.get_num_edges()}")
            print(f"  - Converged at step: {info.get('converged_step', 'N/A')}")
            print("-"*60)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Interactive SRCG model testing')
    parser.add_argument('--checkpoint', type=str, default='outputs/model_final.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu), auto-detect if not specified')
    parser.add_argument('--dataset', type=str, default='hierarchical',
                       help='Dataset type: hierarchical or synthetic')
    parser.add_argument('--level', type=int, default=1,
                       help='Difficulty level (1-4 for hierarchical)')
    
    args = parser.parse_args()
    
    # Device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"[INFO] Using device: {device}")
    
    # Load model
    if not Path(args.checkpoint).exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        for ckpt in Path("outputs").glob("checkpoint_epoch_*.pt"):
            print(f"  - {ckpt}")
        print(f"  - outputs/model_final.pt")
        return
    
    model, input_dim = load_trained_model(args.checkpoint, device)
    
    # Interactive testing
    interactive_test(model, device, input_dim, args.dataset, args.level)


if __name__ == "__main__":
    main()

