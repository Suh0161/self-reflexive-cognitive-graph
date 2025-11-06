#!/usr/bin/env python3
"""
Quick test script - runs some test cases automatically without interaction.
"""

import torch
from pathlib import Path
from test_model_interactive import load_trained_model, create_test_input, predict_with_model, compute_expected_output

def main():
    checkpoint = "outputs/model_final.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not Path(checkpoint).exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint}")
        return
    
    print(f"[INFO] Loading model from {checkpoint}...")
    model, input_dim = load_trained_model(checkpoint, device)
    
    print("\n" + "="*60)
    print("QUICK TEST - Running test cases...")
    print("="*60)
    
    # Test cases
    test_cases = [
        ("ones", "hierarchical", 1, "All ones (should sum to ~32)"),
        ("zeros", "hierarchical", 1, "All zeros (should be 0)"),
        ("sum_test", "hierarchical", 1, "Simple sum pattern"),
        ("random", "hierarchical", 1, "Random input"),
    ]
    
    for test_type, dataset_type, level, description in test_cases:
        print(f"\n[TEST] {description}")
        print("-" * 60)
        
        # Create input
        x = create_test_input(test_type, input_dim)
        
        # Get prediction
        y_pred, info = predict_with_model(model, x, device)
        
        # Compute expected
        try:
            expected = compute_expected_output(x, dataset_type, level)
            error = abs(y_pred - expected)
            
            print(f"Input type: {test_type}")
            print(f"Input sum: {x.squeeze(0).sum().item():.4f}")
            print(f"Expected: {expected:.6f}")
            print(f"Predicted: {y_pred:.6f}")
            print(f"Error: {error:.6f}")
            
            if error < 0.1:
                print("[RESULT] Good prediction!")
            elif error < 0.5:
                print("[RESULT] Moderate error")
            else:
                print("[RESULT] Large error")
        except:
            print(f"Predicted: {y_pred:.6f}")
        
        print(f"Active edges: {model.get_num_edges()}")
    
    print("\n" + "="*60)
    print("For interactive testing, run: python test_model_interactive.py")
    print("="*60)

if __name__ == "__main__":
    main()

