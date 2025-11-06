"""
SRCG Core Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class NodeMLP(nn.Module):
    """Shared MLP for node transformations."""
    
    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, d] -> [N, d]"""
        return F.relu(self.fc2(F.relu(self.fc1(x))))


class InputEncoder(nn.Module):
    """Encodes task inputs into initial node states."""
    
    def __init__(self, input_dim: int, num_nodes: int, node_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        # Project input to node states
        self.proj = nn.Linear(input_dim, num_nodes * node_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim] -> [B, N, d]
        """
        batch_size = x.shape[0]
        h_flat = self.proj(x)  # [B, N*d]
        h = h_flat.view(batch_size, self.num_nodes, self.node_dim)
        return h


class OutputHead(nn.Module):
    """Maps node states to task output."""
    
    def __init__(self, node_dim: int, output_dim: int, hidden: int = 64):
        super().__init__()
        self.output_dim = output_dim
        self.head = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )
    
    def forward(self, h_pooled: torch.Tensor) -> torch.Tensor:
        """
        h_pooled: [B, d] -> [B, output_dim]
        """
        return self.head(h_pooled)


class SRCG(nn.Module):
    """
    Self-Reflexive Cognitive Graph
    
    Core model with dynamic graph structure and reward-modulated plasticity.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int = 100,
        node_dim: int = 128,
        reasoning_steps: int = 20,
        alpha_damping: float = 0.5,
        w_max: float = 0.1,
        prune_threshold: float = 0.02,
        add_threshold: float = 0.8,
        max_new_edges: int = 10,
        eta_w: float = 0.01,
        lambda_energy: float = 0.1,
        lambda_inst: float = 0.05,
        convergence_eps: float = 1e-3,
        init_sparsity: float = 0.1,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.reasoning_steps = reasoning_steps
        self.alpha_damping = alpha_damping
        self.w_max = w_max
        self.prune_threshold = prune_threshold
        self.add_threshold = add_threshold
        self.max_new_edges = max_new_edges
        self.eta_w = eta_w
        self.lambda_energy = lambda_energy
        self.lambda_inst = lambda_inst
        self.convergence_eps = convergence_eps
        
        # Components
        self.encoder = InputEncoder(input_dim, num_nodes, node_dim)
        self.node_transform = nn.Linear(node_dim, node_dim)  # W_self
        self.output_head = OutputHead(node_dim, output_dim)
        
        # Initialize adjacency matrix
        # A[j, i] = weight from node j -> i
        self.register_buffer('A', self._init_adjacency(init_sparsity))
    
    def _init_adjacency(self, sparsity: float) -> torch.Tensor:
        """Initialize sparse adjacency matrix."""
        N = self.num_nodes
        A = torch.zeros(N, N)
        
        # Create sparse initial graph
        num_edges = int(N * N * sparsity)
        indices = torch.randperm(N * N)[:num_edges]
        for idx in indices:
            i, j = idx // N, idx % N
            if i != j:  # No self-loops initially
                A[j, i] = torch.rand(1).item() * 0.1 - 0.05  # [-0.05, 0.05]
        
        return A.clamp(-self.w_max, self.w_max)
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through reasoning steps.
        
        Args:
            x: Input tensor [B, input_dim]
            return_intermediates: If True, return debugging info
        
        Returns:
            y_pred: [B, output_dim]
            info: dict with intermediate states
        """
        batch_size = x.shape[0]
        
        # 1) Encode inputs -> initial node states
        H = self.encoder(x)  # [B, N, d]
        
        # Get adjacency (shared across batch)
        A = self.A  # [N, N]
        
        # 2) Reasoning / message passing
        H_prev_step = None  # Will hold H_{T_r-1} for instability cost
        converged_step = None
        
        for t in range(self.reasoning_steps):
            # Aggregate messages: M = A^T @ H
            # A[j, i] = j -> i, so A^T[i, j] = i -> j for message passing
            # We want messages TO node i FROM other nodes
            # H: [B, N, d], A: [N, N]
            # For each batch: M[i] = sum_j A[j, i] * H[j]
            M = torch.einsum('ji,bjd->bid', A, H)  # [B, N, d]
            
            # Node update proposal
            H_self = self.node_transform(H)  # [B, N, d]
            H_hat = F.relu(H_self + M)  # [B, N, d]
            
            # Damped update
            H_new = (1 - self.alpha_damping) * H + self.alpha_damping * H_hat
            
            # Convergence check
            if converged_step is None:
                diff = (H_new - H).norm(dim=-1).max()  # Max change across nodes
                if diff.item() < self.convergence_eps:
                    converged_step = t
            
            # Track second-to-last state for instability cost
            if t == self.reasoning_steps - 2:
                H_prev_step = H.clone()
            
            H = H_new
        
        # If we didn't get to step T_r-1 (e.g., T_r < 2), use initial state
        if H_prev_step is None:
            H_prev_step = self.encoder(x)
        
        # 3) Output (pool node states)
        h_pooled = H.mean(dim=1)  # [B, d]
        y_pred = self.output_head(h_pooled)  # [B, output_dim]
        
        info = {
            'H_final': H,
            'H_prev_step': H_prev_step,  # H_{T_r-1} for instability
            'A': A,
            'converged_step': converged_step,
        }
        
        return y_pred, info
    
    def compute_structure_costs(
        self,
        H_final: torch.Tensor,
        H_prev_step: torch.Tensor,
        A: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute energy and instability costs.
        
        Args:
            H_final: [B, N, d] final node states (H_{T_r})
            H_prev_step: [B, N, d] previous step states (H_{T_r-1})
            A: [N, N] adjacency matrix
        
        Returns:
            C_energy: scalar tensor
            C_inst: scalar tensor
        """
        # Energy cost: mean absolute edge weight
        C_energy = A.abs().mean()
        
        # Instability: how much nodes moved in last step
        # C_inst = (1/N) Σ_i ||H_{T_r}[i] - H_{T_r-1}[i]||_2
        # Average L2 norm of change per node, then mean across batch
        node_changes = (H_final - H_prev_step).norm(dim=-1)  # [B, N]
        C_inst = node_changes.mean()
        
        return C_energy, C_inst
    
    def compute_reward(
        self,
        task_loss: torch.Tensor,
        C_energy: torch.Tensor,
        C_inst: torch.Tensor
    ) -> float:
        """
        Compute reward from task success and structure costs.
        
        Args:
            task_loss: scalar tensor
            C_energy: scalar tensor
            C_inst: scalar tensor
        
        Returns:
            R_k: scalar reward
        """
        # Success: 1.0 - clamped task loss
        success = 1.0 - task_loss.clamp(max=1.0).item()
        
        # Reward = success - cost penalties
        R_k = success - self.lambda_energy * C_energy.item() - self.lambda_inst * C_inst.item()
        
        return R_k
    
    def update_structure(
        self,
        H_final: torch.Tensor,
        R_k: float
    ):
        """
        Update graph structure based on reward and node activations.
        
        This is a non-gradient operation (called in no_grad context).
        
        Args:
            H_final: [B, N, d] final node states
            R_k: scalar reward
        """
        # Use mean across batch for structure update
        H = H_final.mean(dim=0)  # [N, d]
        A = self.A.clone()  # [N, N]
        
        # Normalize for cosine similarity
        H_norm = H / (H.norm(dim=1, keepdim=True) + 1e-6)  # [N, d]
        
        # 1) Update existing edges (Hebbian-reward)
        edge_mask = A.abs() > 0
        i_indices, j_indices = torch.where(edge_mask)
        
        for idx in range(len(i_indices)):
            i, j = i_indices[idx].item(), j_indices[idx].item()
            
            # Node similarity (dot product)
            s_ji = (H[j] * H[i]).sum().item()
            
            # Weight update
            dA = self.eta_w * R_k * s_ji
            
            # Apply and clip
            A[j, i] = torch.clamp(A[j, i] + dA, -self.w_max, self.w_max)
            
            # Prune if below threshold
            if A[j, i].abs() < self.prune_threshold:
                A[j, i] = 0.0
        
        # 2) Add new edges (correlation-based)
        cos_sim = H_norm @ H_norm.T  # [N, N]
        
        add_count = 0
        # Sort by cosine similarity (descending)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                if A[i, j].abs() < 1e-6 and cos_sim[i, j] > self.add_threshold:
                    if add_count < self.max_new_edges:
                        A[i, j] = 0.05 * cos_sim[i, j].item()
                        add_count += 1
                    else:
                        break
        
        # Write back
        self.A.copy_(A.clamp(-self.w_max, self.w_max))
    
    def get_num_edges(self) -> int:
        """Get current number of non-zero edges."""
        return (self.A.abs() > 1e-6).sum().item()

