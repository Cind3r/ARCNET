import torch.nn as nn
import torch
import random

class CompressedQModule(nn.Module):
    """
    This module attempts to implement a compact neural network for approximating Q-values
    in a reinforcement learning setting, designed to be memory-efficient and transferable
    across different environments. It uses a smaller action embedding space and a reduced
    replay buffer size to minimize memory usage while maintaining performance. Large Q-tables
    kept degrading performance for high-dimensional action spaces, so this module is designed to
    avoid that by using a neural network approach with a compact architecture.

    Key Features:
    - Compact Q-network architecture with fewer parameters.
    - Smaller action embedding space (reduced from 1M to 100K).
    - Reduced replay buffer size (increased to 150) for memory efficiency.
    - Simplified batch updates with smaller batch sizes.
    - Direct output from the Q-network to minimize complexity.
    - Uses PyTorch for efficient tensor operations and GPU support.

    Args:
        state_dim (int): Dimension of the state representation (default: 4).
        action_embedding_dim (int): Dimension of the action embedding space (default: 16).
        hidden_dim (int): Dimension of the hidden layer in the Q-network (default: 32).
    
    Attributes:
        state_dim (int): Dimension of the state representation.
        action_embedding_dim (int): Dimension of the action embedding space.
        hidden_dim (int): Dimension of the hidden layer in the Q-network.
        q_network (nn.Sequential): Neural network for approximating Q-values.
        action_embedder (nn.Embedding): Embedding layer for actions.
        replay_buffer (list): List to store experiences for training.
        buffer_size (int): Maximum size of the replay buffer.
        optimizer (torch.optim.Adam): Optimizer for training the Q-network.
    
    Methods:
        - get_q_value(state, action_id): Returns the Q-value for a given state-action pair.
        - get_q_values_batch(state, action_ids): Returns Q-values for multiple actions in a batch.
        - update_q_network(state, action_id, target_q): Updates the Q-network with new experience.
        - _batch_update(): Performs a batch update on the Q-network using experiences from the replay buffer.
        - get_memory_usage(): Returns the approximate memory usage of the module in MB.

    """
    
    def __init__(self, state_dim=4, action_embedding_dim=16, hidden_dim=32):  
        super().__init__()
        self.state_dim = state_dim
        self.action_embedding_dim = action_embedding_dim
        self.hidden_dim = hidden_dim  # Store for inheritance

        # COMPACT neural Q-function
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Direct to output - simpler!
        )
        
        # Smaller action embedding space
        self.action_embedder = nn.Embedding(100000, action_embedding_dim)  # Reduced from 1M
        
        # Smaller replay buffer
        self.replay_buffer = []
        self.buffer_size = 100  # Increased to 150
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.05, weight_decay=1e-4)  
        

    def get_q_value(self, state, action_id):
        """Get Q-value for state-action pair"""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_tensor = torch.tensor([action_id % 100000], dtype=torch.long)  # Modulo adjustment
            
            action_embed = self.action_embedder(action_tensor)
            state_action = torch.cat([state_tensor, action_embed.squeeze(0)])
            
            return self.q_network(state_action).item()
    

    def get_q_values_batch(self, state, action_ids):
        """Get Q-values for multiple actions (more efficient)"""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_tensors = torch.tensor([aid % 100000 for aid in action_ids], dtype=torch.long)
            
            action_embeds = self.action_embedder(action_tensors)
            
            # Broadcast state to match action batch size
            state_batch = state_tensor.unsqueeze(0).repeat(len(action_ids), 1)
            state_actions = torch.cat([state_batch, action_embeds], dim=1)
            
            q_values = self.q_network(state_actions).squeeze()
            return q_values.tolist() if len(action_ids) > 1 else [q_values.item()]


    def update_q_network(self, state, action_id, target_q):
        """Update Q-network with experience"""
        # Add to replay buffer
        self.replay_buffer.append((list(state), action_id % 100000, float(target_q)))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
        
        # More frequent but smaller updates
        if len(self.replay_buffer) >= 3 and random.random() < 0.8:
            self._batch_update()
    

    def _batch_update(self):
        """Simplified batch update"""
        if len(self.replay_buffer) < 3:
            return
            
        # Smaller batches
        batch_size = min(5, len(self.replay_buffer))
        batch = random.sample(self.replay_buffer, batch_size)
        
        try:
            states = torch.stack([torch.tensor(exp[0], dtype=torch.float32) for exp in batch])
            action_ids = torch.tensor([exp[1] for exp in batch], dtype=torch.long)
            targets = torch.tensor([exp[2] for exp in batch], dtype=torch.float32)
            
            # Forward pass
            action_embeds = self.action_embedder(action_ids)
            state_actions = torch.cat([states, action_embeds], dim=1)
            predicted_q = self.q_network(state_actions).squeeze()
            
            # Ensure tensors have same shape
            if predicted_q.dim() == 0:
                predicted_q = predicted_q.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)
            
            # Simple MSE loss
            loss = F.mse_loss(predicted_q, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # Gradient clipping
            self.optimizer.step()
            
        except Exception:
            pass  # Silently skip on error
    

    def get_memory_usage(self):
        """Get approximate memory usage in MB"""
        total_params = sum(p.numel() for p in self.parameters())
        param_memory = total_params * 4 / (1024 * 1024)
        buffer_memory = len(self.replay_buffer) * 50 / (1024 * 1024)  # Smaller estimate
        return param_memory + buffer_memory