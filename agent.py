"""
agent.py — Deep Q-Network Agent with Dueling Architecture
==========================================================
Implements a DQN agent for job scheduling with:
- Dueling network architecture (separate value & advantage streams)
- Epsilon-greedy exploration with cosine annealing decay
- Soft target network updates
- Integration with prioritized experience replay
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Optional
import math
import os

from config import AgentConfig
from replay_buffer import PrioritizedReplayBuffer, Transition


class DuelingQNetwork(nn.Module):
    """Dueling Deep Q-Network architecture.
    
    Separates state-value estimation and action-advantage estimation,
    enabling the network to learn which states are valuable without
    needing to evaluate every action in each state.
    
    Architecture:
        Input → Shared Encoder → [Value Stream, Advantage Stream] → Q-values
    
    Reference: Wang et al., "Dueling Network Architectures for Deep RL", ICML 2016
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: list = [256, 128, 64], dueling: bool = True):
        """Initialize the Dueling Q-Network.
        
        Args:
            state_dim: Dimension of the state vector
            action_dim: Number of possible actions
            hidden_dims: List of hidden layer sizes for the shared encoder
            dueling: Whether to use dueling architecture
        """
        super().__init__()
        self.dueling = dueling
        self.action_dim = action_dim
        
        # Shared feature encoder
        encoder_layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims[:-1]:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        if dueling:
            # Value stream: estimates V(s)
            self.value_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], 1)
            )
            
            # Advantage stream: estimates A(s, a)
            self.advantage_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim)
            )
        else:
            # Standard Q-network
            self.q_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim)
            )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute Q-values.
        
        Args:
            state: Batch of state vectors, shape (batch_size, state_dim)
            
        Returns:
            Q-values for each action, shape (batch_size, action_dim)
        """
        features = self.encoder(state)
        
        if self.dueling:
            value = self.value_stream(features)          # (batch, 1)
            advantage = self.advantage_stream(features)  # (batch, action_dim)
            
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.q_head(features)
        
        return q_values


class DQNAgent:
    """Deep Q-Network Agent for Job Scheduling.
    
    Combines DuelingQNetwork with epsilon-greedy exploration,
    prioritized experience replay, and soft target updates
    to learn an optimal job scheduling policy.
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 config: Optional[AgentConfig] = None, device: str = "cpu"):
        """Initialize the DQN Agent.
        
        Args:
            state_dim: Dimension of the state vector
            action_dim: Number of possible actions
            config: Agent configuration
            device: Torch device ("cpu" or "cuda")
        """
        self.config = config or AgentConfig()
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.q_network = DuelingQNetwork(
            state_dim, action_dim,
            self.config.hidden_dims,
            self.config.dueling
        ).to(self.device)
        
        self.target_network = DuelingQNetwork(
            state_dim, action_dim,
            self.config.hidden_dims,
            self.config.dueling
        ).to(self.device)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=5000, eta_min=1e-5
        )
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.buffer_capacity,
            alpha=self.config.priority_alpha,
            beta_start=self.config.priority_beta_start,
            beta_end=self.config.priority_beta_end,
        )
        
        # Exploration
        self.epsilon = self.config.epsilon_start
        self.step_count = 0
        
        # Training stats
        self.losses = []
        self.q_values_history = []
    
    def _update_epsilon(self):
        """Update epsilon using cosine annealing schedule."""
        progress = min(self.step_count / max(self.config.epsilon_decay_steps, 1), 1.0)
        # Cosine annealing from epsilon_start to epsilon_end
        self.epsilon = self.config.epsilon_end + 0.5 * (
            self.config.epsilon_start - self.config.epsilon_end
        ) * (1 + math.cos(math.pi * progress))
    
    def select_action(self, state: np.ndarray, mask: np.ndarray,
                      training: bool = True) -> int:
        """Select an action using epsilon-greedy policy with action masking.
        
        Args:
            state: Current state vector
            mask: Action mask (1 = valid, 0 = invalid)
            training: Whether to use exploration
            
        Returns:
            Selected action index
        """
        if training:
            self._update_epsilon()
        
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            # Random action among valid actions
            valid_actions = np.where(mask > 0)[0]
            if len(valid_actions) == 0:
                return 0  # Will trigger episode end
            return np.random.choice(valid_actions)
        
        # Greedy action with masking
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
            
            # Mask invalid actions with -inf
            masked_q = q_values.copy()
            masked_q[mask == 0] = -np.inf
            
            return int(np.argmax(masked_q))
    
    def store_transition(self, state: np.ndarray, mask: np.ndarray,
                         action: int, reward: float,
                         next_state: np.ndarray, next_mask: np.ndarray,
                         done: bool):
        """Store a transition in the replay buffer.
        
        Args:
            state, mask: Current state and action mask
            action: Action taken
            reward: Reward received
            next_state, next_mask: Next state and mask
            done: Whether episode ended
        """
        transition = Transition(
            state=state, mask=mask, action=action, reward=reward,
            next_state=next_state, next_mask=next_mask, done=done
        )
        self.replay_buffer.add(transition)
    
    def train_step(self) -> Optional[float]:
        """Perform one training step using a batch from replay buffer.
        
        Returns:
            Loss value, or None if buffer doesn't have enough samples
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None
        
        self.step_count += 1
        
        # Sample batch
        transitions, weights, indices = self.replay_buffer.sample(
            self.config.batch_size
        )
        
        # Prepare batch tensors
        states = torch.FloatTensor(
            np.array([t.state for t in transitions])
        ).to(self.device)
        masks = torch.FloatTensor(
            np.array([t.mask for t in transitions])
        ).to(self.device)
        actions = torch.LongTensor(
            [t.action for t in transitions]
        ).to(self.device)
        rewards = torch.FloatTensor(
            [t.reward for t in transitions]
        ).to(self.device)
        next_states = torch.FloatTensor(
            np.array([t.next_state for t in transitions])
        ).to(self.device)
        next_masks = torch.FloatTensor(
            np.array([t.next_mask for t in transitions])
        ).to(self.device)
        dones = torch.FloatTensor(
            [float(t.done) for t in transitions]
        ).to(self.device)
        importance_weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select actions, target network to evaluate
        with torch.no_grad():
            next_q_online = self.q_network(next_states)
            # Mask invalid actions
            next_q_online[next_masks == 0] = -1e9
            next_actions = next_q_online.argmax(dim=1)
            
            next_q_target = self.target_network(next_states)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q = rewards + self.config.gamma * next_q * (1 - dones)
        
        # TD errors for priority updates
        td_errors = (current_q - target_q).detach().cpu().numpy()
        
        # Weighted Huber loss
        element_wise_loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = (element_wise_loss * importance_weights).mean()
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)
        
        loss_val = loss.item()
        self.losses.append(loss_val)
        self.q_values_history.append(current_q.mean().item())
        
        return loss_val
    
    def update_target_network(self):
        """Soft update target network: θ_target ← τ*θ_online + (1-τ)*θ_target."""
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * online_param.data + 
                (1 - self.config.tau) * target_param.data
            )
    
    def save(self, filepath: str):
        """Save agent state to file.
        
        Args:
            filepath: Path to save checkpoint
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'losses': self.losses[-1000:],  # Keep last 1000
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state from file.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon_end)
        self.step_count = checkpoint.get('step_count', 0)
        self.losses = checkpoint.get('losses', [])
    
    def get_stats(self) -> dict:
        """Return current training statistics."""
        return {
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "buffer_size": len(self.replay_buffer),
            "avg_loss": np.mean(self.losses[-100:]) if self.losses else 0,
            "avg_q": np.mean(self.q_values_history[-100:]) if self.q_values_history else 0,
            "lr": self.optimizer.param_groups[0]['lr'],
        }
