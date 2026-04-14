"""
replay_buffer.py — Prioritized Experience Replay Buffer
========================================================
Implements a sum-tree-based prioritized replay buffer for
efficient O(log n) sampling of transitions weighted by
TD-error priority. Based on Schaul et al. (2016).
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class Transition:
    """A single experience transition."""
    state: np.ndarray
    mask: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool


class SumTree:
    """Binary sum tree for efficient prioritized sampling.
    
    Each leaf stores the priority of a transition. Internal nodes
    store the sum of their children, enabling O(log n) proportional
    sampling and O(log n) priority updates.
    """
    
    def __init__(self, capacity: int):
        """Initialize sum tree with given leaf capacity.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0
    
    @property
    def total(self) -> float:
        """Total sum of all priorities."""
        return self.tree[0]
    
    def add(self, priority: float, data):
        """Add a new transition with given priority.
        
        Args:
            priority: Priority value (typically |TD-error| + epsilon)
            data: Transition data to store
        """
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(tree_idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update(self, tree_idx: int, priority: float):
        """Update priority at a specific tree index.
        
        Args:
            tree_idx: Index in the tree array
            priority: New priority value
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate change up to root
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get(self, value: float) -> Tuple[int, float, any]:
        """Sample a transition by cumulative priority value.
        
        Args:
            value: Random value in [0, total_priority)
            
        Returns:
            Tuple of (tree_index, priority, data)
        """
        parent = 0
        
        while True:
            left = 2 * parent + 1
            right = left + 1
            
            if left >= len(self.tree):
                break
            
            if value <= self.tree[left] or right >= len(self.tree):
                parent = left
            else:
                value -= self.tree[left]
                parent = right
        
        data_idx = parent - self.capacity + 1
        return parent, self.tree[parent], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer.
    
    Uses a sum tree for efficient proportional sampling.
    Priorities are based on TD-error magnitude, ensuring
    transitions with higher learning potential are sampled
    more frequently.
    
    Reference: Schaul et al., "Prioritized Experience Replay", ICLR 2016
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_steps: int = 100000):
        """Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_end: Final importance sampling weight (annealed)
            beta_steps: Steps over which beta anneals from start to end
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.step_count = 0
        self.max_priority = 1.0
        self.epsilon = 1e-6  # Small constant to ensure non-zero priorities
    
    @property
    def beta(self) -> float:
        """Current importance sampling exponent (annealed)."""
        fraction = min(self.step_count / max(self.beta_steps, 1), 1.0)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)
    
    def __len__(self) -> int:
        return self.tree.size
    
    def add(self, transition: Transition):
        """Add a transition with maximum priority (will be corrected on first sample).
        
        Args:
            transition: Experience transition to store
        """
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)
    
    def sample(self, batch_size: int) -> Tuple[list, np.ndarray, np.ndarray]:
        """Sample a batch of transitions proportional to their priorities.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (transitions, importance_weights, tree_indices)
        """
        self.step_count += 1
        
        transitions = []
        indices = np.zeros(batch_size, dtype=np.int64)
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # Divide priority range into equal segments
        segment = self.tree.total / batch_size
        
        # Compute min probability for normalization
        min_prob = np.inf
        
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            
            idx, priority, data = self.tree.get(value)
            
            prob = priority / max(self.tree.total, self.epsilon)
            min_prob = min(min_prob, prob)
            
            indices[i] = idx
            transitions.append(data)
            weights[i] = prob
        
        # Importance sampling weights
        beta = self.beta
        weights = (self.tree.size * weights) ** (-beta)
        max_weight = (self.tree.size * min_prob) ** (-beta)
        weights /= max(max_weight, self.epsilon)  # Normalize
        
        return transitions, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on new TD-errors.
        
        Args:
            indices: Tree indices of sampled transitions
            td_errors: Absolute TD-errors for each transition
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(idx), priority)
