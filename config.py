"""
config.py — Central Configuration for RL Job Sequencing
========================================================
All hyperparameters, dataset settings, and training configurations
are defined here using dataclasses for clean, type-safe access.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import os


@dataclass
class JobConfig:
    """Configuration for job generation and problem instances."""
    num_jobs_range: Tuple[int, int] = (10, 100)
    profit_range: Tuple[float, float] = (1.0, 100.0)
    deadline_range: Tuple[int, int] = (1, 50)
    processing_time_range: Tuple[int, int] = (1, 10)
    profit_distribution: str = "uniform"  # "uniform", "normal", "pareto"
    deadline_tightness: str = "moderate"   # "tight", "moderate", "loose"


@dataclass
class EnvironmentConfig:
    """Configuration for the RL environment."""
    max_jobs: int = 100          # Max jobs in a single instance (for padding)
    num_features: int = 5        # Features per job: profit, deadline, proc_time, slack, feasible
    reward_completion: float = 1.0   # Multiplier for profit reward
    penalty_miss: float = -0.5       # Penalty for skipping/missing a job
    step_cost: float = -0.01         # Small cost per step to encourage efficiency
    normalize_state: bool = True     # Whether to normalize state features


@dataclass
class AgentConfig:
    """Configuration for the DQN Agent."""
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dueling: bool = True
    
    # Learning
    learning_rate: float = 1e-3
    gamma: float = 0.99              # Discount factor
    tau: float = 0.005               # Soft update parameter
    batch_size: int = 64
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 5000  # Steps over which epsilon decays
    
    # Replay buffer
    buffer_capacity: int = 50000
    min_buffer_size: int = 500       # Min experiences before training starts
    priority_alpha: float = 0.6      # Prioritization exponent
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    num_episodes: int = 3000
    max_steps_per_episode: int = 200
    target_update_freq: int = 100    # Steps between target network updates
    eval_freq: int = 50              # Episodes between evaluations
    num_eval_episodes: int = 20
    checkpoint_freq: int = 200       # Episodes between checkpoints
    early_stop_patience: int = 300   # Episodes without improvement before stopping
    log_freq: int = 10               # Episodes between console logs
    
    # Problem size curriculum
    use_curriculum: bool = True
    curriculum_stages: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 20),       # Episodes 0-500: 20 jobs
        (500, 50),     # Episodes 500-1500: 50 jobs
        (1500, 100),   # Episodes 1500+: 100 jobs
    ])


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    test_sizes: List[int] = field(default_factory=lambda: [20, 50, 100, 200])
    num_test_instances: int = 50
    solvers: List[str] = field(default_factory=lambda: [
        "rl_agent", "greedy_profit", "edf", "sjf", "profit_density"
    ])


@dataclass
class PathConfig:
    """File and directory paths."""
    output_dir: str = "output"
    checkpoint_dir: str = "output/checkpoints"
    plots_dir: str = "output/plots"
    results_dir: str = "output/results"
    data_dir: str = "data"
    
    def create_dirs(self):
        """Create all necessary directories."""
        for d in [self.output_dir, self.checkpoint_dir, 
                  self.plots_dir, self.results_dir, self.data_dir]:
            os.makedirs(d, exist_ok=True)


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    job: JobConfig = field(default_factory=JobConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    seed: int = 42
    device: str = "cpu"  # "cpu" or "cuda"
    
    def __post_init__(self):
        self.paths.create_dirs()


# Global default config instance
DEFAULT_CONFIG = Config()
