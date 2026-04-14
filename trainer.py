"""
trainer.py — Training Loop for the DQN Job Scheduling Agent
=============================================================
Manages the full training pipeline: episode execution, curriculum
learning, evaluation checkpoints, early stopping, and logging.
"""

import numpy as np
import time
from typing import Optional, List, Dict
from tqdm import tqdm

from config import Config, TrainingConfig
from environment import JobSchedulingEnv
from agent import DQNAgent
from dataset_loader import generate_synthetic
from job import Job


class Trainer:
    """Orchestrates DQN training for job scheduling.
    
    Features:
        - Curriculum learning: gradually increase problem size
        - Periodic evaluation on held-out instances
        - Checkpointing and early stopping
        - Rich logging with progress bars
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the trainer.
        
        Args:
            config: Master configuration. Uses defaults if None.
        """
        self.config = config or Config()
        
        # Create environment
        self.env = JobSchedulingEnv(self.config.env)
        
        # Create agent
        self.agent = DQNAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            config=self.config.agent,
            device=self.config.device
        )
        
        # Training history
        self.episode_rewards = []
        self.episode_profits = []
        self.eval_history = []
        self.best_eval_profit = -float('inf')
        self.patience_counter = 0
        
        # Timing
        self.train_start_time = None
    
    def _get_num_jobs(self, episode: int) -> int:
        """Determine number of jobs for current episode (curriculum).
        
        Args:
            episode: Current episode number
            
        Returns:
            Number of jobs for this episode
        """
        if not self.config.training.use_curriculum:
            return self.config.job.num_jobs_range[1]
        
        stages = self.config.training.curriculum_stages
        num_jobs = stages[0][1]  # Default to first stage
        
        for start_ep, n_jobs in stages:
            if episode >= start_ep:
                num_jobs = n_jobs
        
        return num_jobs
    
    def _run_episode(self, num_jobs: int, training: bool = True) -> Dict:
        """Run a single episode of job scheduling.
        
        Args:
            num_jobs: Number of jobs in this episode
            training: Whether to train the agent (vs. evaluation)
            
        Returns:
            Episode statistics dictionary
        """
        # Generate a random job instance
        jobs = generate_synthetic(num_jobs, self.config.job)
        
        # Reset environment
        state, mask = self.env.reset(jobs)
        
        episode_reward = 0.0
        episode_loss = 0.0
        num_steps = 0
        train_steps = 0
        
        while not self.env.done:
            # Select action
            action = self.agent.select_action(state, mask, training=training)
            
            # Execute action
            (next_state, next_mask), reward, done, info = self.env.step(action)
            
            if training:
                # Store transition
                self.agent.store_transition(
                    state, mask, action, reward,
                    next_state, next_mask, done
                )
                
                # Train step
                loss = self.agent.train_step()
                if loss is not None:
                    episode_loss += loss
                    train_steps += 1
                
                # Update target network
                if self.agent.step_count % self.config.training.target_update_freq == 0:
                    self.agent.update_target_network()
            
            state, mask = next_state, next_mask
            episode_reward += reward
            num_steps += 1
        
        metrics = self.env.get_metrics()
        
        return {
            "reward": episode_reward,
            "profit": metrics.get("total_profit", 0),
            "profit_ratio": metrics.get("profit_ratio", 0),
            "completion_rate": metrics.get("completion_rate", 0),
            "utilization": metrics.get("utilization", 0),
            "makespan": metrics.get("makespan", 0),
            "num_scheduled": metrics.get("num_scheduled", 0),
            "num_jobs": num_jobs,
            "steps": num_steps,
            "avg_loss": episode_loss / max(train_steps, 1),
        }
    
    def _evaluate(self, num_episodes: int = 20) -> Dict:
        """Evaluate the agent on fresh instances without training.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Averaged evaluation metrics
        """
        eval_results = []
        for _ in range(num_episodes):
            num_jobs = self.config.job.num_jobs_range[1]
            result = self._run_episode(num_jobs, training=False)
            eval_results.append(result)
        
        # Average metrics
        avg_metrics = {}
        for key in eval_results[0]:
            if isinstance(eval_results[0][key], (int, float)):
                avg_metrics[key] = np.mean([r[key] for r in eval_results])
        
        return avg_metrics
    
    def train(self) -> Dict:
        """Execute the full training loop.
        
        Returns:
            Training history dictionary
        """
        print("=" * 70)
        print("  DQN Training for Job Sequencing with Deadlines")
        print("=" * 70)
        print(f"  Episodes: {self.config.training.num_episodes}")
        print(f"  Agent: Dueling DQN + Prioritized Replay")
        print(f"  Curriculum: {self.config.training.use_curriculum}")
        print(f"  Device: {self.config.device}")
        print("=" * 70)
        
        self.train_start_time = time.time()
        
        pbar = tqdm(
            range(1, self.config.training.num_episodes + 1),
            desc="Training",
            unit="ep",
            ncols=100
        )
        
        for episode in pbar:
            num_jobs = self._get_num_jobs(episode)
            
            # Run training episode
            result = self._run_episode(num_jobs, training=True)
            
            self.episode_rewards.append(result["reward"])
            self.episode_profits.append(result["profit"])
            
            # Update progress bar
            agent_stats = self.agent.get_stats()
            pbar.set_postfix({
                "reward": f"{result['reward']:.1f}",
                "profit": f"{result['profit']:.1f}",
                "ε": f"{agent_stats['epsilon']:.3f}",
                "loss": f"{result['avg_loss']:.4f}",
                "jobs": num_jobs,
            })
            
            # Periodic evaluation
            if episode % self.config.training.eval_freq == 0:
                eval_metrics = self._evaluate(self.config.training.num_eval_episodes)
                self.eval_history.append({
                    "episode": episode,
                    **eval_metrics
                })
                
                # Check for improvement
                if eval_metrics["profit"] > self.best_eval_profit:
                    self.best_eval_profit = eval_metrics["profit"]
                    self.patience_counter = 0
                    # Save best model
                    self.agent.save(
                        f"{self.config.paths.checkpoint_dir}/best_model.pt"
                    )
                else:
                    self.patience_counter += self.config.training.eval_freq
                
                # Log evaluation results
                if episode % (self.config.training.eval_freq * 2) == 0:
                    elapsed = time.time() - self.train_start_time
                    tqdm.write(
                        f"\n  [Eval @ Ep {episode}] "
                        f"Avg Profit: {eval_metrics['profit']:.1f} | "
                        f"Profit Ratio: {eval_metrics['profit_ratio']:.3f} | "
                        f"Completion: {eval_metrics['completion_rate']:.3f} | "
                        f"Best: {self.best_eval_profit:.1f} | "
                        f"Time: {elapsed:.0f}s"
                    )
            
            # Checkpointing
            if episode % self.config.training.checkpoint_freq == 0:
                self.agent.save(
                    f"{self.config.paths.checkpoint_dir}/checkpoint_ep{episode}.pt"
                )
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stop_patience:
                print(f"\nEarly stopping at episode {episode} "
                      f"(no improvement for {self.patience_counter} episodes)")
                break
        
        # Save final model
        self.agent.save(f"{self.config.paths.checkpoint_dir}/final_model.pt")
        
        total_time = time.time() - self.train_start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Best evaluation profit: {self.best_eval_profit:.1f}")
        
        return {
            "episode_rewards": self.episode_rewards,
            "episode_profits": self.episode_profits,
            "eval_history": self.eval_history,
            "total_time": total_time,
            "best_eval_profit": self.best_eval_profit,
        }
    
    def get_training_history(self) -> Dict:
        """Return the complete training history."""
        return {
            "episode_rewards": self.episode_rewards,
            "episode_profits": self.episode_profits,
            "eval_history": self.eval_history,
            "agent_stats": self.agent.get_stats(),
            "losses": self.agent.losses,
            "q_values": self.agent.q_values_history,
        }
