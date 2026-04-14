"""
environment.py -- RL Environment for Job Sequencing with Deadlines
==================================================================
Gym-compatible environment that formulates job scheduling as an MDP.
The agent selects which job to schedule next, receives profit-based
rewards, and the episode ends when no feasible jobs remain.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from job import Job, JobSet
from config import EnvironmentConfig


class JobSchedulingEnv:
    """Reinforcement Learning Environment for Job Sequencing.
    
    MDP Formulation:
        State:  (sorted_job_features_matrix, current_time, num_remaining)
        Action: Index of the job to schedule next (0 to max_jobs-1)
        Reward: +profit if completed on time, penalty for bad choices
        Done:   True when no feasible jobs remain
    
    Jobs in the state are always sorted by profit-density (descending)
    to give the agent a consistent, canonical view of the problem.
    """
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """Initialize the environment.
        
        Args:
            config: Environment configuration. Uses defaults if None.
        """
        self.config = config or EnvironmentConfig()
        self.max_jobs = self.config.max_jobs
        self.num_features = self.config.num_features
        
        # Global normalization constants (fixed, not per-episode)
        self._max_profit = 100.0
        self._max_deadline = 100
        self._max_proc = 15
        
        # State variables (initialized in reset)
        self.job_set: Optional[JobSet] = None
        self.current_time: int = 0
        self.scheduled: List[Tuple[Job, int, int]] = []  # (job, start_time, end_time)
        self.remaining_jobs: List[Job] = []
        self.total_reward: float = 0.0
        self.step_count: int = 0
        self.done: bool = False
    
    @property
    def state_dim(self) -> int:
        """Dimension of the flattened state vector."""
        # job features + global features (time, num_remaining)
        # mask is returned separately, not included in the state vector
        return self.max_jobs * self.num_features + 2
    
    @property
    def action_dim(self) -> int:
        """Number of possible actions (one per job slot)."""
        return self.max_jobs
    
    def reset(self, jobs: List[Job]) -> Tuple[np.ndarray, np.ndarray]:
        """Reset the environment with a new set of jobs.
        
        Args:
            jobs: List of Job objects for this episode
            
        Returns:
            Tuple of (state_vector, action_mask)
        """
        self.job_set = JobSet(jobs)
        self.current_time = 0
        self.scheduled = []
        self.remaining_jobs = list(jobs)
        self.total_reward = 0.0
        self.step_count = 0
        self.done = False
        
        # Update normalization constants from this instance
        if self.job_set.max_profit > 0:
            self._max_profit = max(self._max_profit, self.job_set.max_profit)
        if self.job_set.max_deadline > 0:
            self._max_deadline = max(self._max_deadline, self.job_set.max_deadline)
        if self.job_set.max_processing_time > 0:
            self._max_proc = max(self._max_proc, self.job_set.max_processing_time)
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict]:
        """Execute one scheduling action.
        
        Args:
            action: Index of job to schedule from the sorted remaining jobs list
            
        Returns:
            Tuple of ((next_state, next_mask), reward, done, info_dict)
        """
        assert not self.done, "Episode is finished. Call reset()."
        
        reward = 0.0
        info = {"action": action, "action_type": "invalid"}
        
        # Sort remaining jobs by density (same order as state representation)
        sorted_jobs = sorted(self.remaining_jobs, key=lambda j: j.density, reverse=True)
        
        if 0 <= action < len(sorted_jobs):
            job = sorted_jobs[action]
            
            if job.is_feasible(self.current_time):
                # Schedule the job
                start_time = self.current_time
                end_time = start_time + job.processing_time
                self.scheduled.append((job, start_time, end_time))
                self.current_time = end_time
                
                # Reward = normalized profit (encourages picking high-profit jobs)
                reward = job.profit * self.config.reward_completion
                info["action_type"] = "schedule"
                info["job_id"] = job.job_id
                info["profit"] = job.profit
                
                # Remove job from remaining
                self.remaining_jobs = [j for j in self.remaining_jobs if j.job_id != job.job_id]
            else:
                # Selected an infeasible job — penalty and remove it
                reward = self.config.penalty_miss
                info["action_type"] = "infeasible"
                self.remaining_jobs = [j for j in self.remaining_jobs if j.job_id != job.job_id]
        else:
            # Invalid action index (out of range) — small penalty
            reward = self.config.penalty_miss * 0.5
            info["action_type"] = "invalid"
        
        # Remove jobs that can no longer ever be completed
        self.remaining_jobs = [
            j for j in self.remaining_jobs if j.is_feasible(self.current_time)
        ]
        
        # Check termination
        self.step_count += 1
        if len(self.remaining_jobs) == 0 or self.step_count >= 500:
            self.done = True
        
        self.total_reward += reward
        info["total_reward"] = self.total_reward
        info["current_time"] = self.current_time
        info["remaining_jobs"] = len(self.remaining_jobs)
        info["scheduled_count"] = len(self.scheduled)
        
        next_state = self._get_state()
        return next_state, reward, self.done, info
    
    def _get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Construct the current state representation.
        
        Jobs are sorted by profit-density (descending) so the agent
        always sees a canonical ordering. Features are normalized
        using global constants for stability.
        
        Returns:
            Tuple of (state_vector, action_mask)
            state_vector: shape (max_jobs * num_features + 2,)
            action_mask: shape (max_jobs,) -- 1 for valid actions, 0 for invalid
        """
        # Sort by density for canonical ordering
        sorted_jobs = sorted(self.remaining_jobs, key=lambda j: j.density, reverse=True)
        
        features = np.zeros((self.max_jobs, self.num_features), dtype=np.float32)
        for i, job in enumerate(sorted_jobs[:self.max_jobs]):
            features[i] = job.to_feature_vector(
                self.current_time, self._max_profit, self._max_deadline, self._max_proc
            )
        
        # Global features
        global_feats = np.array([
            self.current_time / max(self._max_deadline, 1),
            len(self.remaining_jobs) / self.max_jobs
        ], dtype=np.float32)
        
        # Flatten and concatenate
        state = np.concatenate([features.flatten(), global_feats])
        
        # Action mask: 1 for feasible jobs, 0 for padding/infeasible
        mask = np.zeros(self.max_jobs, dtype=np.float32)
        for i, job in enumerate(sorted_jobs[:self.max_jobs]):
            if job.is_feasible(self.current_time):
                mask[i] = 1.0
        
        return state, mask
    
    def get_schedule(self) -> List[Tuple[Job, int, int]]:
        """Return the current schedule: list of (job, start_time, end_time)."""
        return list(self.scheduled)
    
    def get_total_profit(self) -> float:
        """Return total profit earned from scheduled jobs."""
        return sum(job.profit for job, _, _ in self.scheduled)
    
    def get_metrics(self) -> Dict:
        """Compute detailed metrics for the current schedule.
        
        Returns:
            Dictionary with profit, utilization, completion rate, makespan
        """
        if not self.job_set:
            return {}
        
        total_profit = self.get_total_profit()
        max_possible = self.job_set.total_profit
        num_scheduled = len(self.scheduled)
        num_total = len(self.job_set)
        
        # Compute machine utilization
        if self.scheduled:
            makespan = max(end for _, _, end in self.scheduled)
            busy_time = sum(end - start for _, start, end in self.scheduled)
            utilization = busy_time / max(makespan, 1)
        else:
            makespan = 0
            utilization = 0.0
        
        return {
            "total_profit": total_profit,
            "max_possible_profit": max_possible,
            "profit_ratio": total_profit / max(max_possible, 1),
            "completion_rate": num_scheduled / max(num_total, 1),
            "num_scheduled": num_scheduled,
            "num_total": num_total,
            "makespan": makespan,
            "utilization": utilization,
        }
    
    def render(self) -> str:
        """Return a text representation of the current schedule."""
        lines = [
            f"=== Job Scheduling Environment ===",
            f"Time: {self.current_time} | Remaining: {len(self.remaining_jobs)} | "
            f"Scheduled: {len(self.scheduled)}",
            f"Total Profit: {self.get_total_profit():.1f}",
            "",
            "Schedule:"
        ]
        for job, start, end in self.scheduled:
            lines.append(
                f"  [{start:3d} - {end:3d}] Job {job.job_id:3d} "
                f"(profit={job.profit:.1f}, deadline={job.deadline})"
            )
        return "\n".join(lines)
