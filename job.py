"""
job.py — Job Data Structures and Utilities
===========================================
Defines the Job dataclass and utility functions for
sorting, filtering, and computing job-level metrics.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Job:
    """Represents a single job with scheduling attributes.
    
    Attributes:
        job_id: Unique identifier for the job
        profit: Reward earned if job completes before deadline
        deadline: Latest time by which job must finish
        processing_time: Time units required to complete the job
        arrival_time: Time at which the job becomes available (default 0)
    """
    job_id: int
    profit: float
    deadline: int
    processing_time: int
    arrival_time: int = 0
    
    @property
    def density(self) -> float:
        """Profit per unit processing time (profit density)."""
        return self.profit / max(self.processing_time, 1)
    
    def slack(self, current_time: int) -> int:
        """Compute slack: time remaining before deadline minus processing time.
        
        Args:
            current_time: Current time step
            
        Returns:
            Positive slack means job can still be completed;
            negative means it's impossible to meet the deadline.
        """
        return self.deadline - current_time - self.processing_time
    
    def is_feasible(self, current_time: int) -> bool:
        """Check if this job can be completed before its deadline.
        
        Args:
            current_time: Current time step
            
        Returns:
            True if the job can start now and finish before deadline
        """
        return (current_time >= self.arrival_time and 
                current_time + self.processing_time <= self.deadline)
    
    def to_feature_vector(self, current_time: int, max_profit: float = 100.0,
                          max_deadline: int = 50, max_proc: int = 10) -> np.ndarray:
        """Convert job to a normalized feature vector for the RL agent.
        
        Features: [normalized_profit, normalized_deadline, normalized_proc_time,
                   normalized_slack, is_feasible]
        
        Args:
            current_time: Current time step
            max_profit: Maximum profit for normalization
            max_deadline: Maximum deadline for normalization
            max_proc: Maximum processing time for normalization
            
        Returns:
            numpy array of shape (5,) with normalized features
        """
        return np.array([
            self.profit / max(max_profit, 1.0),
            self.deadline / max(max_deadline, 1),
            self.processing_time / max(max_proc, 1),
            max(self.slack(current_time), -max_deadline) / max(max_deadline, 1),
            1.0 if self.is_feasible(current_time) else 0.0
        ], dtype=np.float32)
    
    def __repr__(self) -> str:
        return (f"Job(id={self.job_id}, profit={self.profit:.1f}, "
                f"deadline={self.deadline}, proc={self.processing_time})")


class JobSet:
    """A collection of jobs with batch operations.
    
    Provides utility methods for filtering, sorting, and
    computing aggregate statistics over a set of jobs.
    """
    
    def __init__(self, jobs: List[Job]):
        """Initialize with a list of Job objects."""
        self.jobs = list(jobs)
    
    def __len__(self) -> int:
        return len(self.jobs)
    
    def __getitem__(self, idx: int) -> Job:
        return self.jobs[idx]
    
    def __iter__(self):
        return iter(self.jobs)
    
    @property
    def total_profit(self) -> float:
        """Sum of all job profits."""
        return sum(j.profit for j in self.jobs)
    
    @property
    def max_profit(self) -> float:
        """Maximum individual job profit."""
        return max((j.profit for j in self.jobs), default=0.0)
    
    @property
    def max_deadline(self) -> int:
        """Latest deadline across all jobs."""
        return max((j.deadline for j in self.jobs), default=0)
    
    @property
    def max_processing_time(self) -> int:
        """Maximum processing time across all jobs."""
        return max((j.processing_time for j in self.jobs), default=1)
    
    def feasible_jobs(self, current_time: int) -> 'JobSet':
        """Return subset of jobs that can still be completed.
        
        Args:
            current_time: Current time step
            
        Returns:
            New JobSet containing only feasible jobs
        """
        return JobSet([j for j in self.jobs if j.is_feasible(current_time)])
    
    def sort_by_profit(self, descending: bool = True) -> 'JobSet':
        """Sort jobs by profit."""
        return JobSet(sorted(self.jobs, key=lambda j: j.profit, reverse=descending))
    
    def sort_by_deadline(self) -> 'JobSet':
        """Sort jobs by deadline (earliest first)."""
        return JobSet(sorted(self.jobs, key=lambda j: j.deadline))
    
    def sort_by_processing_time(self) -> 'JobSet':
        """Sort jobs by processing time (shortest first)."""
        return JobSet(sorted(self.jobs, key=lambda j: j.processing_time))
    
    def sort_by_density(self, descending: bool = True) -> 'JobSet':
        """Sort jobs by profit density (profit / processing_time)."""
        return JobSet(sorted(self.jobs, key=lambda j: j.density, reverse=descending))
    
    def to_feature_matrix(self, current_time: int, max_size: int = 100) -> np.ndarray:
        """Convert all jobs to a padded feature matrix.
        
        Args:
            current_time: Current time step
            max_size: Maximum number of jobs (for padding)
            
        Returns:
            numpy array of shape (max_size, 5) with zero-padding
        """
        max_p = self.max_profit if self.max_profit > 0 else 100.0
        max_d = self.max_deadline if self.max_deadline > 0 else 50
        max_pt = self.max_processing_time if self.max_processing_time > 0 else 10
        
        features = np.zeros((max_size, 5), dtype=np.float32)
        for i, job in enumerate(self.jobs[:max_size]):
            features[i] = job.to_feature_vector(current_time, max_p, max_d, max_pt)
        return features
    
    def get_mask(self, current_time: int, max_size: int = 100) -> np.ndarray:
        """Create a binary mask: 1 for valid (feasible) jobs, 0 for padding/infeasible.
        
        Args:
            current_time: Current time step
            max_size: Maximum number of jobs (matching feature matrix size)
            
        Returns:
            numpy array of shape (max_size,) with binary values
        """
        mask = np.zeros(max_size, dtype=np.float32)
        for i, job in enumerate(self.jobs[:max_size]):
            if job.is_feasible(current_time):
                mask[i] = 1.0
        return mask
    
    def summary(self) -> dict:
        """Compute summary statistics for the job set."""
        if len(self.jobs) == 0:
            return {"num_jobs": 0}
        
        profits = [j.profit for j in self.jobs]
        deadlines = [j.deadline for j in self.jobs]
        proc_times = [j.processing_time for j in self.jobs]
        
        return {
            "num_jobs": len(self.jobs),
            "total_profit": sum(profits),
            "avg_profit": np.mean(profits),
            "avg_deadline": np.mean(deadlines),
            "avg_processing_time": np.mean(proc_times),
            "max_deadline": max(deadlines),
            "total_processing_time": sum(proc_times),
        }
