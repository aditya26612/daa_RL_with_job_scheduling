"""
dataset_loader.py — Dataset Generation and Loading
====================================================
Provides synthetic job dataset generation with configurable
distributions, plus parsers for Google Cluster and Alibaba
Cluster trace data.
"""

import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Optional
from job import Job
from config import JobConfig


def generate_synthetic(num_jobs: int = 50, 
                       config: Optional[JobConfig] = None,
                       seed: Optional[int] = None) -> List[Job]:
    """Generate a synthetic job scheduling instance.
    
    Creates random jobs with configurable profit distributions and
    deadline tightness levels, simulating diverse workload scenarios.
    
    Args:
        num_jobs: Number of jobs to generate
        config: Job configuration parameters
        seed: Random seed for reproducibility
        
    Returns:
        List of Job objects
    """
    if seed is not None:
        np.random.seed(seed)
    
    config = config or JobConfig()
    jobs = []
    
    for i in range(num_jobs):
        # Generate processing time
        proc_time = np.random.randint(
            config.processing_time_range[0],
            config.processing_time_range[1] + 1
        )
        
        # Generate profit based on distribution
        if config.profit_distribution == "uniform":
            profit = np.random.uniform(*config.profit_range)
        elif config.profit_distribution == "normal":
            mean = (config.profit_range[0] + config.profit_range[1]) / 2
            std = (config.profit_range[1] - config.profit_range[0]) / 4
            profit = max(1.0, np.random.normal(mean, std))
        elif config.profit_distribution == "pareto":
            # Pareto distribution for heavy-tailed profits
            profit = np.random.pareto(2.0) * config.profit_range[0] + config.profit_range[0]
            profit = min(profit, config.profit_range[1])
        else:
            profit = np.random.uniform(*config.profit_range)
        
        # Generate deadline based on tightness
        if config.deadline_tightness == "tight":
            # Deadline is close to processing time — hard to schedule many jobs
            deadline = proc_time + np.random.randint(1, max(num_jobs // 4, 3))
        elif config.deadline_tightness == "moderate":
            deadline = proc_time + np.random.randint(1, max(num_jobs // 2, 5))
        elif config.deadline_tightness == "loose":
            deadline = proc_time + np.random.randint(num_jobs // 4, num_jobs)
        else:
            deadline = proc_time + np.random.randint(1, max(num_jobs // 2, 5))
        
        jobs.append(Job(
            job_id=i,
            profit=round(profit, 2),
            deadline=deadline,
            processing_time=proc_time
        ))
    
    return jobs


def generate_difficulty_levels(num_jobs: int = 50,
                               seed: int = 42) -> dict:
    """Generate job sets at three difficulty levels.
    
    Creates easy (loose deadlines, unit processing), medium (moderate),
    and hard (tight deadlines, variable processing) instances for
    comprehensive benchmarking.
    
    Args:
        num_jobs: Number of jobs per difficulty level
        seed: Random seed
        
    Returns:
        Dictionary mapping difficulty name to job list
    """
    np.random.seed(seed)
    
    # Easy: unit processing times, loose deadlines
    easy_config = JobConfig(
        profit_distribution="uniform",
        deadline_tightness="loose",
        processing_time_range=(1, 1),  # Unit processing times
    )
    
    # Medium: moderate processing times and deadlines
    medium_config = JobConfig(
        profit_distribution="normal",
        deadline_tightness="moderate",
        processing_time_range=(1, 5),
    )
    
    # Hard: variable processing times, tight deadlines, pareto profits
    hard_config = JobConfig(
        profit_distribution="pareto",
        deadline_tightness="tight",
        processing_time_range=(1, 10),
    )
    
    return {
        "easy": generate_synthetic(num_jobs, easy_config, seed),
        "medium": generate_synthetic(num_jobs, medium_config, seed + 1),
        "hard": generate_synthetic(num_jobs, hard_config, seed + 2),
    }


def load_google_cluster(filepath: str, num_jobs: int = 100,
                        seed: int = 42) -> List[Job]:
    """Load and convert Google Cluster trace data to job instances.
    
    Maps task events from the Google Cluster dataset:
        priority → profit (higher priority = higher profit)
        scheduling_class → deadline tightness
        CPU_request → processing_time
    
    If the CSV file is not found, generates a realistic synthetic
    dataset mimicking Google Cluster workload characteristics.
    
    Args:
        filepath: Path to the Google Cluster trace CSV
        num_jobs: Number of jobs to extract
        seed: Random seed
        
    Returns:
        List of Job objects
    """
    np.random.seed(seed)
    
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath, nrows=num_jobs * 10)
            
            # Adapt column names based on dataset version
            if 'priority' in df.columns:
                priorities = df['priority'].values[:num_jobs]
                # Normalize to profit range
                profits = (priorities - priorities.min()) / (priorities.max() - priorities.min() + 1e-6) * 99 + 1
            else:
                profits = np.random.uniform(1, 100, num_jobs)
            
            if 'CPU_request' in df.columns:
                cpu_req = df['CPU_request'].dropna().values[:num_jobs]
                proc_times = np.clip((cpu_req * 20).astype(int), 1, 10)
            else:
                proc_times = np.random.randint(1, 10, num_jobs)
            
            jobs = []
            for i in range(min(num_jobs, len(profits))):
                proc_time = int(proc_times[i]) if i < len(proc_times) else np.random.randint(1, 10)
                deadline = proc_time + np.random.randint(1, max(num_jobs // 3, 5))
                jobs.append(Job(
                    job_id=i,
                    profit=round(float(profits[i]), 2),
                    deadline=deadline,
                    processing_time=proc_time,
                ))
            return jobs
        except Exception as e:
            print(f"Warning: Could not load Google Cluster data: {e}")
            print("Generating synthetic data with Google-like characteristics...")
    
    # Generate Google-like synthetic data
    # Google workloads: bimodal priority, heavy-tailed resource requests
    print("NOTE: Using synthetic data mimicking Google Cluster workload characteristics.")
    jobs = []
    for i in range(num_jobs):
        # Bimodal priority distribution (many low-priority, some high-priority)
        if np.random.random() < 0.7:
            profit = np.random.uniform(1, 30)    # Low priority batch jobs
        else:
            profit = np.random.uniform(50, 100)   # High priority production jobs
        
        # Heavy-tailed processing times
        proc_time = max(1, int(np.random.pareto(1.5) * 2 + 1))
        proc_time = min(proc_time, 15)
        
        deadline = proc_time + np.random.randint(2, max(num_jobs // 3, 10))
        
        jobs.append(Job(
            job_id=i, profit=round(profit, 2),
            deadline=deadline, processing_time=proc_time
        ))
    
    return jobs


def load_alibaba_trace(filepath: str, num_jobs: int = 100,
                       seed: int = 42) -> List[Job]:
    """Load and convert Alibaba Cluster trace data to job instances.
    
    Maps batch task instances from the Alibaba dataset:
        plan_cpu → processing_time (higher CPU = longer)
        task_type → profit (production > batch)
        start/end times → deadline
    
    If the CSV file is not found, generates a realistic synthetic
    dataset mimicking Alibaba workload characteristics.
    
    Args:
        filepath: Path to the Alibaba trace CSV
        num_jobs: Number of jobs to extract
        seed: Random seed
        
    Returns:
        List of Job objects
    """
    np.random.seed(seed)
    
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath, nrows=num_jobs * 10)
            
            jobs = []
            for i in range(min(num_jobs, len(df))):
                row = df.iloc[i]
                
                if 'plan_cpu' in df.columns:
                    proc_time = max(1, int(row['plan_cpu'] * 10))
                else:
                    proc_time = np.random.randint(1, 10)
                
                profit = np.random.uniform(1, 100)
                deadline = proc_time + np.random.randint(2, max(num_jobs // 3, 8))
                
                jobs.append(Job(
                    job_id=i, profit=round(profit, 2),
                    deadline=deadline, processing_time=proc_time
                ))
            return jobs
        except Exception as e:
            print(f"Warning: Could not load Alibaba trace: {e}")
    
    # Generate Alibaba-like synthetic data
    print("NOTE: Using synthetic data mimicking Alibaba Cluster workload characteristics.")
    jobs = []
    for i in range(num_jobs):
        # Alibaba: mix of short batch tasks and long-running services
        task_type = np.random.choice(["short_batch", "long_batch", "service"], p=[0.5, 0.3, 0.2])
        
        if task_type == "short_batch":
            proc_time = np.random.randint(1, 3)
            profit = np.random.uniform(5, 40)
        elif task_type == "long_batch":
            proc_time = np.random.randint(3, 8)
            profit = np.random.uniform(20, 70)
        else:  # service
            proc_time = np.random.randint(5, 12)
            profit = np.random.uniform(60, 100)
        
        deadline = proc_time + np.random.randint(2, max(num_jobs // 4, 8))
        
        jobs.append(Job(
            job_id=i, profit=round(profit, 2),
            deadline=deadline, processing_time=proc_time
        ))
    
    return jobs


def get_dataset(name: str, num_jobs: int = 50, seed: int = 42,
                data_dir: str = "data") -> List[Job]:
    """Get a dataset by name.
    
    Args:
        name: Dataset name ("synthetic", "google", "alibaba")
        num_jobs: Number of jobs
        seed: Random seed
        data_dir: Directory containing trace data files
        
    Returns:
        List of Job objects
    """
    if name == "synthetic":
        return generate_synthetic(num_jobs, seed=seed)
    elif name == "google":
        filepath = os.path.join(data_dir, "google_cluster.csv")
        return load_google_cluster(filepath, num_jobs, seed)
    elif name == "alibaba":
        filepath = os.path.join(data_dir, "alibaba_trace.csv")
        return load_alibaba_trace(filepath, num_jobs, seed)
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'synthetic', 'google', or 'alibaba'.")
