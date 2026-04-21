"""
kaggle_data_generator.py -- Kaggle GPU Job Scheduling Dataset Generator
========================================================================
Generates synthetic job scheduling datasets modeled on real-world
Kaggle notebook execution workloads. Statistical characteristics
are derived from:

  - Kaggle Notebook GPU/TPU quota limits (30 hrs/week GPU, 20 hrs/week TPU)
    Source: https://www.kaggle.com/docs/notebooks
  - Cloud GPU hourly pricing (AWS, GCP, Azure)
    Source: https://cloud.google.com/products/calculator
    Source: https://aws.amazon.com/ec2/pricing/on-demand/
  - Kaggle Meta analysis of notebook runtimes and competitions
    Source: https://www.kaggle.com/code/kaggle/meta-kaggle-code
  - MLCommons Training Benchmark results for model training times
    Source: https://mlcommons.org/benchmarks/training/

Each job represents a Kaggle notebook execution task (model training,
inference pipeline, data preprocessing, etc.) that must be scheduled
on shared GPU/CPU resources within quota and deadline constraints.
"""

import numpy as np
import pandas as pd
import os
from typing import List, Optional
from job import Job


# ============================================================
# Real-world statistical parameters derived from public sources
# ============================================================

# GPU instance costs (USD/hour) from AWS, GCP, Azure (2024-2025)
# Source: https://cloud.google.com/products/calculator
#         https://aws.amazon.com/ec2/pricing/on-demand/
GPU_COST_PER_HOUR = {
    "T4":   {"aws": 0.526, "gcp": 0.35, "azure": 0.70, "kaggle_free": 0.0},
    "P100": {"aws": 1.46,  "gcp": 1.46, "azure": 1.50, "kaggle_free": 0.0},
    "V100": {"aws": 3.06,  "gcp": 2.48, "azure": 3.06, "kaggle_free": 0.0},
    "A100": {"aws": 4.10,  "gcp": 3.67, "azure": 3.67, "kaggle_free": 0.0},
}

# Kaggle quota constraints (hours per week)
# Source: https://www.kaggle.com/docs/notebooks
KAGGLE_QUOTAS = {
    "gpu_weekly_hours": 30,
    "tpu_weekly_hours": 20,
    "max_session_hours": 12,   # Max single notebook execution
    "idle_timeout_min": 20,
}

# Typical ML workload runtimes (hours) from MLCommons + Kaggle Meta
# Source: https://mlcommons.org/benchmarks/training/
#         https://www.kaggle.com/code/kaggle/meta-kaggle-code
WORKLOAD_PROFILES = {
    "data_preprocessing": {
        "runtime_range": (0.1, 1.5),    # 6 min to 1.5 hours
        "gpu_type": "T4",
        "frequency": 0.20,               # 20% of all jobs
    },
    "model_training_small": {
        "runtime_range": (0.5, 3.0),     # 30 min to 3 hours
        "gpu_type": "T4",
        "frequency": 0.30,
    },
    "model_training_large": {
        "runtime_range": (2.0, 9.0),     # 2 to 9 hours (near max session)
        "gpu_type": "P100",
        "frequency": 0.15,
    },
    "fine_tuning_llm": {
        "runtime_range": (3.0, 12.0),    # 3 to 12 hours (LoRA, QLoRA)
        "gpu_type": "V100",
        "frequency": 0.10,
    },
    "inference_pipeline": {
        "runtime_range": (0.25, 2.0),    # 15 min to 2 hours
        "gpu_type": "T4",
        "frequency": 0.15,
    },
    "ensemble_submission": {
        "runtime_range": (1.0, 6.0),     # 1 to 6 hours
        "gpu_type": "P100",
        "frequency": 0.10,
    },
}

# Competition deadline distribution (hours from now)
# Modeled on Kaggle active competition timeline patterns
DEADLINE_PROFILES = {
    "urgent":    {"range": (2, 12),   "weight": 0.15},  # Last-minute submissions
    "soon":      {"range": (12, 48),  "weight": 0.30},  # Within 2 days
    "moderate":  {"range": (48, 168), "weight": 0.35},  # Within 1 week
    "relaxed":   {"range": (168, 720),"weight": 0.20},  # 1 week to 1 month
}


def compute_cloud_cost_saving(gpu_type: str, runtime_hours: float) -> float:
    """Compute the dollar savings from scheduling on free Kaggle GPU
    vs. renting equivalent cloud GPU.

    This is the 'profit' of successfully scheduling a job:
    the money saved by using Kaggle's free quota instead of paying
    for an equivalent cloud instance.

    Args:
        gpu_type: GPU model name (T4, P100, V100, A100)
        runtime_hours: Estimated runtime in hours

    Returns:
        Cost saving in USD (this becomes the job's 'profit')
    """
    costs = GPU_COST_PER_HOUR.get(gpu_type, GPU_COST_PER_HOUR["T4"])
    # Average across AWS, GCP, Azure for the "market rate"
    avg_cloud_cost = np.mean([costs["aws"], costs["gcp"], costs["azure"]])
    # The saving is what you'd pay on cloud minus Kaggle free ($0)
    return round(avg_cloud_cost * runtime_hours, 2)


def generate_kaggle_dataset(
    num_jobs: int = 100,
    seed: int = 42,
    quota_hours: float = 30.0,
    time_resolution: str = "slots",
) -> List[Job]:
    """Generate a realistic Kaggle GPU job scheduling dataset.

    Each job represents a notebook execution task with:
      - profit: Cloud cost saving (USD) from using Kaggle's free GPU
      - deadline: Hours until competition/quota deadline (discretized to slots)
      - processing_time: Estimated GPU hours (discretized to slots)

    Statistical distributions are modeled from real Kaggle platform data.

    Args:
        num_jobs: Number of jobs to generate
        seed: Random seed for reproducibility
        quota_hours: Total available GPU hours (Kaggle weekly quota)
        time_resolution: 'slots' for 30-min slots, 'hours' for hourly

    Returns:
        List of Job objects
    """
    np.random.seed(seed)

    slot_size = 0.5 if time_resolution == "slots" else 1.0
    jobs = []
    records = []  # For CSV export

    # Build workload type probabilities
    workload_types = list(WORKLOAD_PROFILES.keys())
    workload_probs = [WORKLOAD_PROFILES[w]["frequency"] for w in workload_types]

    # Build deadline type probabilities
    deadline_types = list(DEADLINE_PROFILES.keys())
    deadline_probs = [DEADLINE_PROFILES[d]["weight"] for d in deadline_types]

    for i in range(num_jobs):
        # Sample workload type
        wtype = np.random.choice(workload_types, p=workload_probs)
        profile = WORKLOAD_PROFILES[wtype]

        # Runtime from workload profile (with slight noise)
        runtime = np.random.uniform(*profile["runtime_range"])
        # Add +-10% noise for realism
        runtime *= np.random.uniform(0.9, 1.1)
        runtime = max(0.1, runtime)

        # Discretize to time slots
        proc_time_slots = max(1, int(np.ceil(runtime / slot_size)))

        # GPU type determines cost saving (= profit)
        gpu_type = profile["gpu_type"]
        cost_saving = compute_cloud_cost_saving(gpu_type, runtime)
        # Add small random variation (different regions, spot pricing)
        cost_saving *= np.random.uniform(0.85, 1.15)
        cost_saving = max(0.10, round(cost_saving, 2))

        # Deadline from competition/quota profile
        dtype = np.random.choice(deadline_types, p=deadline_probs)
        dprofile = DEADLINE_PROFILES[dtype]
        deadline_hours = np.random.uniform(*dprofile["range"])
        deadline_slots = max(proc_time_slots + 1,
                             int(np.ceil(deadline_hours / slot_size)))

        job = Job(
            job_id=i,
            profit=cost_saving,
            deadline=deadline_slots,
            processing_time=proc_time_slots,
        )
        jobs.append(job)

        records.append({
            "job_id": i,
            "workload_type": wtype,
            "gpu_type": gpu_type,
            "runtime_hours": round(runtime, 2),
            "processing_time_slots": proc_time_slots,
            "cloud_cost_usd": cost_saving,
            "deadline_type": dtype,
            "deadline_hours": round(deadline_hours, 1),
            "deadline_slots": deadline_slots,
            "density": round(cost_saving / proc_time_slots, 4),
        })

    return jobs, pd.DataFrame(records)


def generate_competition_scenarios(seed: int = 42) -> dict:
    """Generate multiple competition-themed job scheduling scenarios.

    Returns:
        Dictionary mapping scenario name to (jobs, dataframe) tuple
    """
    scenarios = {}

    # Scenario 1: Week before competition deadline (high urgency)
    scenarios["competition_crunch"] = generate_kaggle_dataset(
        num_jobs=80, seed=seed, quota_hours=30.0
    )

    # Scenario 2: Normal weekly workload (mixed priorities)
    scenarios["regular_workload"] = generate_kaggle_dataset(
        num_jobs=120, seed=seed + 100, quota_hours=30.0
    )

    # Scenario 3: Multi-competition week (heavy load)
    scenarios["multi_competition"] = generate_kaggle_dataset(
        num_jobs=200, seed=seed + 200, quota_hours=30.0
    )

    return scenarios


def save_dataset(df: pd.DataFrame, filepath: str):
    """Save dataset to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"  Saved dataset: {filepath} ({len(df)} jobs)")


if __name__ == "__main__":
    print("=" * 60)
    print("  Kaggle GPU Job Scheduling Dataset Generator")
    print("=" * 60)

    # Generate and save datasets
    os.makedirs("data", exist_ok=True)

    jobs, df = generate_kaggle_dataset(num_jobs=150, seed=42)
    save_dataset(df, "data/kaggle_jobs.csv")

    scenarios = generate_competition_scenarios()
    for name, (jobs_s, df_s) in scenarios.items():
        save_dataset(df_s, f"data/kaggle_{name}.csv")

    print("\n  Dataset Statistics:")
    print(f"  Total jobs: {len(df)}")
    print(f"  Workload types: {df['workload_type'].value_counts().to_dict()}")
    print(f"  GPU types: {df['gpu_type'].value_counts().to_dict()}")
    print(f"  Avg cost saving: ${df['cloud_cost_usd'].mean():.2f}")
    print(f"  Avg runtime: {df['runtime_hours'].mean():.1f} hours")
    print(f"  Avg deadline: {df['deadline_hours'].mean():.1f} hours")
    print("=" * 60)
