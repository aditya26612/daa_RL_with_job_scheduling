"""
kaggle_application.py -- Kaggle GPU Cost-Optimized Job Scheduler
==================================================================
Real-world application of the RL Job Sequencing system:
Optimally schedules Kaggle notebook execution tasks to maximize
cloud computing cost savings under GPU quota and deadline constraints.

Usage:
    python kaggle_application.py

This script:
  1. Generates realistic Kaggle workload datasets (with CSV export)
  2. Trains the Dueling DQN agent on Kaggle job scheduling
  3. Evaluates against classical baselines
  4. Produces publication-quality visualizations
  5. Computes total cloud cost savings
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

os.environ['PYTHONIOENCODING'] = 'utf-8'

from config import Config, JobConfig, EnvironmentConfig, TrainingConfig, AgentConfig
from environment import JobSchedulingEnv
from agent import DQNAgent
from greedy_solver import run_all_baselines
from trainer import Trainer
from job import Job
from kaggle_data_generator import (
    generate_kaggle_dataset,
    generate_competition_scenarios,
    save_dataset,
    GPU_COST_PER_HOUR,
    KAGGLE_QUOTAS,
    WORKLOAD_PROFILES,
)

# ---- Plot style ----
plt.style.use('dark_background')
COLORS = {
    'rl': '#00D4AA',
    'greedy': '#FF6B6B',
    'edf': '#4ECDC4',
    'sjf': '#45B7D1',
    'density': '#FFA07A',
    'accent': '#FFD93D',
    'bg': '#1a1a2e',
    'grid': '#333355',
}
OUTPUT_DIR = "output/kaggle_application"


def setup_output():
    """Create output directories."""
    for d in [OUTPUT_DIR, f"{OUTPUT_DIR}/plots", f"{OUTPUT_DIR}/data"]:
        os.makedirs(d, exist_ok=True)


def run_rl_agent(agent, env, jobs):
    """Run trained RL agent on a job instance."""
    state, mask = env.reset(jobs)
    while not env.done:
        action = agent.select_action(state, mask, training=False)
        (state, mask), reward, done, info = env.step(action)
    return env.get_metrics(), env.get_schedule()


def evaluate_on_kaggle_data(agent, config, num_instances=30):
    """Evaluate RL agent vs baselines on Kaggle workload data."""
    env = JobSchedulingEnv(config.env)
    sizes = [30, 60, 100, 150]
    all_records = []

    for n_jobs in sizes:
        print(f"  Evaluating {n_jobs}-job Kaggle workloads...")
        for inst_id in range(num_instances):
            seed = 5000 + inst_id * 100 + n_jobs
            jobs, _ = generate_kaggle_dataset(n_jobs, seed=seed)

            # RL Agent
            metrics, _ = run_rl_agent(agent, env, jobs)
            all_records.append({
                "num_jobs": n_jobs, "instance": inst_id, "solver": "RL Agent (DQN)",
                "total_profit": metrics["total_profit"],
                "profit_ratio": metrics["profit_ratio"],
                "completion_rate": metrics["completion_rate"],
                "utilization": metrics["utilization"],
                "num_scheduled": metrics["num_scheduled"],
            })

            # Baselines
            baseline_results = run_all_baselines(jobs)
            for key, result in baseline_results.items():
                all_records.append({
                    "num_jobs": n_jobs, "instance": inst_id,
                    "solver": result["solver"],
                    "total_profit": result["total_profit"],
                    "profit_ratio": result["profit_ratio"],
                    "completion_rate": result["completion_rate"],
                    "utilization": result["utilization"],
                    "num_scheduled": result["num_scheduled"],
                })

    return pd.DataFrame(all_records)


# ================================================================
#  VISUALIZATION FUNCTIONS
# ================================================================

def plot_workload_distribution(df_dataset):
    """Plot the Kaggle workload type and GPU distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=COLORS['bg'])
    fig.suptitle("Kaggle Notebook Workload Distribution",
                 fontsize=16, fontweight='bold', color='white', y=1.02)

    for ax in axes:
        ax.set_facecolor(COLORS['bg'])

    # 1. Workload type pie chart
    wtype_counts = df_dataset['workload_type'].value_counts()
    labels = [w.replace('_', '\n') for w in wtype_counts.index]
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#FFD93D', '#96CEB4']
    axes[0].pie(wtype_counts.values, labels=labels, autopct='%1.1f%%',
                colors=colors_pie[:len(wtype_counts)], textprops={'fontsize': 8, 'color': 'white'})
    axes[0].set_title("Workload Types", color='white', fontsize=12)

    # 2. Runtime distribution
    for wtype in df_dataset['workload_type'].unique():
        subset = df_dataset[df_dataset['workload_type'] == wtype]
        axes[1].hist(subset['runtime_hours'], bins=15, alpha=0.6,
                     label=wtype.replace('_', ' ').title())
    axes[1].set_xlabel("Runtime (hours)", color='white')
    axes[1].set_ylabel("Count", color='white')
    axes[1].set_title("Runtime Distribution by Workload", color='white', fontsize=12)
    axes[1].legend(fontsize=7, loc='upper right')
    axes[1].tick_params(colors='white')

    # 3. Cost saving distribution
    for gpu in df_dataset['gpu_type'].unique():
        subset = df_dataset[df_dataset['gpu_type'] == gpu]
        axes[2].hist(subset['cloud_cost_usd'], bins=15, alpha=0.6, label=gpu)
    axes[2].set_xlabel("Cloud Cost Saving (USD)", color='white')
    axes[2].set_ylabel("Count", color='white')
    axes[2].set_title("Cost Saving by GPU Type", color='white', fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].tick_params(colors='white')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/plots/workload_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"  Saved: {path}")


def plot_gpu_pricing_comparison():
    """Plot GPU pricing across cloud providers."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    gpus = list(GPU_COST_PER_HOUR.keys())
    providers = ['aws', 'gcp', 'azure', 'kaggle_free']
    provider_labels = ['AWS', 'GCP', 'Azure', 'Kaggle (Free)']
    provider_colors = ['#FF9900', '#4285F4', '#0078D4', '#20BEFF']

    x = np.arange(len(gpus))
    width = 0.2

    for idx, (provider, label, color) in enumerate(zip(providers, provider_labels, provider_colors)):
        costs = [GPU_COST_PER_HOUR[gpu][provider] for gpu in gpus]
        bars = ax.bar(x + idx * width, costs, width, label=label, color=color, alpha=0.85)
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'${cost:.2f}', ha='center', va='bottom', fontsize=8, color='white')

    ax.set_xlabel("GPU Model", color='white', fontsize=12)
    ax.set_ylabel("Cost per Hour (USD)", color='white', fontsize=12)
    ax.set_title("Cloud GPU Hourly Pricing -- Kaggle vs. Commercial Providers",
                 color='white', fontsize=14, fontweight='bold')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(gpus, color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.2, color=COLORS['grid'])

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/plots/gpu_pricing_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"  Saved: {path}")


def plot_cost_savings_comparison(df_results):
    """Plot total cost savings: RL vs baselines."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=COLORS['bg'])
    fig.suptitle("Kaggle GPU Cost Savings -- RL Agent vs. Classical Baselines",
                 fontsize=15, fontweight='bold', color='white', y=1.02)

    solver_colors = {
        'RL Agent (DQN)': COLORS['rl'],
        'Greedy-by-Profit': COLORS['greedy'],
        'Earliest Deadline First': COLORS['edf'],
        'Shortest Job First': COLORS['sjf'],
        'Profit Density': COLORS['density'],
    }

    for ax in axes:
        ax.set_facecolor(COLORS['bg'])

    # 1. Average cost savings by solver and problem size
    agg = df_results.groupby(['num_jobs', 'solver'])['total_profit'].mean().reset_index()
    solvers = agg['solver'].unique()

    x_labels = sorted(agg['num_jobs'].unique())
    x = np.arange(len(x_labels))
    w = 0.15

    for idx, solver in enumerate(solvers):
        vals = [agg[(agg['num_jobs'] == n) & (agg['solver'] == solver)]['total_profit'].values
                for n in x_labels]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        color = solver_colors.get(solver, '#999999')
        axes[0].bar(x + idx * w, vals, w, label=solver, color=color, alpha=0.85)

    axes[0].set_xlabel("Number of Pending Jobs", color='white')
    axes[0].set_ylabel("Avg Cost Saved (USD)", color='white')
    axes[0].set_title("Total Cost Savings", color='white', fontsize=12)
    axes[0].set_xticks(x + 2 * w)
    axes[0].set_xticklabels(x_labels, color='white')
    axes[0].legend(fontsize=7, loc='upper left')
    axes[0].tick_params(colors='white')
    axes[0].grid(axis='y', alpha=0.2, color=COLORS['grid'])
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # 2. Improvement % over baselines (at largest size)
    largest = max(x_labels)
    sub = df_results[df_results['num_jobs'] == largest]
    avg_by_solver = sub.groupby('solver')['total_profit'].mean()
    rl_profit = avg_by_solver.get('RL Agent (DQN)', 0)

    improvements = {}
    for solver, profit in avg_by_solver.items():
        if solver != 'RL Agent (DQN)' and profit > 0:
            improvements[solver] = ((rl_profit - profit) / profit) * 100

    solver_names = list(improvements.keys())
    pcts = list(improvements.values())
    bar_colors = [COLORS['rl'] if p >= 0 else COLORS['greedy'] for p in pcts]

    bars = axes[1].barh(solver_names, pcts, color=bar_colors, alpha=0.85)
    for bar, pct in zip(bars, pcts):
        sign = '+' if pct >= 0 else ''
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f'{sign}{pct:.1f}%', va='center', color='white', fontsize=10)

    axes[1].axvline(x=0, color='white', linewidth=0.5)
    axes[1].set_xlabel("Improvement (%)", color='white')
    axes[1].set_title(f"RL vs Baselines ({largest} Jobs)", color='white', fontsize=12)
    axes[1].tick_params(colors='white')
    axes[1].grid(axis='x', alpha=0.2, color=COLORS['grid'])

    # 3. Weekly cost savings projection
    weekly_scenarios = {
        "Light\n(30 jobs/wk)": 30,
        "Moderate\n(60 jobs/wk)": 60,
        "Heavy\n(100 jobs/wk)": 100,
        "Intensive\n(150 jobs/wk)": 150,
    }

    rl_weekly = []
    best_baseline_weekly = []
    for label, n in weekly_scenarios.items():
        sub = df_results[df_results['num_jobs'] == n] if n in df_results['num_jobs'].values else \
              df_results[df_results['num_jobs'] == min(x_labels, key=lambda x: abs(x - n))]
        solver_avg = sub.groupby('solver')['total_profit'].mean()
        rl_weekly.append(solver_avg.get('RL Agent (DQN)', 0))
        baselines_only = {k: v for k, v in solver_avg.items() if k != 'RL Agent (DQN)'}
        best_baseline_weekly.append(max(baselines_only.values()) if baselines_only else 0)

    labels_list = list(weekly_scenarios.keys())
    x2 = np.arange(len(labels_list))

    axes[2].bar(x2 - 0.15, rl_weekly, 0.3, label='RL Agent', color=COLORS['rl'], alpha=0.85)
    axes[2].bar(x2 + 0.15, best_baseline_weekly, 0.3, label='Best Baseline', color=COLORS['density'], alpha=0.85)

    axes[2].set_xlabel("Weekly Workload", color='white')
    axes[2].set_ylabel("Weekly Savings (USD)", color='white')
    axes[2].set_title("Projected Weekly Cost Savings", color='white', fontsize=12)
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(labels_list, color='white', fontsize=8)
    axes[2].legend(fontsize=9)
    axes[2].tick_params(colors='white')
    axes[2].grid(axis='y', alpha=0.2, color=COLORS['grid'])
    axes[2].yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f'))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/plots/cost_savings_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"  Saved: {path}")


def plot_performance_heatmap(df_results):
    """Heatmap of solver performance across problem sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=COLORS['bg'])
    fig.suptitle("Performance Heatmap -- Profit Ratio & Completion Rate",
                 fontsize=14, fontweight='bold', color='white', y=1.02)

    # Profit ratio heatmap
    pivot_profit = df_results.pivot_table(
        values='total_profit', index='solver', columns='num_jobs', aggfunc='mean'
    )
    sns.heatmap(pivot_profit, annot=True, fmt='.1f', cmap='YlOrRd',
                ax=axes[0], cbar_kws={'label': 'Avg Cost Saved (USD)'},
                linewidths=1, linecolor=COLORS['bg'])
    axes[0].set_title("Average Cost Savings (USD)", color='white', fontsize=12)
    axes[0].set_xlabel("Number of Jobs", color='white')
    axes[0].set_ylabel("")
    axes[0].tick_params(colors='white')
    axes[0].set_facecolor(COLORS['bg'])

    # Completion rate heatmap
    pivot_comp = df_results.pivot_table(
        values='completion_rate', index='solver', columns='num_jobs', aggfunc='mean'
    )
    sns.heatmap(pivot_comp, annot=True, fmt='.2%', cmap='YlGnBu',
                ax=axes[1], cbar_kws={'label': 'Completion Rate'},
                linewidths=1, linecolor=COLORS['bg'])
    axes[1].set_title("Job Completion Rate", color='white', fontsize=12)
    axes[1].set_xlabel("Number of Jobs", color='white')
    axes[1].set_ylabel("")
    axes[1].tick_params(colors='white')
    axes[1].set_facecolor(COLORS['bg'])

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/plots/kaggle_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"  Saved: {path}")


def plot_gantt_kaggle(agent, config, jobs, df_jobs):
    """Gantt chart showing RL schedule with workload type colors."""
    env = JobSchedulingEnv(config.env)
    metrics, schedule = run_rl_agent(agent, env, jobs)

    # Also get Profit Density schedule for comparison
    baseline_results = run_all_baselines(jobs)
    pd_schedule = baseline_results.get('profit_density', {}).get('schedule', [])

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), facecolor=COLORS['bg'])
    fig.suptitle("Schedule Comparison -- Kaggle GPU Job Allocation",
                 fontsize=14, fontweight='bold', color='white', y=1.02)

    wtype_colors = {
        'data_preprocessing': '#4ECDC4',
        'model_training_small': '#45B7D1',
        'model_training_large': '#FF6B6B',
        'fine_tuning_llm': '#FFD93D',
        'inference_pipeline': '#96CEB4',
        'ensemble_submission': '#FFA07A',
    }

    schedules = [
        ("RL Agent (DQN)", schedule, metrics['total_profit']),
        ("Profit Density", pd_schedule, baseline_results.get('profit_density', {}).get('total_profit', 0)),
    ]

    for ax_idx, (title, sched, profit) in enumerate(schedules):
        ax = axes[ax_idx]
        ax.set_facecolor(COLORS['bg'])

        for job, start, end in sched:
            # Look up workload type from dataset
            job_row = df_jobs[df_jobs['job_id'] == job.job_id]
            wtype = job_row['workload_type'].iloc[0] if len(job_row) > 0 else 'model_training_small'
            color = wtype_colors.get(wtype, '#999999')

            ax.barh(0, end - start, left=start, height=0.6,
                    color=color, edgecolor='white', linewidth=0.5, alpha=0.85)
            # Label with job ID and cost
            if end - start > 1:
                ax.text(start + (end - start) / 2, 0,
                        f"J{job.job_id}\n${job.profit:.1f}",
                        ha='center', va='center', fontsize=7,
                        color='white', fontweight='bold')

        ax.set_xlim(0, max(end for _, _, end in sched) + 2 if sched else 30)
        ax.set_yticks([])
        ax.set_xlabel("Time (30-min slots)", color='white')
        ax.set_title(f"{title} -- Savings: ${profit:.2f} | Jobs: {len(sched)}",
                     color=COLORS['rl'] if ax_idx == 0 else COLORS['density'],
                     fontsize=11, fontweight='bold')
        ax.tick_params(colors='white')
        ax.grid(axis='x', alpha=0.2, color=COLORS['grid'])

    # Legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=c, label=k.replace('_', ' ').title())
                       for k, c in wtype_colors.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6,
               fontsize=8, framealpha=0.3, facecolor=COLORS['bg'],
               edgecolor='white', labelcolor='white')

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    path = f"{OUTPUT_DIR}/plots/kaggle_gantt.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"  Saved: {path}")


def plot_deadline_analysis(df_dataset):
    """Visualize deadline distribution and job density."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLORS['bg'])
    fig.suptitle("Kaggle Competition Deadline & Job Density Analysis",
                 fontsize=14, fontweight='bold', color='white', y=1.02)

    for ax in axes:
        ax.set_facecolor(COLORS['bg'])

    # 1. Deadline distribution by urgency
    urgency_colors = {'urgent': '#FF4444', 'soon': '#FFA500', 'moderate': '#4ECDC4', 'relaxed': '#45B7D1'}
    for dtype in df_dataset['deadline_type'].unique():
        subset = df_dataset[df_dataset['deadline_type'] == dtype]
        axes[0].hist(subset['deadline_hours'], bins=20, alpha=0.6,
                     color=urgency_colors.get(dtype, '#999'),
                     label=dtype.title())
    axes[0].set_xlabel("Deadline (hours)", color='white')
    axes[0].set_ylabel("Number of Jobs", color='white')
    axes[0].set_title("Job Deadline Distribution", color='white', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].tick_params(colors='white')
    axes[0].grid(alpha=0.2, color=COLORS['grid'])

    # 2. Cost/hour density scatter
    scatter = axes[1].scatter(
        df_dataset['runtime_hours'],
        df_dataset['cloud_cost_usd'],
        c=df_dataset['deadline_hours'],
        cmap='RdYlGn', s=50, alpha=0.7, edgecolors='white', linewidth=0.3
    )
    plt.colorbar(scatter, ax=axes[1], label='Deadline (hours)')
    axes[1].set_xlabel("Runtime (hours)", color='white')
    axes[1].set_ylabel("Cloud Cost Saving (USD)", color='white')
    axes[1].set_title("Cost vs. Runtime (colored by deadline)", color='white', fontsize=12)
    axes[1].tick_params(colors='white')
    axes[1].grid(alpha=0.2, color=COLORS['grid'])

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/plots/deadline_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"  Saved: {path}")


def plot_annual_savings_projection(df_results):
    """Project annual savings from using RL scheduler."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    # Calculate weekly savings at moderate load (100 jobs)
    n = 100
    sub = df_results[df_results['num_jobs'] == n] if n in df_results['num_jobs'].values else df_results
    solver_avg = sub.groupby('solver')['total_profit'].mean()

    solvers_ordered = solver_avg.sort_values(ascending=True)
    weeks = 52
    annual = solvers_ordered * weeks

    colors_list = []
    for s in solvers_ordered.index:
        if 'RL' in s:
            colors_list.append(COLORS['rl'])
        elif 'Greedy' in s:
            colors_list.append(COLORS['greedy'])
        elif 'Earliest' in s:
            colors_list.append(COLORS['edf'])
        elif 'Shortest' in s:
            colors_list.append(COLORS['sjf'])
        else:
            colors_list.append(COLORS['density'])

    bars = ax.barh(solvers_ordered.index, annual.values, color=colors_list, alpha=0.85,
                   edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, annual.values):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'${val:,.0f}/yr', va='center', color='white', fontsize=11, fontweight='bold')

    ax.set_xlabel("Projected Annual Savings (USD)", color='white', fontsize=12)
    ax.set_title("Annual Cloud Cost Savings Projection (100 jobs/week x 52 weeks)",
                 color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.grid(axis='x', alpha=0.2, color=COLORS['grid'])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/plots/annual_savings.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
#  MAIN APPLICATION
# ================================================================

def main():
    print("=" * 70)
    print("  KAGGLE GPU COST-OPTIMIZED JOB SCHEDULER")
    print("  Application of RL-based Job Sequencing with Deadlines")
    print("=" * 70)

    setup_output()
    start_time = time.time()

    # ------ Phase 1: Data Generation ------
    print("\n--- Phase 1: Generating Kaggle Workload Datasets ---")
    jobs_main, df_main = generate_kaggle_dataset(num_jobs=150, seed=42)
    save_dataset(df_main, f"{OUTPUT_DIR}/data/kaggle_main_dataset.csv")

    scenarios = generate_competition_scenarios(seed=42)
    for name, (jobs_s, df_s) in scenarios.items():
        save_dataset(df_s, f"{OUTPUT_DIR}/data/kaggle_{name}.csv")

    print(f"\n  Dataset Summary:")
    print(f"    Total jobs generated: {len(df_main)}")
    print(f"    Workload types: {len(df_main['workload_type'].unique())}")
    print(f"    GPU types used: {list(df_main['gpu_type'].unique())}")
    print(f"    Avg cloud cost per job: ${df_main['cloud_cost_usd'].mean():.2f}")
    print(f"    Total possible savings: ${df_main['cloud_cost_usd'].sum():.2f}")

    # ------ Phase 2: Visualize Dataset ------
    print("\n--- Phase 2: Dataset Visualizations ---")
    plot_workload_distribution(df_main)
    plot_gpu_pricing_comparison()
    plot_deadline_analysis(df_main)

    # ------ Phase 3: Train RL Agent ------
    print("\n--- Phase 3: Training DQN Agent on Kaggle Workloads ---")

    config = Config()
    config.training.num_episodes = 600
    config.training.use_curriculum = True
    config.training.curriculum_stages = [
        (0, 20),
        (150, 50),
        (300, 80),
        (450, 100),
    ]
    config.training.eval_freq = 50
    config.training.checkpoint_freq = 200

    # Override dataset generation in trainer to use Kaggle data
    trainer = Trainer(config)

    # Monkey-patch the trainer to use Kaggle data generator
    original_run_episode = trainer._run_episode
    def kaggle_run_episode(num_jobs, training=True):
        from kaggle_data_generator import generate_kaggle_dataset as gen_kag
        seed = np.random.randint(0, 100000) if training else 99999
        jobs, _ = gen_kag(num_jobs, seed=seed)
        state, mask = trainer.env.reset(jobs)
        episode_reward = 0.0
        episode_loss = 0.0
        num_steps = 0
        train_steps = 0

        while not trainer.env.done:
            action = trainer.agent.select_action(state, mask, training=training)
            (next_state, next_mask), reward, done, info = trainer.env.step(action)
            if training:
                trainer.agent.store_transition(state, mask, action, reward,
                                               next_state, next_mask, done)
                loss = trainer.agent.train_step()
                if loss is not None:
                    episode_loss += loss
                    train_steps += 1
                if trainer.agent.step_count % config.training.target_update_freq == 0:
                    trainer.agent.update_target_network()
            state, mask = next_state, next_mask
            episode_reward += reward
            num_steps += 1

        metrics = trainer.env.get_metrics()
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

    trainer._run_episode = kaggle_run_episode
    training_history = trainer.train()
    # Merge in agent's loss/q-value history for complete training curves
    full_history = trainer.get_training_history()
    training_history.update(full_history)

    agent = trainer.agent

    # ------ Phase 4: Evaluate on Kaggle Data ------
    print("\n--- Phase 4: Evaluation on Kaggle Workloads ---")
    df_results = evaluate_on_kaggle_data(agent, config, num_instances=30)
    df_results.to_csv(f"{OUTPUT_DIR}/data/kaggle_evaluation_results.csv", index=False)

    # Print summary table
    print("\n  Evaluation Summary (Avg Cost Saved in USD):")
    summary = df_results.groupby(['num_jobs', 'solver'])['total_profit'].mean().unstack()
    print(summary.round(2).to_string())

    # Improvement summary
    print("\n  RL Agent Improvement Over Baselines:")
    for n in sorted(df_results['num_jobs'].unique()):
        sub = df_results[df_results['num_jobs'] == n]
        avg = sub.groupby('solver')['total_profit'].mean()
        rl_val = avg.get('RL Agent (DQN)', 0)
        for solver, val in avg.items():
            if solver != 'RL Agent (DQN)' and val > 0:
                pct = ((rl_val - val) / val) * 100
                sign = '+' if pct >= 0 else ''
                print(f"    [{n} jobs] vs {solver}: {sign}{pct:.1f}%")

    # ------ Phase 5: Generate All Visualizations ------
    print("\n--- Phase 5: Generating Application Visualizations ---")
    plot_cost_savings_comparison(df_results)
    plot_performance_heatmap(df_results)
    plot_annual_savings_projection(df_results)

    # Gantt chart with sample data
    sample_jobs, sample_df = generate_kaggle_dataset(30, seed=777)
    plot_gantt_kaggle(agent, config, sample_jobs, sample_df)

    # Training curves (reuse from trainer)
    from visualizer import plot_training_curves as viz_training
    viz_training(training_history, save_dir=f"{OUTPUT_DIR}/plots")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  APPLICATION COMPLETE")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Plots: {len(os.listdir(f'{OUTPUT_DIR}/plots'))} files")
    print(f"  Data: {len(os.listdir(f'{OUTPUT_DIR}/data'))} files")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
