"""
visualizer.py — Visualization Module for Job Scheduling Results
================================================================
Produces publication-quality plots:
1. Training reward/loss curves
2. Gantt chart of job schedules
3. Bar chart comparison across solvers
4. Heatmap of performance vs. problem size
5. Epsilon decay and Q-value evolution curves
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import os

from job import Job


# ─── Plot Style Configuration ───────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
    'font.size': 11,
    'axes.titlesize': 14,
    'figure.titlesize': 16,
    'legend.fontsize': 10,
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': '#0d1117',
})

# Color palette (GitHub dark theme inspired)
COLORS = {
    'rl_agent': '#58a6ff',      # Blue
    'greedy_profit': '#f78166',  # Orange
    'edf': '#7ee787',            # Green
    'sjf': '#d2a8ff',            # Purple
    'profit_density': '#ffa657', # Amber
    'accent': '#ff7b72',         # Red accent
    'reward': '#58a6ff',
    'loss': '#f78166',
    'profit': '#7ee787',
}

SOLVER_NAMES = {
    'rl_agent': 'RL Agent (DQN)',
    'greedy_profit': 'Greedy by Profit',
    'edf': 'Earliest Deadline First',
    'sjf': 'Shortest Job First',
    'profit_density': 'Profit Density',
}


def _smooth(data: List[float], window: int = 20) -> np.ndarray:
    """Apply moving average smoothing.
    
    Args:
        data: Raw data series
        window: Smoothing window size
        
    Returns:
        Smoothed data array
    """
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_training_curves(history: Dict, save_dir: str = "output/plots"):
    """Plot training reward and loss curves.
    
    Creates a 2×2 subplot figure with:
    - Episode rewards (raw + smoothed)
    - Episode profits
    - Training loss
    - Epsilon decay schedule
    
    Args:
        history: Training history dictionary from Trainer
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('DQN Training Progress - Job Sequencing with Deadlines', 
                 fontweight='bold', fontsize=16)
    
    # ── Reward Curve ──
    ax = axes[0, 0]
    rewards = history.get("episode_rewards", [])
    if rewards:
        ax.plot(rewards, alpha=0.2, color=COLORS['reward'], linewidth=0.5)
        smoothed = _smooth(rewards, min(50, len(rewards) // 5 + 1))
        ax.plot(range(len(rewards) - len(smoothed), len(rewards)), 
                smoothed, color=COLORS['reward'], linewidth=2, label='Smoothed')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # ── Profit Curve ──
    ax = axes[0, 1]
    profits = history.get("episode_profits", [])
    if profits:
        ax.plot(profits, alpha=0.2, color=COLORS['profit'], linewidth=0.5)
        smoothed = _smooth(profits, min(50, len(profits) // 5 + 1))
        ax.plot(range(len(profits) - len(smoothed), len(profits)),
                smoothed, color=COLORS['profit'], linewidth=2, label='Smoothed')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Profit')
        ax.set_title('Episode Profits')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # ── Loss Curve ──
    ax = axes[1, 0]
    losses = history.get("losses", [])
    if losses:
        smoothed_loss = _smooth(losses, min(100, len(losses) // 10 + 1))
        ax.plot(smoothed_loss, color=COLORS['loss'], linewidth=1.5)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss (Huber)')
        ax.set_title('Training Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # ── Q-Value Evolution ──
    ax = axes[1, 1]
    q_values = history.get("q_values", [])
    if q_values:
        smoothed_q = _smooth(q_values, min(100, len(q_values) // 10 + 1))
        ax.plot(smoothed_q, color=COLORS['accent'], linewidth=1.5)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Mean Q-Value')
        ax.set_title('Mean Q-Value Evolution')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, "training_curves.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  ✅ Saved: {filepath}")
    return filepath


def plot_gantt_chart(schedules: Dict[str, List[Tuple]], num_jobs: int,
                     save_dir: str = "output/plots"):
    """Plot Gantt charts comparing schedules from different solvers.
    
    Each solver gets a row showing its job schedule as colored bars
    on a shared timeline.
    
    Args:
        schedules: Dict mapping solver name to list of (job, start, end)
        num_jobs: Total number of jobs in the instance
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_solvers = len(schedules)
    fig, axes = plt.subplots(n_solvers, 1, figsize=(16, 3 * n_solvers + 1),
                              sharex=True)
    if n_solvers == 1:
        axes = [axes]
    
    fig.suptitle('Schedule Comparison - Gantt Charts', fontweight='bold', fontsize=16)
    
    # Color map for jobs
    cmap = plt.cm.get_cmap('plasma', num_jobs)
    
    solver_keys = list(schedules.keys())
    for idx, (solver_name, schedule) in enumerate(schedules.items()):
        ax = axes[idx]
        color_key = solver_keys[idx] if solver_keys[idx] in COLORS else 'rl_agent'
        
        for job, start, end in schedule:
            color = cmap(job.job_id % num_jobs)
            ax.barh(0, end - start, left=start, height=0.6,
                    color=color, edgecolor='white', linewidth=0.5, alpha=0.85)
            
            # Label the job if bar is wide enough
            if end - start >= 2:
                ax.text((start + end) / 2, 0, f"J{job.job_id}\n${job.profit:.0f}",
                       ha='center', va='center', fontsize=7, color='white',
                       fontweight='bold')
        
        total_profit = sum(j.profit for j, _, _ in schedule)
        display_name = SOLVER_NAMES.get(solver_name, solver_name)
        ax.set_ylabel(display_name, fontsize=10, rotation=0, ha='right', va='center')
        ax.set_yticks([])
        ax.set_title(f"{display_name} — Profit: {total_profit:.1f} | "
                     f"Jobs: {len(schedule)}/{num_jobs}", fontsize=11, loc='left')
        ax.grid(True, axis='x', alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    filepath = os.path.join(save_dir, "gantt_chart.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  ✅ Saved: {filepath}")
    return filepath


def plot_solver_comparison(df: pd.DataFrame, save_dir: str = "output/plots"):
    """Plot bar chart comparing average profit across solvers and sizes.
    
    Creates a grouped bar chart where each group is a problem size
    and each bar within a group represents a solver.
    
    Args:
        df: Results DataFrame from Evaluator
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Solver Comparison Across Problem Sizes', fontweight='bold', fontsize=16)
    
    metrics = [
        ("total_profit", "Average Total Profit", ""),
        ("completion_rate", "Job Completion Rate", ""),
        ("utilization", "Machine Utilization", ""),
    ]
    
    for ax_idx, (metric, title, emoji) in enumerate(metrics):
        ax = axes[ax_idx]
        
        sizes = sorted(df["num_jobs"].unique())
        solvers = df["solver_key"].unique()
        
        x = np.arange(len(sizes))
        width = 0.15
        
        for i, solver in enumerate(solvers):
            solver_data = df[df["solver_key"] == solver]
            means = [solver_data[solver_data["num_jobs"] == s][metric].mean() 
                     for s in sizes]
            
            color = COLORS.get(solver, '#8b949e')
            label = SOLVER_NAMES.get(solver, solver)
            
            ax.bar(x + i * width - (len(solvers) - 1) * width / 2, means,
                   width, label=label, color=color, alpha=0.85,
                   edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Number of Jobs')
        ax.set_ylabel(title)
        ax.set_title(f'{title}')
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, "solver_comparison.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  ✅ Saved: {filepath}")
    return filepath


def plot_performance_heatmap(df: pd.DataFrame, save_dir: str = "output/plots"):
    """Plot heatmap of profit ratio across solvers and problem sizes.
    
    Args:
        df: Results DataFrame from Evaluator
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Pivot table: solver vs problem size
    pivot = df.groupby(["solver_key", "num_jobs"])["profit_ratio"].mean().reset_index()
    pivot_table = pivot.pivot(index="solver_key", columns="num_jobs", values="profit_ratio")
    
    # Rename index
    pivot_table.index = [SOLVER_NAMES.get(s, s) for s in pivot_table.index]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(
        pivot_table, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=2, linecolor='#0d1117',
        cbar_kws={'label': 'Profit Ratio (higher = better)'},
        ax=ax, vmin=0, vmax=1,
        annot_kws={'fontsize': 13, 'fontweight': 'bold'}
    )
    
    ax.set_title('Performance Heatmap - Profit Ratio by Solver x Problem Size',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Jobs', fontsize=12)
    ax.set_ylabel('Solver', fontsize=12)
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, "performance_heatmap.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  ✅ Saved: {filepath}")
    return filepath


def plot_evaluation_progress(eval_history: List[Dict], save_dir: str = "output/plots"):
    """Plot evaluation metrics over training episodes.
    
    Shows how the agent's performance on held-out instances
    improves during training.
    
    Args:
        eval_history: List of evaluation result dicts from training
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if not eval_history:
        return None
    
    episodes = [e["episode"] for e in eval_history]
    profits = [e.get("profit", 0) for e in eval_history]
    ratios = [e.get("profit_ratio", 0) for e in eval_history]
    completions = [e.get("completion_rate", 0) for e in eval_history]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Evaluation Progress During Training', fontweight='bold', fontsize=16)
    
    # Profit
    axes[0].plot(episodes, profits, color=COLORS['profit'], linewidth=2, marker='o', markersize=3)
    axes[0].fill_between(episodes, profits, alpha=0.1, color=COLORS['profit'])
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Profit')
    axes[0].set_title('Evaluation Profit')
    axes[0].grid(True, alpha=0.3)
    
    # Profit Ratio
    axes[1].plot(episodes, ratios, color=COLORS['reward'], linewidth=2, marker='o', markersize=3)
    axes[1].fill_between(episodes, ratios, alpha=0.1, color=COLORS['reward'])
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Profit Ratio')
    axes[1].set_title('Profit Ratio')
    axes[1].grid(True, alpha=0.3)
    
    # Completion Rate
    axes[2].plot(episodes, completions, color=COLORS['accent'], linewidth=2, marker='o', markersize=3)
    axes[2].fill_between(episodes, completions, alpha=0.1, color=COLORS['accent'])
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Completion Rate')
    axes[2].set_title('Job Completion Rate')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, "evaluation_progress.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  ✅ Saved: {filepath}")
    return filepath


def plot_all(training_history: Dict, eval_df: pd.DataFrame,
             example_schedules: Dict = None, num_jobs: int = 50,
             save_dir: str = "output/plots") -> List[str]:
    """Generate all plots.
    
    Args:
        training_history: From Trainer.get_training_history()
        eval_df: From Evaluator.evaluate_batch()
        example_schedules: Optional dict of schedules for Gantt chart
        num_jobs: Number of jobs for Gantt chart
        save_dir: Directory to save plots
        
    Returns:
        List of saved file paths
    """
    print("\n" + "=" * 50)
    print("  Generating Visualizations")
    print("=" * 50)
    
    saved = []
    
    # Training curves
    path = plot_training_curves(training_history, save_dir)
    if path:
        saved.append(path)
    
    # Evaluation progress
    path = plot_evaluation_progress(
        training_history.get("eval_history", []), save_dir
    )
    if path:
        saved.append(path)
    
    # Solver comparison
    if eval_df is not None and len(eval_df) > 0:
        path = plot_solver_comparison(eval_df, save_dir)
        if path:
            saved.append(path)
        
        path = plot_performance_heatmap(eval_df, save_dir)
        if path:
            saved.append(path)
    
    # Gantt chart
    if example_schedules:
        path = plot_gantt_chart(example_schedules, num_jobs, save_dir)
        if path:
            saved.append(path)
    
    print(f"\n  Total plots saved: {len(saved)}")
    return saved
