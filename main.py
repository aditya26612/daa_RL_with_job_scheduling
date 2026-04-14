"""
main.py -- Entry Point for RL Job Sequencing Project
=====================================================
Runs the complete pipeline: train the DQN agent, evaluate against
baselines, generate visualizations, and print results.

Usage:
    python main.py                    # Full demo (train + evaluate + plots)
    python main.py --mode train       # Train only
    python main.py --mode evaluate    # Evaluate only (requires trained model)
    python main.py --mode demo        # Quick demo with fewer episodes
"""

import argparse
import numpy as np
import torch
import os
import sys
import time
os.environ['PYTHONIOENCODING'] = 'utf-8'

from rich.console import Console
from rich.panel import Panel
from rich.progress import track

from config import Config
from trainer import Trainer
from evaluator import Evaluator
from visualizer import plot_all, plot_gantt_chart
from dataset_loader import generate_synthetic, get_dataset, generate_difficulty_levels
from greedy_solver import run_all_baselines
from agent import DQNAgent
from environment import JobSchedulingEnv
from job import JobSet


console = Console(force_terminal=True)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_banner():
    """Print a styled project banner."""
    banner = (
        "\n"
        "  Reinforcement-Learned Strategy for Job Sequencing\n"
        "  with Deadlines\n"
        "\n"
        "  Algorithm: Dueling DQN + Prioritized Experience Replay\n"
        "  Baselines: Greedy, EDF, SJF, Profit Density\n"
    )
    console.print(Panel(banner, border_style="bright_blue", title="[bold]RL Job Scheduler[/bold]"))


def print_job_instance(jobs, title="Sample Job Instance"):
    """Print a formatted job instance."""
    from rich.table import Table
    
    table = Table(title=f"{title} ({len(jobs)} jobs)", 
                  border_style="bright_blue", show_lines=False)
    table.add_column("Job ID", style="cyan", justify="center")
    table.add_column("Profit", style="green", justify="right")
    table.add_column("Deadline", style="yellow", justify="right")
    table.add_column("Proc Time", style="magenta", justify="right")
    table.add_column("Density", style="blue", justify="right")
    
    for job in jobs[:15]:  # Show first 15 jobs
        table.add_row(
            str(job.job_id),
            f"{job.profit:.1f}",
            str(job.deadline),
            str(job.processing_time),
            f"{job.density:.2f}",
        )
    
    if len(jobs) > 15:
        table.add_row("...", "...", "...", "...", "...")
    
    console.print(table)


def run_baseline_demo():
    """Run and display baseline solver results on a sample instance."""
    console.print("\n[bold cyan]--- Phase 1: Baseline Solver Demonstration ---[/bold cyan]")
    
    # Generate sample instance
    jobs = generate_synthetic(30, seed=42)
    print_job_instance(jobs)
    
    # Run all baselines
    console.print("\n[bold]Running classical scheduling algorithms...[/bold]")
    results = run_all_baselines(jobs)
    
    from rich.table import Table
    table = Table(title="Baseline Results on Sample Instance",
                  border_style="bright_green")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Total Profit", style="green", justify="right")
    table.add_column("Jobs Scheduled", style="yellow", justify="right")
    table.add_column("Completion %", style="blue", justify="right")
    table.add_column("Utilization", style="magenta", justify="right")
    
    for name, result in results.items():
        table.add_row(
            result["solver"],
            f"{result['total_profit']:.1f}",
            f"{result['num_scheduled']}/{result['num_total']}",
            f"{result['completion_rate']:.1%}",
            f"{result['utilization']:.3f}",
        )
    
    console.print(table)
    
    return jobs, results


def run_training(config: Config) -> Trainer:
    """Train the DQN agent."""
    console.print("\n[bold cyan]--- Phase 2: DQN Agent Training ---[/bold cyan]")
    
    trainer = Trainer(config)
    training_history = trainer.train()
    
    return trainer


def run_evaluation(agent: DQNAgent, config: Config, 
                   baseline_results: dict = None,
                   sample_jobs: list = None) -> dict:
    """Evaluate the trained agent against baselines."""
    console.print("\n[bold cyan]--- Phase 3: Comprehensive Evaluation ---[/bold cyan]")
    
    evaluator = Evaluator(agent, config)
    
    # Evaluate on synthetic data
    console.print("\n[bold]Evaluating on synthetic dataset...[/bold]")
    df_synthetic = evaluator.evaluate_batch(
        "synthetic", test_sizes=[20, 50, 100], num_instances=30
    )
    evaluator.print_comparison_table(df_synthetic, "synthetic")
    
    # Evaluate on Google-like data
    console.print("\n[bold]Evaluating on Google Cluster-like dataset...[/bold]")
    df_google = evaluator.evaluate_batch(
        "google", test_sizes=[20, 50, 100], num_instances=30
    )
    evaluator.print_comparison_table(df_google, "google")
    
    # Evaluate on Alibaba-like data
    console.print("\n[bold]Evaluating on Alibaba Cluster-like dataset...[/bold]")
    df_alibaba = evaluator.evaluate_batch(
        "alibaba", test_sizes=[20, 50, 100], num_instances=30
    )
    evaluator.print_comparison_table(df_alibaba, "alibaba")
    
    # Print overall summary
    evaluator.print_summary()
    
    # Generate example schedule for Gantt chart
    example_schedules = {}
    if sample_jobs:
        # RL agent schedule
        env = JobSchedulingEnv(config.env)
        state, mask = env.reset(sample_jobs)
        while not env.done:
            action = agent.select_action(state, mask, training=False)
            (state, mask), _, _, _ = env.step(action)
        example_schedules["rl_agent"] = env.get_schedule()
        
        # Baseline schedules
        if baseline_results:
            for key, result in baseline_results.items():
                example_schedules[key] = result["schedule"]
    
    return {
        "df_synthetic": df_synthetic,
        "df_google": df_google,
        "df_alibaba": df_alibaba,
        "example_schedules": example_schedules,
        "evaluator": evaluator,
    }


def run_visualizations(training_history: dict, eval_results: dict,
                       config: Config, sample_jobs_count: int = 30):
    """Generate all visualizations."""
    console.print("\n[bold cyan]--- Phase 4: Generating Visualizations ---[/bold cyan]")
    
    # Use synthetic results for the main comparison plot
    eval_df = eval_results.get("df_synthetic")
    example_schedules = eval_results.get("example_schedules", {})
    
    saved_plots = plot_all(
        training_history, eval_df,
        example_schedules=example_schedules,
        num_jobs=sample_jobs_count,
        save_dir=config.paths.plots_dir
    )
    
    # Also generate plots for Google and Alibaba if available
    from visualizer import plot_solver_comparison, plot_performance_heatmap
    
    for dataset_key in ["df_google", "df_alibaba"]:
        df = eval_results.get(dataset_key)
        if df is not None and len(df) > 0:
            name = dataset_key.replace("df_", "")
            plot_solver_comparison(df, config.paths.plots_dir)
            plot_performance_heatmap(df, config.paths.plots_dir)
    
    return saved_plots


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RL Job Sequencing with Deadlines — Dueling DQN Agent"
    )
    parser.add_argument("--mode", type=str, default="demo",
                        choices=["train", "evaluate", "demo"],
                        help="Execution mode")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training episodes (overrides config)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Configuration
    config = Config(seed=args.seed)
    
    # Adjust episodes for demo mode
    if args.mode == "demo":
        config.training.num_episodes = args.episodes or 500
        config.training.eval_freq = 25
        config.training.early_stop_patience = 200
        config.training.curriculum_stages = [
            (0, 20),
            (150, 50),
            (350, 100),
        ]
        config.eval.num_test_instances = 30
    elif args.episodes:
        config.training.num_episodes = args.episodes
    
    set_seed(config.seed)
    config.paths.create_dirs()
    
    console.print(f"[dim]Mode: {args.mode} | Seed: {args.seed} | "
                  f"Episodes: {config.training.num_episodes} | "
                  f"Device: {config.device}[/dim]\n")
    
    # ────────────────────────────────────────────────────────────
    # Phase 1: Baseline Demo
    # ────────────────────────────────────────────────────────────
    sample_jobs, baseline_results = run_baseline_demo()
    
    # ────────────────────────────────────────────────────────────
    # Phase 2: Training
    # ────────────────────────────────────────────────────────────
    if args.mode in ["train", "demo"]:
        trainer = run_training(config)
        training_history = trainer.get_training_history()
        agent = trainer.agent
    elif args.mode == "evaluate":
        # Load trained model
        model_path = f"{config.paths.checkpoint_dir}/best_model.pt"
        if not os.path.exists(model_path):
            model_path = f"{config.paths.checkpoint_dir}/final_model.pt"
        
        if not os.path.exists(model_path):
            console.print("[red]Error: No trained model found. Run training first.[/red]")
            sys.exit(1)
        
        env = JobSchedulingEnv(config.env)
        agent = DQNAgent(env.state_dim, env.action_dim, config.agent, config.device)
        agent.load(model_path)
        training_history = {"episode_rewards": [], "episode_profits": [], 
                           "eval_history": [], "losses": [], "q_values": []}
        console.print(f"[green]Loaded model from {model_path}[/green]")
    
    # ────────────────────────────────────────────────────────────
    # Phase 3: Evaluation
    # ────────────────────────────────────────────────────────────
    eval_results = run_evaluation(
        agent, config,
        baseline_results=baseline_results,
        sample_jobs=sample_jobs
    )
    
    # ────────────────────────────────────────────────────────────
    # Phase 4: Visualization
    # ────────────────────────────────────────────────────────────
    saved_plots = run_visualizations(
        training_history, eval_results, config,
        sample_jobs_count=len(sample_jobs)
    )
    
    # ────────────────────────────────────────────────────────────
    # Final Summary
    # ────────────────────────────────────────────────────────────
    console.print("\n" + "=" * 70)
    console.print(Panel(
        f"[bold green]Pipeline Complete![/bold green]\n\n"
        f"   Output directory: {os.path.abspath(config.paths.output_dir)}\n"
        f"   Plots saved: {len(saved_plots)} files\n"
        f"   Model checkpoints: {config.paths.checkpoint_dir}\n"
        f"   Results: {config.paths.results_dir}",
        border_style="green",
        title="Summary"
    ))
    
    # Save results to CSV
    for key in ["df_synthetic", "df_google", "df_alibaba"]:
        df = eval_results.get(key)
        if df is not None:
            csv_path = os.path.join(config.paths.results_dir, f"{key.replace('df_', '')}_results.csv")
            df.to_csv(csv_path, index=False)
            console.print(f"  Saved: {csv_path}")
    
    console.print("\n[bold]Thank you for using the RL Job Scheduling System![/bold]\n")


if __name__ == "__main__":
    main()
