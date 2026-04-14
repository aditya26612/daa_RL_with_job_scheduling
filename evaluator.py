"""
evaluator.py -- Evaluation and Comparison Engine
=================================================
Runs the trained RL agent and all baseline solvers on
identical test instances, computes metrics, and produces
comparison tables and data for visualization.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import Config, EvalConfig
from environment import JobSchedulingEnv
from agent import DQNAgent
from greedy_solver import BASELINE_SOLVERS, run_all_baselines
from dataset_loader import generate_synthetic, get_dataset
from job import Job


console = Console(force_terminal=True)


class Evaluator:
    """Evaluates and compares RL agent against classical baselines.
    
    Runs all solvers on identical job instances across multiple
    problem sizes and difficulty levels, collecting detailed
    metrics for analysis.
    """
    
    def __init__(self, agent: DQNAgent, config: Optional[Config] = None):
        """Initialize the evaluator.
        
        Args:
            agent: Trained DQN agent
            config: Configuration
        """
        self.agent = agent
        self.config = config or Config()
        self.env = JobSchedulingEnv(self.config.env)
        self.results = {}
    
    def _run_rl_agent(self, jobs: List[Job]) -> Dict:
        """Run the RL agent on a job instance.
        
        Args:
            jobs: List of jobs to schedule
            
        Returns:
            Results dictionary matching baseline format
        """
        state, mask = self.env.reset(jobs)
        
        while not self.env.done:
            action = self.agent.select_action(state, mask, training=False)
            (state, mask), reward, done, info = self.env.step(action)
        
        metrics = self.env.get_metrics()
        schedule = self.env.get_schedule()
        
        return {
            "solver": "RL Agent (DQN)",
            "schedule": schedule,
            "total_profit": metrics["total_profit"],
            "max_possible_profit": metrics["max_possible_profit"],
            "profit_ratio": metrics["profit_ratio"],
            "completion_rate": metrics["completion_rate"],
            "num_scheduled": metrics["num_scheduled"],
            "num_total": metrics["num_total"],
            "makespan": metrics["makespan"],
            "utilization": metrics["utilization"],
        }
    
    def evaluate_single(self, jobs: List[Job]) -> Dict[str, Dict]:
        """Evaluate all solvers on a single job instance.
        
        Args:
            jobs: List of jobs
            
        Returns:
            Dictionary mapping solver name to results
        """
        # Run baselines
        results = run_all_baselines(jobs)
        
        # Run RL agent
        results["rl_agent"] = self._run_rl_agent(jobs)
        
        return results
    
    def evaluate_batch(self, dataset_name: str = "synthetic",
                       test_sizes: Optional[List[int]] = None,
                       num_instances: int = 50) -> pd.DataFrame:
        """Evaluate all solvers across multiple problem sizes.
        
        Args:
            dataset_name: Dataset to use
            test_sizes: List of problem sizes to test
            num_instances: Number of instances per size
            
        Returns:
            DataFrame with all results
        """
        test_sizes = test_sizes or self.config.eval.test_sizes
        
        all_records = []
        
        for n_jobs in test_sizes:
            console.print(f"\n[bold cyan]Evaluating with {n_jobs} jobs...[/bold cyan]")
            
            for instance_id in range(num_instances):
                seed = 1000 + instance_id
                jobs = get_dataset(dataset_name, n_jobs, seed)
                
                results = self.evaluate_single(jobs)
                
                for solver_name, result in results.items():
                    all_records.append({
                        "num_jobs": n_jobs,
                        "instance_id": instance_id,
                        "solver": result["solver"],
                        "solver_key": solver_name,
                        "total_profit": result["total_profit"],
                        "profit_ratio": result["profit_ratio"],
                        "completion_rate": result["completion_rate"],
                        "utilization": result["utilization"],
                        "makespan": result["makespan"],
                        "num_scheduled": result["num_scheduled"],
                    })
        
        df = pd.DataFrame(all_records)
        self.results[dataset_name] = df
        return df
    
    def print_comparison_table(self, df: Optional[pd.DataFrame] = None,
                               dataset_name: str = "synthetic"):
        """Print a rich comparison table to the console.
        
        Args:
            df: Results DataFrame (uses stored results if None)
            dataset_name: Dataset name for title
        """
        if df is None:
            df = self.results.get(dataset_name)
            if df is None:
                console.print("[red]No results available. Run evaluate_batch first.[/red]")
                return
        
        # Aggregate by solver and problem size
        agg = df.groupby(["num_jobs", "solver"]).agg({
            "total_profit": ["mean", "std"],
            "profit_ratio": "mean",
            "completion_rate": "mean",
            "utilization": "mean",
        }).reset_index()
        
        # Flatten column names
        agg.columns = [
            "num_jobs", "solver",
            "profit_mean", "profit_std",
            "profit_ratio", "completion_rate", "utilization"
        ]
        
        # Print table for each problem size
        for n_jobs in sorted(df["num_jobs"].unique()):
            table = Table(
                title=f"Performance Comparison -- {n_jobs} Jobs ({dataset_name})",
                show_header=True,
                header_style="bold magenta",
                border_style="bright_blue",
            )
            
            table.add_column("Solver", style="cyan", width=22)
            table.add_column("Avg Profit", justify="right", style="green")
            table.add_column("± Std", justify="right", style="dim")
            table.add_column("Profit Ratio", justify="right", style="yellow")
            table.add_column("Completion %", justify="right", style="blue")
            table.add_column("Utilization", justify="right", style="magenta")
            
            subset = agg[agg["num_jobs"] == n_jobs].sort_values("profit_mean", ascending=False)
            
            best_profit = subset["profit_mean"].max()
            
            for _, row in subset.iterrows():
                is_best = abs(row["profit_mean"] - best_profit) < 0.01
                style = "bold green" if is_best else ""
                
                table.add_row(
                    f"{'>> ' if is_best else '   '}{row['solver']}",
                    f"{row['profit_mean']:.1f}",
                    f"{row['profit_std']:.1f}",
                    f"{row['profit_ratio']:.3f}",
                    f"{row['completion_rate']:.1%}",
                    f"{row['utilization']:.3f}",
                    style=style,
                )
            
            console.print(table)
            console.print()
    
    def get_improvement_summary(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """Calculate RL agent's improvement over baselines.
        
        Args:
            df: Results DataFrame
            
        Returns:
            Dictionary with improvement percentages
        """
        if df is None:
            df = list(self.results.values())[0] if self.results else None
        if df is None:
            return {}
        
        # Average profit per solver
        avg = df.groupby("solver_key")["total_profit"].mean()
        
        rl_profit = avg.get("rl_agent", 0)
        
        improvements = {}
        for solver_key, profit in avg.items():
            if solver_key != "rl_agent" and profit > 0:
                pct = ((rl_profit - profit) / profit) * 100
                improvements[solver_key] = {
                    "baseline_profit": profit,
                    "rl_profit": rl_profit,
                    "improvement_pct": pct,
                }
        
        return improvements
    
    def print_summary(self):
        """Print a high-level summary of all evaluations."""
        console.print(Panel(
            "[bold]Evaluation Summary[/bold]",
            border_style="green"
        ))
        
        for dataset_name, df in self.results.items():
            improvements = self.get_improvement_summary(df)
            
            console.print(f"\n[bold cyan]Dataset: {dataset_name}[/bold cyan]")
            
            for solver, data in improvements.items():
                sign = "+" if data["improvement_pct"] >= 0 else ""
                color = "green" if data["improvement_pct"] >= 0 else "red"
                console.print(
                    f"  vs {solver}: "
                    f"[{color}]{sign}{data['improvement_pct']:.1f}%[/{color}] "
                    f"(RL: {data['rl_profit']:.1f} vs Baseline: {data['baseline_profit']:.1f})"
                )
