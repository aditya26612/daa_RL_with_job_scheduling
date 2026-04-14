"""
greedy_solver.py — Classical Baseline Solvers for Job Sequencing
=================================================================
Implements four traditional scheduling heuristics as baselines:
1. Greedy-by-Profit: Schedule highest-profit feasible job first
2. Earliest Deadline First (EDF): Schedule nearest-deadline job first
3. Shortest Job First (SJF): Schedule quickest job first
4. Profit Density: Schedule by profit/processing_time ratio
"""

from typing import List, Tuple, Dict
from job import Job, JobSet
import copy


def _simulate_schedule(jobs: List[Job], sort_key, sort_reverse: bool = True) -> Tuple[List[Tuple[Job, int, int]], float]:
    """Generic scheduling simulation using a priority function.
    
    Iteratively selects the highest-priority feasible job at each time step,
    schedules it, and advances the clock. Continues until no feasible jobs remain.
    
    Args:
        jobs: List of jobs to schedule
        sort_key: Key function for sorting (determines priority)
        sort_reverse: If True, higher values = higher priority
        
    Returns:
        Tuple of (schedule, total_profit)
        schedule: List of (job, start_time, end_time)
    """
    remaining = list(jobs)
    schedule = []
    current_time = 0
    total_profit = 0.0
    
    while remaining:
        # Find feasible jobs
        feasible = [j for j in remaining if j.is_feasible(current_time)]
        
        if not feasible:
            # Try advancing time to the earliest arrival/deadline
            future_feasible = [j for j in remaining 
                             if current_time + j.processing_time <= j.deadline]
            if not future_feasible:
                break
            # Advance time to when something becomes feasible
            current_time += 1
            continue
        
        # Sort by priority and pick the best
        feasible.sort(key=sort_key, reverse=sort_reverse)
        best_job = feasible[0]
        
        # Schedule it
        start_time = current_time
        end_time = start_time + best_job.processing_time
        schedule.append((best_job, start_time, end_time))
        total_profit += best_job.profit
        current_time = end_time
        
        # Remove from remaining
        remaining = [j for j in remaining if j.job_id != best_job.job_id]
    
    return schedule, total_profit


def greedy_by_profit(jobs: List[Job]) -> Dict:
    """Schedule jobs greedily by highest profit first.
    
    Strategy: At each step, among feasible jobs, pick the one with the
    highest profit. This is the classic greedy approach.
    
    Args:
        jobs: List of jobs to schedule
        
    Returns:
        Dictionary with schedule, total_profit, and metrics
    """
    schedule, total_profit = _simulate_schedule(
        jobs, sort_key=lambda j: j.profit, sort_reverse=True
    )
    return _build_result("Greedy-by-Profit", jobs, schedule, total_profit)


def earliest_deadline_first(jobs: List[Job]) -> Dict:
    """Schedule jobs by earliest deadline first (EDF).
    
    Strategy: At each step, among feasible jobs, pick the one with the
    nearest deadline. Minimizes the chance of missing deadlines.
    
    Args:
        jobs: List of jobs to schedule
        
    Returns:
        Dictionary with schedule, total_profit, and metrics
    """
    schedule, total_profit = _simulate_schedule(
        jobs, sort_key=lambda j: j.deadline, sort_reverse=False
    )
    return _build_result("Earliest Deadline First", jobs, schedule, total_profit)


def shortest_job_first(jobs: List[Job]) -> Dict:
    """Schedule jobs by shortest processing time first (SJF).
    
    Strategy: At each step, among feasible jobs, pick the one that
    takes the least time. Maximizes throughput.
    
    Args:
        jobs: List of jobs to schedule
        
    Returns:
        Dictionary with schedule, total_profit, and metrics
    """
    schedule, total_profit = _simulate_schedule(
        jobs, sort_key=lambda j: j.processing_time, sort_reverse=False
    )
    return _build_result("Shortest Job First", jobs, schedule, total_profit)


def profit_density(jobs: List[Job]) -> Dict:
    """Schedule jobs by profit density (profit / processing_time).
    
    Strategy: At each step, among feasible jobs, pick the one with the
    highest profit-per-time-unit ratio. Balances profit and speed.
    
    Args:
        jobs: List of jobs to schedule

    Returns:
        Dictionary with schedule, total_profit, and metrics
    """
    schedule, total_profit = _simulate_schedule(
        jobs, sort_key=lambda j: j.density, sort_reverse=True
    )
    return _build_result("Profit Density", jobs, schedule, total_profit)


def _build_result(solver_name: str, jobs: List[Job],
                  schedule: List[Tuple[Job, int, int]], 
                  total_profit: float) -> Dict:
    """Build a standardized result dictionary.
    
    Args:
        solver_name: Name of the scheduling algorithm
        jobs: Original job list
        schedule: Computed schedule
        total_profit: Total profit achieved
        
    Returns:
        Standardized results dictionary
    """
    max_possible = sum(j.profit for j in jobs)
    num_scheduled = len(schedule)
    num_total = len(jobs)
    
    if schedule:
        makespan = max(end for _, _, end in schedule)
        busy_time = sum(end - start for _, start, end in schedule)
        utilization = busy_time / max(makespan, 1)
    else:
        makespan = 0
        utilization = 0.0
    
    return {
        "solver": solver_name,
        "schedule": schedule,
        "total_profit": total_profit,
        "max_possible_profit": max_possible,
        "profit_ratio": total_profit / max(max_possible, 1),
        "completion_rate": num_scheduled / max(num_total, 1),
        "num_scheduled": num_scheduled,
        "num_total": num_total,
        "makespan": makespan,
        "utilization": utilization,
    }


# Registry of all baseline solvers
BASELINE_SOLVERS = {
    "greedy_profit": greedy_by_profit,
    "edf": earliest_deadline_first,
    "sjf": shortest_job_first,
    "profit_density": profit_density,
}


def run_all_baselines(jobs: List[Job]) -> Dict[str, Dict]:
    """Run all baseline solvers on the same job set.
    
    Args:
        jobs: List of jobs to schedule
        
    Returns:
        Dictionary mapping solver name to results
    """
    results = {}
    for name, solver in BASELINE_SOLVERS.items():
        results[name] = solver(jobs)
    return results
