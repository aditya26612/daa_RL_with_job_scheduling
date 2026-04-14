# Reinforcement-Learned Strategy for Job Sequencing with Deadlines

## 1. Problem Statement

The **Job Sequencing with Deadlines** problem is a classic combinatorial optimization problem:
- Given **n** jobs, each with a **profit**, a **deadline**, and a **processing time**, schedule jobs on a single machine to **maximize total profit**.
- Each job must be completed before its deadline, and only one job can run at a time.

Traditional approaches (greedy by profit/deadline ratio) provide approximate solutions but lack **adaptability** to varying workload distributions. This project enhances scheduling using a **Reinforcement Learning (RL)** agent that learns a priority adaptation strategy, outperforming static heuristics on diverse and dynamic workloads.

---

## 2. Objectives

| # | Objective | Deliverable |
|---|-----------|-------------|
| 1 | Formalize the Job Sequencing problem as a Markov Decision Process (MDP) | `environment.py` |
| 2 | Implement classical baselines (Greedy, EDF, SJF) | `greedy_solver.py` |
| 3 | Build a Deep Q-Network (DQN) agent with experience replay | `agent.py`, `replay_buffer.py` |
| 4 | Support real-world datasets (Google Cluster, Alibaba Trace) | `dataset_loader.py` |
| 5 | Train and evaluate the RL agent against baselines | `trainer.py`, `evaluator.py` |
| 6 | Produce rich visualizations (reward curves, Gantt charts, comparisons) | `visualizer.py` |
| 7 | Provide a single-command demo for evaluation | `main.py` |

---

## 3. Theoretical Background

### 3.1 Job Sequencing — NP-Hard Variant
When jobs have **unit processing times** and deadlines, a greedy algorithm is optimal. However, with **variable processing times** and **dependent constraints**, the problem becomes NP-hard, motivating heuristic and learning-based approaches.

### 3.2 MDP Formulation

| MDP Element | Definition |
|-------------|-----------|
| **State** | Current time, remaining jobs (features: profit, deadline, processing time, slack), machine status |
| **Action** | Select the next job to schedule from the set of feasible jobs |
| **Reward** | `+profit` if job completes before deadline; `−penalty` for missed deadlines; small negative step cost |
| **Transition** | Time advances by processing time of selected job; selected job is removed from pending set |
| **Terminal** | No feasible jobs remain or all jobs are scheduled/skipped |

### 3.3 DQN Agent Architecture
- **State Encoder**: Fully-connected network encoding variable-length job sets via attention-pooling
- **Action Selection**: ε-greedy with decaying exploration
- **Experience Replay**: Prioritized replay buffer (capacity 50,000)
- **Target Network**: Soft-updated every 100 steps (τ = 0.005)
- **Dueling Architecture**: Separate value and advantage streams

### 3.4 Classical Baselines
1. **Greedy-by-Profit**: Schedule highest-profit feasible job first
2. **Earliest Deadline First (EDF)**: Schedule job with nearest deadline first
3. **Shortest Job First (SJF)**: Schedule job with smallest processing time first
4. **Profit-Density**: Schedule by profit/processing_time ratio

---

## 4. Dataset Support

### 4.1 Synthetic Dataset
- Configurable number of jobs (10–500)
- Profits: Uniform, Normal, or Pareto distributions
- Deadlines: Tight, Moderate, or Loose settings
- Processing times: Unit or Variable

### 4.2 Google Cluster Dataset
- Source: https://github.com/google/cluster-data
- Extract task events → map (priority, scheduling_class, CPU_request) to (profit, deadline, processing_time)
- Preprocessing pipeline in `dataset_loader.py`

### 4.3 Alibaba Cluster Trace
- Source: https://github.com/alibaba/clusterdata
- Extract batch task instances → similar mapping
- Supports trace-driven evaluation

---

## 5. Project Architecture

```
daa/
├── plan.md                 # This document
├── config.py               # All hyperparameters and settings
├── job.py                  # Job data class and utilities
├── environment.py          # RL Environment (Gym-like interface)
├── replay_buffer.py        # Prioritized experience replay
├── agent.py                # DQN Agent with dueling architecture
├── greedy_solver.py        # Classical baseline solvers
├── dataset_loader.py       # Synthetic + real dataset loaders
├── trainer.py              # Training loop with logging
├── evaluator.py            # Evaluation and comparison engine
├── visualizer.py           # Plots: reward curves, Gantt, comparisons
├── main.py                 # Entry point — runs everything
├── requirements.txt        # Dependencies
└── output/                 # Generated plots and results
```

---

## 6. Module Descriptions

### `config.py`
Central configuration: hyperparameters (learning rate, gamma, epsilon schedule), dataset parameters, training settings. Uses dataclasses for clean access.

### `job.py`
`Job` dataclass with fields: `id`, `profit`, `deadline`, `processing_time`. Utility functions for sorting, filtering feasible jobs, computing slack.

### `environment.py`
Gym-compatible environment. Manages state representation (fixed-size feature matrix with padding/masking), action space (select job index), reward computation, and episode termination.

### `replay_buffer.py`
Prioritized experience replay with sum-tree for O(log n) sampling. Stores (s, a, r, s', done) transitions with TD-error-based priorities.

### `agent.py`
Dueling DQN with:
- Feature encoder (3-layer MLP)
- Value stream and Advantage stream
- ε-greedy exploration with cosine annealing
- Soft target network updates

### `greedy_solver.py`
Four baseline solvers: Greedy-by-Profit, EDF, SJF, Profit-Density. Each returns a schedule and total profit.

### `dataset_loader.py`
- `generate_synthetic()`: Configurable random instances
- `load_google_cluster()`: Parse Google Cluster trace CSVs
- `load_alibaba_trace()`: Parse Alibaba batch task CSVs

### `trainer.py`
Training loop: episodes, batch updates, target sync, logging. Supports checkpointing and early stopping.

### `evaluator.py`
Runs all solvers (RL + baselines) on test instances. Computes metrics: total profit, utilization, makespan, job completion rate.

### `visualizer.py`
- Training reward/loss curves
- Gantt chart of schedules
- Bar chart comparisons across solvers
- Heatmap of performance vs. problem size

### `main.py`
CLI entry point. Modes: `--train`, `--evaluate`, `--demo` (runs everything end-to-end and opens plots).

---

## 7. Evaluation Plan

| Metric | Description |
|--------|-------------|
| Total Profit | Sum of profits from completed jobs |
| Completion Rate | % of jobs that met their deadline |
| Machine Utilization | % of time the machine was busy |
| Makespan | Total time from start to last job completion |
| Training Convergence | Episodes to reach 95% of optimal on small instances |

### Comparison Matrix
- 4 baselines × RL agent
- 3 dataset types (Synthetic Easy/Medium/Hard)
- 5 problem sizes (20, 50, 100, 200, 500 jobs)

---

## 8. Expected Results

Based on literature and preliminary experiments:
1. RL agent should match greedy on easy instances (unit processing times)
2. RL agent should **outperform all baselines by 10–25%** on hard instances (variable processing times, tight deadlines)
3. RL agent should show **generalization** — trained on 50-job instances, tested on 100+ jobs

---

## 9. Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Research & Design | Day 1 | Literature review, MDP formulation, architecture design |
| Core Implementation | Day 2–3 | Environment, Agent, Baselines, Dataset loader |
| Training & Tuning | Day 4–5 | Hyperparameter search, training runs |
| Evaluation & Visualization | Day 6 | Benchmarking, plots, analysis |
| Documentation & Demo | Day 7 | Report, demo script, presentation prep |

---

## 10. References

1. Cormen, T. H., et al. *Introduction to Algorithms*, 4th Edition — Job Sequencing
2. Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd Edition
3. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*
4. Wang, Z., et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." *ICML*
5. Google Cluster Workload Traces: https://github.com/google/cluster-data
6. Alibaba Cluster Trace Program: https://github.com/alibaba/clusterdata
7. Schaul, T., et al. (2016). "Prioritized Experience Replay." *ICLR*
