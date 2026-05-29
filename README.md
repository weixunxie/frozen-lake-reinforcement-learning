# Frozen Lake Reinforcement Learning: Dynamic Programming and Tabular Control

A clean, reproducible implementation of classical reinforcement learning methods for solving
the [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment
from [Gymnasium](https://gymnasium.farama.org/). All algorithms are implemented from scratch
using NumPy.

---

## Project Overview

This project studies how classical RL algorithms solve discrete Markov Decision Processes
(MDPs). Frozen Lake is a small grid-world environment where an agent must navigate across
a frozen surface to reach a goal while avoiding holes, with optional stochastic (slippery)
transitions. Its small state space makes it ideal for directly inspecting learned value
functions and policies, while still capturing key RL challenges: sparse rewards, stochastic
dynamics, and exploration under uncertainty.

The project implements and compares:

- **Dynamic programming** (model-based): Policy Evaluation, Policy Improvement, Policy
  Iteration, and Value Iteration
- **Model-free RL**: Q-Learning with ε-greedy exploration

Experiments cover both the deterministic and stochastic (slippery) variants of the standard
4×4 map, plus a custom narrow-corridor map used in the original research presentation.

---

## Research Background

This repository is an extended and cleaned-up version of an undergraduate research project
completed during junior year at **Kean University**, Department of Mathematical Sciences,
under the supervision of **Dr. Israel R. Curbelo**.

The original project was presented at **Student Research Day** by Qiutong Liu, Weixun Xie,
Sihan Fu, Junyang Li, and Zhirui Chen. The research poster is preserved exactly as an
archival artifact — see [docs/frozen_lake_challenge.pdf](docs/frozen_lake_challenge.pdf).

This repository reorganizes the original code into a cleaner structure, adds experiment
organization, and expands the documentation. It does not overstate the original work: the
poster reflects what was done at the time, and some details (e.g., the exact reward for
holes) differ slightly between the poster version and the current implementation.

---

## Motivation

Frozen Lake is a useful teaching environment because:

- The state and action spaces are small enough to inspect value functions and policies directly
- The stochastic variant captures real challenges: an agent cannot fully control outcomes
- Sparse rewards (+1 at goal, 0 elsewhere by default) make exploration necessary
- It cleanly separates model-based from model-free approaches

---

## Environment

| Property | Value |
|---|---|
| Library | `gymnasium` |
| Environment ID | `FrozenLake-v1` (wrapped as `CustomFrozenLakeEnv`) |
| State space | Discrete, 16 states (4×4 grid) |
| Action space | Discrete: 0=Left, 1=Down, 2=Right, 3=Up |
| Reward (goal) | +1 |
| Reward (hole) | −1 (custom penalty; standard env gives 0) |
| Reward (frozen) | 0 |
| Variants | Non-slippery (deterministic) and Slippery (stochastic) |
| Custom map | 3×5 narrow corridor (`HHHHH / HSFFG / HHHHH`) |

The `CustomFrozenLakeEnv` wrapper adds a −1 penalty for falling into a hole, providing a
denser reward signal for model-free methods. The dynamic programming methods use the full
transition model `env.P` regardless.

---

## Methods

### Policy Evaluation

Iteratively applies the Bellman expectation equation for a fixed policy until the value
function converges:

$$V(s) \leftarrow \sum_{s'} P(s' \mid s, \pi(s))\bigl[R(s,\pi(s),s') + \gamma V(s')\bigr]$$

### Policy Improvement

Computes the greedy policy with respect to a given value function:

$$\pi'(s) = \arg\max_a \sum_{s'} P(s'\mid s,a)\bigl[R(s,a,s') + \gamma V(s')\bigr]$$

### Policy Iteration

Alternates between Policy Evaluation and Policy Improvement until the policy stops changing.
Guaranteed to converge to the optimal policy in a finite MDP.

### Value Iteration

Applies Bellman *optimality* backups directly, skipping the separate evaluation step:

$$V_{k+1}(s) = \max_a \sum_{s'} P(s'\mid s,a)\bigl[R(s,a,s') + \gamma V_k(s')\bigr]$$

The greedy policy is extracted after convergence.

### Q-Learning

Off-policy temporal-difference learning that estimates action values from experience,
without needing the model:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\bigr]$$

Exploration uses an ε-greedy strategy with exponential decay.

---

## Repository Structure

```
frozen-lake-reinforcement-learning/
│
├── notebooks/
│   ├── frozen_lake_dp_and_rl.ipynb     # Main notebook: all algorithms + experiments
│   └── minigrid_experiments.ipynb      # Exploratory extension (requires minigrid package)
│
├── docs/
│   └── frozen_lake_challenge.pdf       # Student Research Day poster (original artifact)
│
├── results/
│   ├── figures/                        # Generated plots (created by running the notebook)
│   └── tables/                         # Generated CSV tables
│
├── src/                                # Reserved for future reusable Python modules
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

**Note on `src/`:** The current implementation lives entirely in the notebooks. The `src/`
directory is reserved for future refactoring into reusable modules (e.g., `dynamic_programming.py`,
`model_free_rl.py`, `visualization.py`).

---

## Experiments

### 4×4 Standard Map

```
S F F F
F H F H
F F F H
H F F G
```

Both the **deterministic** (non-slippery) and **stochastic** (slippery, 1/3 lateral slip
probability) variants are evaluated. All three algorithms are run and compared.

### Custom Narrow Corridor

```
H H H H H
H S F F G
H H H H H
```

The only viable path is moving right along the single open row. Used in the original
poster to illustrate policy learning under tight constraints.

### Evaluation Protocol

- Policies from DP methods are evaluated over 1,000 episodes each
- Q-learning policies use ε-greedy training followed by greedy evaluation
- Metrics: success rate (fraction of episodes reaching the goal) and mean episode reward
- Results are saved to `results/tables/algorithm_comparison.csv`

---

## Results

Results are generated by running [notebooks/frozen_lake_dp_and_rl.ipynb](notebooks/frozen_lake_dp_and_rl.ipynb)
and may vary slightly depending on random seed and environment stochasticity.

Key observations from the implementation:

- Policy Iteration and Value Iteration converge to the same optimal policy when given
  the full model; they differ in computational structure, not in solution quality.
- Q-Learning matches the DP policies on the non-slippery variant given sufficient training
  episodes. On the slippery variant, performance depends on the exploration schedule.
- The slippery environment imposes an inherent upper bound on success rate: even an optimal
  policy cannot guarantee reaching the goal due to uncontrollable lateral transitions.

Figures saved by the notebook:

| File | Contents |
|---|---|
| `results/figures/value_function_heatmap.png` | Value function heatmaps (VI, both variants) |
| `results/figures/policy_comparison_grid.png` | Policy grids for all three algorithms |
| `results/figures/q_learning_reward_curve.png` | Q-learning episode reward curves |
| `results/figures/algorithm_comparison_bar.png` | Success-rate bar chart |
| `results/figures/custom_map_results.png` | Custom corridor map results |
| `results/tables/algorithm_comparison.csv` | Numeric evaluation results |

---

## Research Artifacts

The file [docs/frozen_lake_challenge.pdf](docs/frozen_lake_challenge.pdf) is the original
**Student Research Day poster** presented at Kean University, titled *"Navigating Uncertainty:
A Reinforcement Learning Approach to Solving the Frozen Lake Challenge."* It is preserved
exactly as an archival record of the original undergraduate research presentation.

The poster describes the two algorithms implemented at the time (Value Iteration and
Q-Learning) and shows policy results on the 4×4 and custom maps. Some details differ
from this cleaned repository (e.g., the poster uses a −0.1 hole penalty; this repo uses −1).

---

## How to Reproduce

```bash
# 1. Clone the repository
git clone <repo-url>
cd frozen-lake-reinforcement-learning

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter and open the main notebook
jupyter notebook notebooks/frozen_lake_dp_and_rl.ipynb

# 5. Run all cells (Kernel → Restart & Run All)
#    Results are saved to results/figures/ and results/tables/
```

For the MiniGrid extension:

```bash
pip install minigrid
jupyter notebook notebooks/minigrid_experiments.ipynb
```

---

## Requirements

```
numpy>=1.23
pandas>=1.5
matplotlib>=3.6
gymnasium>=0.26
jupyter>=1.0

# Optional (MiniGrid extension only)
minigrid>=2.3
```

No deep learning frameworks (PyTorch, TensorFlow) are required for the main notebook.

---

## Limitations

- Frozen Lake is a small tabular environment. Results do not directly generalize to
  high-dimensional problems requiring function approximation.
- Sparse rewards make Q-learning sensitive to the exploration schedule. No systematic
  hyperparameter sweep is included.
- SARSA (on-policy TD) is not implemented; a comparison with Q-learning would be a natural
  next step.
- The MiniGrid extension is exploratory: the tabular approach is a rough fit for the
  image-based observation space and is not a production-quality implementation.
- The Student Research Day poster reflects the original project scope and may not match
  every detail of this cleaned repository.

---

## Future Improvements

- Systematic hyperparameter search for Q-learning (α, γ, ε-decay)
- Multi-seed evaluation with confidence intervals
- SARSA implementation for on-policy vs. off-policy comparison
- Experiments on larger randomly generated maps (`generate_random_map(size=8)`)
- DQN extension for the MiniGrid environment
- Comparison table with convergence iteration counts across map sizes

---

## Portfolio Notes

This project demonstrates foundational reinforcement learning concepts:

- **Markov Decision Processes** — formal problem setup, transition model, reward structure
- **Dynamic programming** — Policy Evaluation, Policy Improvement, Policy Iteration,
  Value Iteration
- **Tabular control** — Q-learning with ε-greedy exploration
- **Policy and value visualization** — heatmaps, arrow-grid policy displays
- **Experiment organization** — reproducible evaluation protocol, saved figures and tables
- **Research continuity** — extending and cleaning an earlier undergraduate research project
  while preserving the original presentation artifact

---

## Authors

Weixun Xie et al.  
Original project: Qiutong Liu, Weixun Xie, Sihan Fu, Junyang Li, Zhirui Chen  
Advisor: Dr. Israel R. Curbelo, Department of Mathematical Sciences, Kean University
