# Adaptive Difficulty Scaling with Reinforcement Learning

**Team 13** — Nurgissa (20210785), Youngjin Cho (20190627)

A two-stage game AI system that dynamically adjusts opponent difficulty in real time based on how well the player is actually playing — not just whether they win or lose. The system estimates player skill from the quality of their moves and uses a learned Q-Learning controller to decide when to increase, hold, or decrease difficulty.

---

## How It Works

```
Player makes a move
        ↓
QValueDifficultyEstimator ranks that move against all legal moves
using Q-values from the trained opponent pool → estimates player skill
        ↓
QValueDDAMetrics tracks a sliding window of:
  win rate · move quality · game duration · difficulty variance
        ↓
QLearningDDA (RL meta-agent) selects action: [-1] down / [0] stay / [+1] up
        ↓
Next game uses the updated difficulty level
        ↓
DDA Q-table updates via Bellman equation using reward:
  -(mismatch_error + duration_penalty + variance_penalty + switch_penalty)
```

There are N pre-trained opponent agents at discrete skill levels. The DDA controller learns — through its own Q-table — when switching difficulty actually helps maintain engagement, rather than following hard-coded rules.

---

## Project Structure

```
adaptive_difficulty_rl/
├── envs/                      # Game environments
│   ├── base_game.py           # Abstract game interface
│   ├── tictactoe.py           # Tic-Tac-Toe (3×3, k=3)
│   └── mnk_game.py            # MNK generalized game — stub (in progress)
│
├── rl_training/               # All agent and training code
│   ├── q_learning_agent.py    # Base Q-Learning game agent
│   ├── train_base_agent.py    # Trains the opponent pool (5 levels)
│   ├── meta_agent.py          # Rule-based difficulty selector (baseline)
│   ├── difficulty_scaling.py  # RL-based DDA controller (main system)
│   ├── difficulty_scaling_explanation.md
│   └── utils.py               # Shared helpers
│
├── experiments/               # Evaluation and analysis scripts
│   ├── evaluate_agent.py
│   ├── test_adaptation.py
│   ├── compare_rewards.py
│   ├── sensitivity_analysis.py
│   ├── failure_analysis.py
│   └── results/               # Output folder for plots and reports
│
├── agents/                    # Saved trained opponent models (.pkl)
├── notebooks/                 # Optional Jupyter notebooks
└── requirements.txt
```

---

## Directory Breakdown

### `envs/` — Game Environments

| File | What it does |
|------|-------------|
| `base_game.py` | Abstract interface every game must implement: `reset`, `get_state`, `get_valid_moves`, `make_move`, `get_winner` |
| `tictactoe.py` | Tic-Tac-Toe (3,3,3) — full board logic, win detection across rows/columns/diagonals |
| `mnk_game.py` | Generalized m×n board, k-in-a-row win condition. Covers both Tic-Tac-Toe (3,3,3) and Gomoku (15,15,5). **Currently a stub — in progress.** Required by `difficulty_scaling.py`. |

---

### `rl_training/` — Agent and Training Code

#### Base game agents

| File | What it does |
|------|-------------|
| `q_learning_agent.py` | Q-Learning game agent. Maintains a Q-table mapping board states to move scores. Learns via the Bellman equation, selects moves via epsilon-greedy. Supports save/load with pickle. |
| `train_base_agent.py` | Trains the agent for 50,000 games against a random opponent, saving a snapshot every 10,000 games. Produces 5 `.pkl` files in `agents/` at increasing skill levels. |
| `utils.py` | `play_game()` — runs a full game between two agents. `train_one_game()` — single-game training helper. |

#### Difficulty controllers

| File | What it does |
|------|-------------|
| `meta_agent.py` | **Baseline (rule-based).** Watches win rate over a sliding window of 5 games. Win rate > 60% → increase difficulty; < 40% → decrease. No learning — used as a comparison baseline. |
| `difficulty_scaling.py` | **Main system (RL-based).** Three components working together: |

`difficulty_scaling.py` components in detail:

**`QValueDifficultyEstimator`** — Measures move quality by ranking the player's actual move against every legal move using Q-values from the opponent pool. A move that scores in the 75th percentile across 4 difficulty levels → estimated player skill of 3.0. This is how the system tracks skill from behavior rather than just outcomes.

**`QValueDDAMetrics`** — Sliding window tracker (default: last 20 games) for win rate, move quality, game duration, and difficulty estimate variance. Computes the reward signal after each game:
```
reward = -(
    q_match_weight     × |opponent_difficulty − estimated_player_skill|
  + duration_weight    × |game_duration − target_duration|
  + variance_weight    × Var(recent_estimated_difficulty)
  + switch_weight      × I[difficulty changed this episode]
)
```

**`QLearningDDA`** — The RL meta-agent. Has its own Q-table keyed on a 4-tuple state `(skill_bucket, duration_bucket, variance_bucket, current_level)`. Actions: `-1` (go down), `0` (stay), `+1` (go up). Uses epsilon-greedy with decay (`ε` starts at 0.25, decays to 0.04 minimum). Updates its Q-table via Bellman after every game.

---

### `agents/` — Saved Opponent Models

Stores the `.pkl` files generated by `train_base_agent.py`. Empty until training is run.

| File | Training episodes | Expected win rate vs random |
|------|------------------|-----------------------------|
| `tictactoe_level_1.pkl` | 10,000 | ~25% |
| `tictactoe_level_2.pkl` | 20,000 | ~50% |
| `tictactoe_level_3.pkl` | 30,000 | ~70% |
| `tictactoe_level_4.pkl` | 40,000 | ~85% |
| `tictactoe_level_5.pkl` | 50,000 | ~95% |

---

### `experiments/` — Evaluation Scripts

Run these after training to measure system behavior.

| File | What it does |
|------|-------------|
| `evaluate_agent.py` | Tests each saved level against a random opponent (50 games each). Confirms the difficulty gradient is correct. |
| `test_adaptation.py` | End-to-end simulation of the rule-based adaptive system over 50 games. Reports win rate, average difficulty, and levels used. |
| `compare_rewards.py` | Trains fresh agents under 4 reward schemes (`sparse`, `draw_penalty`, `draw_reward`, `asymmetric`) and compares win/draw rates. |
| `sensitivity_analysis.py` | Sweeps learning rate `α`, discount factor `γ`, and epsilon `ε` to show how each hyperparameter affects win rate. |
| `failure_analysis.py` | Detects missed wins (agent had a winning move, ignored it) and missed blocks (opponent about to win, agent didn't defend) per level. |

---

## Testing Locally

### If you are cloning for the first time

```bash
git clone git@github.com:nurgissas/GameAI.git
cd GameAI/adaptive_difficulty_rl
```

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the opponent pool (~5–10 minutes)

```bash
python rl_training/train_base_agent.py
```

Expected output every 10,000 episodes:
```
Episode 10000/50000 | States: 5427 | Win rate: 75.2%
Episode 20000/50000 | States: 5489 | Win rate: 82.1%
...
Training complete!
```
This saves `agents/tictactoe_level_1.pkl` through `tictactoe_level_5.pkl`.

### 4. Verify the difficulty gradient

```bash
python experiments/evaluate_agent.py
```

Expected output:
```
Level 1: ~25% win rate
Level 2: ~50% win rate
Level 3: ~70% win rate
Level 4: ~85% win rate
Level 5: ~95% win rate
```

### 5. Test the rule-based adaptive system

```bash
python experiments/test_adaptation.py
```

Expected output: player win rate near 50%, difficulty oscillating across middle levels.

### 6. Optional analysis (longer runs)

```bash
python experiments/compare_rewards.py       # ~20 min — reward scheme comparison
python experiments/sensitivity_analysis.py  # ~30 min — hyperparameter sweep
python experiments/failure_analysis.py      # ~2 min  — needs trained agents
```

### What cannot be tested yet

`rl_training/difficulty_scaling.py` (the RL DDA controller) depends on `envs/mnk_game.py`, which is currently a stub. Once `MNKGame` is fully implemented, the full RL pipeline can be tested with:

```python
from rl_training.difficulty_scaling import (
    QLearningDDA, QValueDDAMetrics,
    QValueDifficultyEstimator, RewardConfig,
    run_online_difficulty_learning_episode,
)
```

---

## Key Concepts

**Q-Learning (base agent)** — Learns to play the game by maintaining a table of scores for every (board state, move) pair. Updated after each game using the Bellman equation:
```
Q(s, a) = Q(s, a) + α × [R + γ × max Q(s') − Q(s, a)]
```

**Q-Learning (DDA controller)** — A second, separate Q-table that learns when to change difficulty. Its state is a 4-tuple of bucketed metrics; its actions are `{-1, 0, +1}`. It learns that unnecessary switches are costly (switch penalty) and that matching opponent skill to player skill is the primary goal (mismatch penalty).

**Move quality estimation** — Instead of only asking "did the player win?", the system asks "how good was the player's move?" by comparing it against all legal options using the opponent pool's Q-values. This gives a continuous skill signal within each game, not just at the end.

**Reward design** — The DDA reward deliberately avoids directly rewarding 50% win rate. Instead it penalizes the gap between estimated player skill and current opponent difficulty. The theory is that matched difficulty naturally produces balanced outcomes without overfitting to win rate as a metric.

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Tic-Tac-Toe environment | Done |
| Q-Learning base agent | Done |
| Opponent pool training (5 levels) | Done |
| Rule-based meta-agent (baseline) | Done |
| RL DDA controller (`difficulty_scaling.py`) | Done |
| `envs/mnk_game.py` (MNK game engine) | In progress — stub only |
| Gomoku environment and training | Not started |

---

## Requirements

```
numpy
matplotlib
pandas
```

Python 3.8+. All dependencies are pip-installable.
