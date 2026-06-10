# Adaptive Difficulty Scaling with Reinforcement Learning

Authors: Nurgissa Sailaubek, Youngjin Cho

A two-stage game AI system that dynamically adjusts opponent difficulty in real time based on how well the player is actually playing — not just whether they win or lose. The system estimates player skill from the quality of their moves and uses a learned Q-Learning controller to decide when to increase, hold, or decrease difficulty.

Supports any m×n board and k-in-a-row win condition. Tic-Tac-Toe is `MNKGame(3,3,3)`; Gomoku is `MNKGame(15,15,5)`.

---

## How It Works

```
Player makes a move
        ↓
QValueDifficultyEstimator ranks that move against all legal moves
using Q-values from the opponent pool → estimates player skill
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

The opponent pool contains N agents at discrete skill levels, built from a single trained model with varying epsilon values (see below). The DDA controller learns when switching difficulty actually helps maintain engagement, rather than following hard-coded rules.

---

## Agent Design: Linear Function Approximation

The opponent pool agents use **linear function approximation** instead of a Q-table.

A Q-table stores one value per (board state, move) pair. For a 3×3 board this is manageable (~5,000 states), but for a 15×15 Gomoku board the state space is astronomically large — the table would never converge.

Instead, each agent learns a single **weight vector** `w` of size 18:

```
Q(board, move) = w · φ(board, move)
```

`φ` is a fixed 18-dimensional feature vector extracted from any board position (see below). The weights are updated via semi-gradient TD at every agent turn:

```
δ = r + γ · max_a' Q(s', a') − Q(s, a)
w += α · δ · φ(s, a)
```

This approach generalizes to any board size because the feature vector has **fixed dimension regardless of n or k**.

### Feature Vector (18 dimensions)

For the move being evaluated, across each of 4 directions (row, column, diagonal, anti-diagonal):

| Features | Description |
|----------|-------------|
| 0–3   | My run length after placing ÷ k (clamped to 1) |
| 4–7   | My open ends ÷ 2 |
| 8–11  | Opponent pieces in forward direction ÷ k |
| 12–15 | Opponent pieces in backward direction ÷ k |
| 16    | Normalized distance from board center |
| 17    | Board fill ratio |

---

## Difficulty Pool: Epsilon-Based Levels

A naive approach would train 5 separate agents and use training snapshots as difficulty levels. This does not work reliably with linear FA because the weights converge early — empirically, win rate stops improving after ~2,000 episodes and remains flat through 50,000. Snapshot levels 1–5 end up nearly identical in strength.

Instead, **one fully-converged agent is trained**, and 5 difficulty levels are created at inference time by varying epsilon:

| Level | Epsilon | Behavior |
|-------|---------|----------|
| 1 | 1.00 | Plays randomly — easiest |
| 2 | 0.70 | 70% random, 30% greedy |
| 3 | 0.40 | 40% random, 60% greedy |
| 4 | 0.15 | Nearly greedy |
| 5 | 0.00 | Always picks the best move — hardest |

This guarantees a **monotonic difficulty gradient by construction** — the gradient does not depend on when or how fast the weights converged during training.

The pool is created by `build_opponent_pool(path)` in `mnk_q_agent.py`, which loads one `.pkl` file and returns 5 agent copies with the epsilon values above.

Actual win rates measured against a random opponent (200 games per level, 3×3 board):

```
Level 1  ε=1.00   45.5%
Level 2  ε=0.70   60.5%
Level 3  ε=0.40   62.5%
Level 4  ε=0.15   76.0%
Level 5  ε=0.00   84.0%
```

---

## Project Structure

```
adaptive_difficulty_rl/
├── envs/
│   └── mnk_game.py            # Generalized m×n board, k-in-a-row win condition
│
├── rl_training/
│   ├── feature_extractor.py   # 18-dim feature vector for any board/move
│   ├── mnk_q_agent.py         # Linear FA agent, train_mnk_agent, build_opponent_pool
│   ├── train_base_agent.py    # CLI script: trains and saves one model
│   ├── meta_agent.py          # Rule-based difficulty selector (baseline)
│   └── difficulty_scaling.py  # RL-based DDA controller (main system)
│
├── experiments/
│   ├── evaluate_agent.py      # Win rate of each epsilon level vs random
│   ├── test_adaptation.py     # End-to-end rule-based DDA simulation
│   ├── compare_rewards.py     # Compares 4 reward formulations
│   └── sensitivity_analysis.py # Hyperparameter sweep (α, γ, ε)
│
├── agents/                    # Saved .pkl files (one per board config)
└── requirements.txt
```

---

## Module Breakdown

### `envs/mnk_game.py`

Generalized game engine. `MNKGame(m, n, k)` runs games on any m×n board with k-in-a-row winning. Key exports:

- `MNKGame` — runs a full game between two policies via `run_round()`
- `RandomPolicy` — uniform-random baseline
- `legal_moves(board)` — returns all empty cells
- `_check_winner(board, m, n, k, last_move, marker)` — O(4k) win detection

### `rl_training/feature_extractor.py`

Extracts the 18-dimensional feature vector `φ(board, move, marker, k)` from any board position. Operates on the board state **before** the move is placed.

### `rl_training/mnk_q_agent.py`

`MNKQLearningAgent` — the linear FA game agent.

| Interface | Description |
|-----------|-------------|
| `choose_move(board, marker, enemy, k)` | ε-greedy move selection |
| `q_value(board, move, marker, enemy, k)` | Returns `w · φ` |
| `train_episode(game, opponent, rewards)` | Plays one game, updates weights |
| `save(path)` / `load(path)` | Saves/loads the weight vector |

Module-level functions:

| Function | Description |
|----------|-------------|
| `train_mnk_agent(n, k, ...)` | Trains one agent to convergence, saves a single `*_trained.pkl` |
| `build_opponent_pool(path, num_levels)` | Loads one model, returns list of agents with `EPSILON_LEVELS` |
| `EPSILON_LEVELS` | `[1.0, 0.7, 0.4, 0.15, 0.0]` — epsilon per difficulty level |

### `rl_training/train_base_agent.py`

CLI wrapper around `train_mnk_agent`. Trains one agent and saves a single file:
`agents/mnk_{n}x{n}_k{k}_trained.pkl`

### `rl_training/difficulty_scaling.py`

The main RL-based DDA system. Three components:

**`QValueDifficultyEstimator`** — Measures move quality by ranking the player's actual move against every legal move using Q-values from the opponent pool. A move that scores in the 75th percentile across all levels → estimated player skill ≈ 3.0.

**`QValueDDAMetrics`** — Sliding window tracker (default: last 20 games) for win rate, move quality, game duration, and difficulty estimate variance. Computes the reward signal after each game:
```
reward = -(
    q_match_weight     × |opponent_difficulty − estimated_player_skill|
  + duration_weight    × |game_duration − target_duration|
  + variance_weight    × Var(recent_estimated_difficulty)
  + switch_weight      × I[difficulty changed this episode]
)
```

**`QLearningDDA`** — RL meta-agent with its own Q-table keyed on a 4-tuple state `(skill_bucket, duration_bucket, variance_bucket, current_level)`. Actions: `{-1, 0, +1}`. ε-greedy with decay (0.25 → 0.04).

### `rl_training/meta_agent.py`

Rule-based baseline DDA controller. Adjusts difficulty up/down based on win rate over a sliding window of 5 games. No learning — used as a comparison baseline against `QLearningDDA`.

---

## Quickstart

### 1. Setup

```bash
git clone git@github.com:nurgissas/GameAI.git
cd GameAI/adaptive_difficulty_rl
python -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Train the agent

Trains one agent to convergence and saves a single model file.

```bash
# 3×3 Tic-Tac-Toe
python rl_training/train_base_agent.py --n 3 --k 3

# 5×5 board, k=4
python rl_training/train_base_agent.py --n 5 --k 4 --episodes 60000

# 15×15 Gomoku
python rl_training/train_base_agent.py --n 15 --k 5 --episodes 200000
```

Expected output:
```
Training MNK(3,3,3) for 30,000 episodes
------------------------------------------------------------
  Episode 6,000 / 30,000 | Win rate: 64.7%
  Episode 12,000 / 30,000 | Win rate: 66.3%
  ...
Saved -> agents/mnk_3x3_k3_trained.pkl  (non-zero weights: 18)
------------------------------------------------------------
Done. Saved to agents/mnk_3x3_k3_trained.pkl
```

This produces **one file**: `agents/mnk_{n}x{n}_k{k}_trained.pkl`. The 5 difficulty levels are created from this single file at runtime by `build_opponent_pool()`.

### 3. Verify the difficulty gradient

```bash
python experiments/evaluate_agent.py --n 3 --k 3
```

Expected output:
```
Level    Epsilon    Win Rate
-----------------------------------
  1      ε=1.00     ~45%
  2      ε=0.70     ~60%
  3      ε=0.40     ~63%
  4      ε=0.15     ~76%
  5      ε=0.00     ~84%
```

### 4. Test the rule-based adaptive system

```bash
python experiments/test_adaptation.py --n 3 --k 3 --games 50
```

Expected: player win rate near 50%, difficulty oscillating across middle levels.

### 5. Play against the adaptive opponent

Start a local browser-based playable MNK/Gomoku session:

```bash
python experiments/play_mnk.py 15 15 5
```

The three positional numbers are board width, board height, and the number of
connected moves needed to win. If the matching trained model does not exist yet,
the script trains one and saves it under `agents/` before opening the web UI.

The right panel shows the current opponent difficulty. After each completed
game, the player's moves are scored with the learned Q-value policy, the
estimated player difficulty is updated, and the DDA controller selects the next
opponent difficulty.

Useful options:

```bash
python experiments/play_mnk.py 7 7 5 --episodes 10000
python experiments/play_mnk.py 15 15 5 --port 9000 --no-open
```

### 6. Optional analysis

```bash
python experiments/compare_rewards.py --n 3 --k 3        # ~5 min
python experiments/sensitivity_analysis.py --n 3 --k 3   # ~15 min
```

---

## Key Concepts

**Linear Function Approximation** — Instead of storing Q-values per state, a weight vector `w` is learned such that `Q(s,a) ≈ w·φ(s,a)`. The feature vector `φ` encodes board structure (runs, threats, position) in a fixed 18 dimensions regardless of board size. Updated via semi-gradient TD.

**Epsilon-based difficulty pool** — A single trained agent is reused at multiple epsilon values to form the opponent pool. High epsilon means mostly random play (easy); epsilon=0 means fully greedy play (hard). This is more reliable than training snapshots because it guarantees a monotonic gradient independent of training dynamics.

**Q-Learning (DDA controller)** — A second, separate Q-table that learns when to change difficulty. Its state is a 4-tuple of bucketed metrics; its actions are `{-1, 0, +1}`. It learns that unnecessary switches are costly (switch penalty) and that matching opponent skill to player skill is the primary goal (mismatch penalty).

**Move quality estimation** — Instead of only asking "did the player win?", the system asks "how good was the player's move?" by comparing it against all legal options using the opponent pool's Q-values. This gives a continuous skill signal within each game, not just at the end.

**Reward design** — The DDA reward deliberately avoids directly rewarding 50% win rate. Instead it penalizes the gap between estimated player skill and current opponent difficulty. The theory is that matched difficulty naturally produces balanced outcomes without overfitting to win rate as a metric.

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Generalized MNK game engine (any n, k) | Done |
| Linear FA feature extractor (18-dim, board-size independent) | Done |
| Linear FA agent with semi-gradient TD | Done |
| Single-model training (`train_mnk_agent`) | Done |
| Epsilon-based opponent pool (`build_opponent_pool`) | Done |
| Rule-based meta-agent (DDA baseline) | Done |
| RL DDA controller | Done |
| Evaluation and analysis scripts | Done |

---

## Requirements

```
numpy
matplotlib
pandas
```

Python 3.8+. All dependencies are pip-installable.
