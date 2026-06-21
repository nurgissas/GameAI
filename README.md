# Adaptive Difficulty Scaling with Reinforcement Learning

Authors: Nurgissa Sailaubek, Youngjin Cho

A two-stage game AI for online dynamic difficulty adjustment (DDA) in generalized MNK games. A board-playing agent learns to play via linear-function-approximation Q-learning; a separate Q-learning controller estimates player skill from move quality and decides when to raise, hold, or lower difficulty. Supports any m×n board with k-in-a-row to win — Tic-Tac-Toe is `MNKGame(3,3,3)`, Gomoku is `MNKGame(15,15,5)`.

See the project report for the full method, equations, experiments, and analysis.

---

## How It Works

```
Player makes a move
        ↓
QValueDifficultyEstimator ranks that move against all legal moves
using Q-values from the opponent pool → estimates player skill
        ↓
QValueDDAMetrics tracks a sliding window:
  win rate · move quality · game duration · difficulty variance
        ↓
QLearningDDA (RL controller) selects: [-1] down / [0] stay / [+1] up
        ↓
Next game uses the new difficulty level; the DDA Q-table updates from a
reward penalizing skill mismatch, bad duration, variance, and switching
```

---

## Approach

- **Board-playing agent** — Linear function approximation `Q(s, a) = w · φ(s, a)` over an 18-dimensional feature vector, trained with semi-gradient TD. The fixed feature size works on any board dimension.
- **Tactical rules** — Before using learned Q-values, `choose_move` checks forced moves (own win, block opponent win, extend/block an open threat), firing with probability `1 − ε` so the difficulty ordering is preserved.
- **Mixed-curriculum training** — Starts against a random opponent, then blends random, self-play, and past snapshots so the agent learns to defend rather than only attack.
- **Epsilon-based difficulty pool** — One trained model is reused at five exploration rates to form ordered opponent levels (more reliable than training snapshots, which converge to near-identical strength):

| Level | Epsilon | Behavior |
|-------|---------|----------|
| 1 | 1.00 | Plays randomly — easiest |
| 2 | 0.70 | 70% random, 30% greedy |
| 3 | 0.40 | 40% random, 60% greedy |
| 4 | 0.15 | Nearly greedy |
| 5 | 0.00 | Always picks the best move — hardest |

- **DDA controller** — A separate tabular Q-learning agent keyed on bucketed `(skill, duration, variance, current level)`, with actions `{-1, 0, +1}`, learning when changing difficulty actually helps.

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
│   ├── train_base_agent.py    # CLI: trains and saves one model
│   ├── meta_agent.py          # Rule-based difficulty selector (baseline)
│   └── difficulty_scaling.py  # RL-based DDA controller (main system)
│
├── experiments/
│   ├── evaluate_agent.py      # Win rate of each epsilon level vs random
│   ├── test_adaptation.py     # End-to-end rule-based DDA simulation
│   ├── play_mnk.py            # Browser-based playable demo with online DDA
│   ├── compare_rewards.py     # Compares reward formulations
│   └── sensitivity_analysis.py # Hyperparameter sweep (α, γ, ε)
│
├── agents/                    # Saved .pkl files (one per board config)
└── requirements.txt
```

---

## Quickstart

### 1. Setup

```bash
git clone git@github.com:nurgissas/GameAI.git
cd GameAI/
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

# 7×7 board, k=5
python rl_training/train_base_agent.py --n 7 --k 5 --episodes 10000

# 15×15 Gomoku
python rl_training/train_base_agent.py --n 15 --k 5 --episodes 200000
```

This produces one file: `agents/mnk_{n}x{n}_k{k}_trained.pkl`. The 5 difficulty levels are created from this single file at runtime by `build_opponent_pool()`.

### 3. Verify the difficulty gradient

```bash
python experiments/evaluate_agent.py --n 7 --k 5 --games 200
```

Win rate against a random opponent increases monotonically from level 1 to level 5. See the project report's 7×7 benchmark for the measured figures.

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

## Requirements

```
numpy
matplotlib
pandas
```

Python 3.8+. All dependencies are pip-installable.
