# How the System Works — Concept Guide

This document explains the ideas behind the project in plain terms, how each piece
connects to the others, and where the implementation is heading.

---

## 1. The Core Problem

Most game AIs are designed to win as often as possible. That is the wrong goal
when your player is a human trying to enjoy a game.

A player who wins every game gets bored. A player who loses every game gets
frustrated. The sweet spot is somewhere around 50% — the player wins roughly half
the time and feels like they are improving.

The naive fix is to let the designer pick a difficulty slider. The problem is that
players improve (or decline) over a session, and one fixed setting stops fitting.

This project builds a system that watches how well the player is actually playing
and automatically adjusts the opponent difficulty in real time.

---

## 2. Two Separate Learning Problems

The system has two AI agents that solve two completely different problems.

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Agent 1: Game-playing agent (MNKQLearningAgent)   │
│   Learns HOW TO PLAY the game well.                 │
│   Trained once. Saved as a file. Never changes      │
│   during a play session.                            │
│                                                     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                                                     │
│   Agent 2: DDA controller (QLearningDDA)            │
│   Learns WHICH DIFFICULTY LEVEL TO SHOW next.       │
│   Runs live during a play session.                  │
│   Updates after every game.                         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

They share no weights and never talk to each other directly. The DDA controller
uses the game-playing agents as tools to evaluate what the player is doing.

---

## 3. Q-Learning in Plain Terms

Both agents use Q-Learning, so it is worth understanding the idea once.

Imagine you are trying to learn to play chess. Every time you make a move, you
do not immediately know if it was a good move — you find out at the end of the
game. Q-Learning is a way to work backwards from the outcome and assign credit
(or blame) to earlier moves.

The agent keeps a table — the Q-table — that maps every situation it has seen
to a score for each possible action:

```
Q-table:
  (board state A, move to center) → 0.72
  (board state A, move to corner) → 0.45
  (board state B, move to edge)   → -0.30
  ...
```

After a game ends, the agent updates the score of the last move it made:

```
new score = old score + α × (actual reward + γ × best future score − old score)
```

- `α` (alpha): how fast to learn. High = update aggressively. Low = update slowly.
- `γ` (gamma): how much to value future rewards vs. immediate ones. Close to 1 =
  care a lot about the future.
- `actual reward`: +1 if the agent won, -1 if it lost, 0 if it drew.
- `best future score`: the highest Q-value available in the next state. For a
  terminal state (game over) this is 0.

Over many games, scores propagate backwards. Moves that reliably lead to wins
accumulate high scores; moves that tend to lose accumulate negative scores.

**Exploration vs exploitation:** during training, the agent picks a random move
with probability ε (epsilon) instead of its best known move. This is called
exploration — without it, the agent would get stuck repeating the first decent
strategy it found instead of searching for better ones. ε decreases over time
as the agent becomes more confident.

---

## 4. The Game-Playing Agent (`MNKQLearningAgent`)

Located in `rl_training/mnk_q_agent.py`.

### Board representation

The board is a 2-D list of integers:
- `0` = empty cell
- `+1` = this player's stone
- `-1` = opponent's stone

### State normalization

The same agent can play as either PLAYER or OPPONENT. To avoid doubling the
Q-table size, the board is always converted to the current player's perspective
before it is used as a key:

```python
state = tuple(tuple(marker * cell for cell in row) for row in board)
```

If the agent is PLAYER (+1), the board stays as-is. If the agent is OPPONENT (-1),
all signs are flipped so the agent's own pieces are always +1 in the key.
This means one Q-table works for both roles.

### Training

The agent trains by playing against a random opponent. Each episode:
1. Randomly decide whether the agent goes first or second.
2. Alternate moves until someone wins or the board fills.
3. At the end, update the Q-value of the last move the agent made:
   - Won → reward +1
   - Lost → reward −1
   - Drew → reward 0

After 50,000 episodes against a random opponent, the agent reaches around 95%
win rate on Tic-Tac-Toe (3×3). Saving snapshots at 10,000-episode intervals
produces five agents at increasing skill levels — the opponent pool.

### The two methods DDA needs

```python
agent.choose_move(board, marker, enemy_marker, k) → (row, col)
agent.q_value(board, move, marker, enemy_marker, k) → float
```

`choose_move` is how the agent plays during a game. `q_value` is used by the
DDA system to measure how good the player's moves are (see Section 6).

---

## 5. The MNK Game Engine (`envs/mnk_game.py`)

MNK stands for: M rows, N columns, K in a row to win.

| Configuration | M | N | K |
|---------------|---|---|---|
| Tic-Tac-Toe   | 3 | 3 | 3 |
| Gomoku        |15 |15 | 5 |

The engine is generic — the same code runs both games. Key decisions:

**Win detection** runs in O(4k) per move by only checking the four axis directions
through the last placed stone, rather than scanning the entire board.

**`run_round()`** handles one full game. It accepts any policy object with a
`choose_move` method, alternates turns, and calls an optional
`player_move_observer` callback before each player move. The callback receives
the board before the move is placed — this is important so it can see all legal
options including the one the player is about to choose.

---

## 6. Measuring Player Skill (`QValueDifficultyEstimator`)

Located in `rl_training/difficulty_scaling.py`.

Winning and losing tells you something about skill, but not much — a lucky
random player can win. What tells you more is **how good each individual move
was**.

`QValueDifficultyEstimator` uses the trained opponent pool to score each move:

1. For the current board state, find all legal moves.
2. Ask the strongest trained policy: what is the Q-value of each legal move?
3. Find the percentile rank of the player's actual move among all legal options.
   - Player chose the best available move → percentile close to 1.0
   - Player chose a weak move → percentile close to 0.0
4. Multiply percentile by the maximum difficulty level to get an estimated
   skill level.

```
Example (4 difficulty levels, 0–3):
  All legal move Q-values: [0.1, 0.3, 0.7, 0.8]
  Player chose the move with Q-value 0.7
  3 out of 4 values are ≤ 0.7  →  percentile = 0.75
  estimated skill = 0.75 × 3 = 2.25
```

This runs after every player move, so the estimate is updated continuously
throughout the game.

---

## 7. Tracking the Session (`QValueDDAMetrics`)

Located in `rl_training/difficulty_scaling.py`.

After each game, `QValueDDAMetrics` keeps a sliding window of the last N games
and tracks four things:

| Metric | What it measures |
|--------|-----------------|
| Win rate | Did the player win or lose? |
| Move quality | Average skill estimate from QValueDifficultyEstimator |
| Game duration | How many moves did the game last (normalized to board size)? |
| Difficulty variance | How stable is the estimated skill across recent games? |

These four values are bucketed into discrete ranges to form the DDA state:

```
state = (skill_bucket, duration_bucket, variance_bucket, current_level)
```

This state tuple is the key for the DDA agent's Q-table.

### The reward signal

After each game, a reward is computed for the DDA agent:

```
reward = -(
    1.20 × |current_opponent_level − estimated_player_skill|   ← match error
  + 0.70 × |game_duration − target_duration|                   ← too short/long
  + 0.35 × variance_of_recent_skill_estimates                  ← instability
  + 0.08 × I[difficulty changed this game]                     ← unnecessary switches
)
```

Everything is a penalty (negative), so the DDA agent learns to minimize these.
The most important term is the mismatch between opponent difficulty and estimated
player skill. The switch penalty discourages the system from oscillating
between levels unnecessarily.

Notice this reward does **not** directly chase a 50% win rate. Instead it chases
**skill match**. In theory, when the opponent difficulty matches the player's
skill, the player wins about 50% of the time — but the system gets there by
understanding skill rather than blindly tracking outcomes.

---

## 8. The DDA Controller (`QLearningDDA`)

Located in `rl_training/difficulty_scaling.py`.

This is the second Q-learning agent. Its job is to decide whether to go up,
stay, or go down in difficulty after each game.

```
Actions: [-1] decrease  [0] stay  [+1] increase
```

It has its own Q-table keyed on the 4-tuple state from `QValueDDAMetrics`.
It learns using the reward defined above. After each game:

```
Q(state, action) += α × [reward + γ × max Q(next_state) − Q(state, action)]
```

With epsilon-greedy exploration (starting at 0.25, decaying to 0.04), it
gradually learns when to change difficulty and when to hold steady.

The full loop for one episode:

```
state  = metrics.state()
level  = dda.select_difficulty(state)
result = game.run_round(player, opponent_pool[level], observer=estimator.estimate_move)
update = metrics.update(result, level)
dda.observe(state, level, update["reward"], metrics.state())
```

---

## 9. How It All Fits Together

```
                ┌──────────────────┐
                │  Opponent pool   │  (5 MNKQLearningAgents at increasing skill,
                │  levels 0 – 4    │   trained offline and saved to agents/)
                └────────┬─────────┘
                         │ q_value(), choose_move()
                         ▼
Player makes ──► QValueDifficultyEstimator ──► estimated skill score per move
a move                   │
                         │ aggregated per game
                         ▼
                  QValueDDAMetrics ──► state tuple, reward signal
                         │
                         ▼
                  QLearningDDA ──► picks difficulty for next game
                         │
                         └──► opponent_pool[difficulty] faces the player
```

---

## 10. What the Baseline Does Differently (`meta_agent.py`)

`meta_agent.py` is the rule-based comparison. It watches your last 5 win/loss
results and applies two fixed rules:

```
win rate > 60%  →  increase difficulty
win rate < 40%  →  decrease difficulty
```

No learning. No move quality. No duration signal.

The DDA controller (`difficulty_scaling.py`) is expected to outperform this
baseline, especially in two situations:
- A player who wins by luck but plays badly → meta_agent raises difficulty,
  DDA controller holds because it sees weak moves.
- A player who loses but is improving → meta_agent lowers difficulty, DDA
  controller detects rising move quality and holds.

The experiments in `experiments/` compare these two approaches.

---

## 11. Where the Project Is Going

### Stage 1 — Tic-Tac-Toe (current)

The full pipeline is implemented and running:
- `MNKGame(3, 3, 3)` as the environment
- `MNKQLearningAgent` trained and saved as 5 levels
- Full DDA loop operational

The experiment scripts measure the three stability metrics from the proposal:
1. Win rate variance across episodes
2. Difficulty switching frequency
3. Convergence speed (how quickly DDA adapts to a new skill level)

### Stage 2 — Gomoku

The same code runs Gomoku by changing three numbers:

```python
game = MNKGame(15, 15, 5)
```

The challenge is scale. A 15×15 board has 225 cells. Q-tables for Tic-Tac-Toe
have ~5,000 states. For Gomoku the state space is astronomically larger —
a tabular Q-table will not work well without tricks like:
- Restricting the board window to k×k cells around recent moves
- Using pattern-based features instead of raw board state
- Switching to a function approximator (neural network)

For the scope of this project, Gomoku training will be attempted with the same
tabular Q-learning approach but with more training episodes, and the results will
be analyzed in the experiments section.

### Evaluation plan

The proposal defines four metrics to report:

| Metric | Script | What to look for |
|--------|--------|-----------------|
| Stable win rate ~50% | `test_adaptation.py` | Mean win rate across 100 games |
| Win rate variance | to be added | Lower = more consistent experience |
| Switching frequency | `difficulty_scaling.py` metrics | DDA switches fewer times than meta_agent |
| Convergence speed | to be added | Episodes until DDA settles after a skill jump |

The reward formulation comparison (`compare_rewards.py`) will run both the
rule-based and RL-based controllers under different reward weights and report
which configuration best maintains balance.

---

## 12. Files at a Glance

```
envs/
  mnk_game.py          → game logic, legal_moves, win detection, run_round
  tictactoe.py         → legacy TicTacToe (uses different board format)

rl_training/
  mnk_q_agent.py       → game-playing agent with q_value() interface
  difficulty_scaling.py → QValueDifficultyEstimator, QValueDDAMetrics, QLearningDDA
  meta_agent.py        → rule-based baseline for comparison
  train_base_agent.py  → legacy training script for tictactoe.py agents
  q_learning_agent.py  → legacy Q-agent used by train_base_agent.py

experiments/
  test_adaptation.py   → end-to-end test with rule-based meta_agent
  evaluate_agent.py    → win rate per difficulty level
  compare_rewards.py   → reward formulation comparison
  sensitivity_analysis.py → hyperparameter sweep
  failure_analysis.py  → missed wins and blocks per level
```
