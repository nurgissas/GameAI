## English Version

### 1. Module Purpose

`difficulty_scaling.py` estimates the player's skill online from the moves they make during gameplay, and adjusts the opponent difficulty to match that estimated skill.

The overall flow is:

```text
1. The player makes a move.
2. The move is evaluated using the Q-values from a trained board-playing policy.
3. The Q-value percentile is converted into an estimated player difficulty.
4. The difference between the current opponent difficulty and the estimated player difficulty becomes the reward signal.
5. The DDA Q-learning controller learns whether to decrease, keep, or increase the difficulty.
```

This module does not train the board-playing agent itself. Instead, it uses the Q-values produced by an already trained board-playing policy to perform difficulty scaling.

---

### 2. `QValueDifficultyEstimator`

Role:

```text
Evaluate how good the player's move is using Q-values,
then convert that move quality into an estimated player difficulty.
```

Initialization:

```python
q_estimator = QValueDifficultyEstimator(policies)
```

Required input:

```text
policies:
  A list of opponent policies for each difficulty level.
  Each policy must provide q_value(board, move, marker, enemy_marker, k).
```

Example use of `estimate_move()`:

```python
estimate = q_estimator.estimate_move(
    board=board_before,
    move=player_move,
    marker=PLAYER,
    enemy_marker=OPPONENT,
    k=5,
)
```

Return value:

```python
{
    "difficulty": estimated_player_difficulty,
    "quality": average_move_quality,
}
```

Meaning:

```text
difficulty:
  Estimated player skill converted from the Q-value of the player's move.

quality:
  Average quality of the move across the policy pool.
```

Current difficulty estimation method:

```text
1. Compute Q-values for all legal moves using the strongest policy.
2. Compute the percentile rank of the actual player move among all legal moves.
3. Convert it to difficulty using percentile * max_difficulty_level.
```

Example:

```text
max difficulty = 4
player move percentile = 0.75

estimated_player_difficulty = 0.75 * 4 = 3.0
```

---

### 3. `RewardConfig`

Role:

```text
Configure the weights of the penalty terms used in the DDA reward.
```

Example:

```python
reward_config = RewardConfig(
    target_duration=0.48,
    duration_weight=0.70,
    duration_variance_weight=0.35,
    switch_weight=0.08,
    q_match_weight=1.20,
)
```

Parameter meanings:

```text
target_duration:
  Target round duration.
  Duration is normalized as moves / board_cells.

duration_weight:
  Penalty weight when the actual round duration deviates from target_duration.

duration_variance_weight:
  Penalty weight when the estimated difficulty fluctuates too much in the recent window.

switch_weight:
  Penalty weight when the difficulty changes from the previous episode.

q_match_weight:
  Penalty weight for the mismatch between current opponent difficulty and estimated player difficulty.
  This is the most important reward term.
```

---

### 4. `QValueDDAMetrics`

Role:

```text
Store recent estimated difficulties, move qualities, durations, and difficulty history,
then compute the DDA state and reward.
```

Initialization:

```python
metrics = QValueDDAMetrics(
    window=20,
    reward_config=reward_config,
    n_levels=len(opponent_pool),
)
```

Required parameters:

```text
window:
  Number of recent episodes used to compute moving averages and variances.

reward_config:
  Reward weight configuration.

n_levels:
  Number of difficulty levels.
  For example, if difficulties are 0 through 4, n_levels = 5.
```

Return value of `state()`:

```python
state = metrics.state()
```

State format:

```text
(
  estimated player difficulty bucket,
  duration bucket,
  estimated difficulty variance bucket,
  current difficulty
)
```

This state is used as the Q-table key for `QLearningDDA`.

---

### 5. Reward Design

The reward is computed inside `QValueDDAMetrics.update()`.

```python
difficulty_error = abs(difficulty - estimated) / max_level
duration_penalty = abs(duration - target_duration)

reward = -(
    q_match_weight * difficulty_error
    + duration_weight * duration_penalty
    + duration_variance_weight * estimate_var / max_level
    + switch_weight * switch
)
```

As a formula:

```text
reward = -(
  w_match * |current_difficulty - estimated_player_difficulty|
  + w_dur * |recent_duration - target_duration|
  + w_var * Var(recent_estimated_difficulty)
  + w_sw * I[difficulty changed]
)
```

Meaning of each term:

```text
difficulty_error:
  Difference between the current opponent difficulty and the player difficulty estimated from Q-values.
  This is the core reward term.

duration_penalty:
  Penalty when the game ends too quickly or lasts too long.

estimate_var:
  Penalty when recent estimated player difficulty is unstable.

switch:
  Penalty to discourage changing difficulty too frequently.
```

Important point:

```text
This reward does not directly force a 50% win rate.
Instead, it estimates player skill from the Q-values of the player's moves,
then tries to match the opponent difficulty to that estimated skill.
```

---

### 6. `QLearningDDA`

Role:

```text
Learn whether to decrease, keep, or increase difficulty from the current state.
```

Initialization:

```python
dda_controller = QLearningDDA(
    n_levels=len(opponent_pool),
    rng=random.Random(13),
    alpha=0.18,
    gamma=0.90,
    epsilon=0.25,
)
```

Main parameters:

```text
n_levels:
  Number of difficulty levels.

rng:
  random.Random object.
  Used for exploration and tie-breaking.

alpha:
  Q-learning learning rate.

gamma:
  Future reward discount factor.

epsilon:
  Exploration probability.
  With probability epsilon, the controller selects a random action.

epsilon_decay:
  Rate at which exploration decreases over episodes.

min_epsilon:
  Minimum epsilon value.
```

Action space:

```text
-1 -> difficulty down
 0 -> stay
+1 -> difficulty up
```

Usage:

```python
state = metrics.state()
difficulty = dda_controller.select_difficulty(state)
```

After the episode ends:

```python
dda_controller.observe(state, difficulty, reward, next_state)
```

The DDA Q-table is updated inside `observe()`.

---

### 7. `run_online_difficulty_learning_episode`

Role:

```text
Helper function that runs one full episode of online difficulty learning.
```

Example:

```python
result = run_online_difficulty_learning_episode(
    game=game,
    player_policy=player,
    opponent_pool=opponent_pool,
    q_estimator=q_estimator,
    metrics=metrics,
    dda_controller=dda_controller,
    player_first=True,
)
```

Required parameters:

```text
game:
  MNKGame environment.

player_policy:
  Player policy.
  This can be a human input policy or a simulated player.

opponent_pool:
  List of opponent policies for each difficulty level.

q_estimator:
  QValueDifficultyEstimator instance.

metrics:
  QValueDDAMetrics instance.

dda_controller:
  QLearningDDA instance.

player_first:
  Whether the player moves first.
```

Return value:

```python
{
    "difficulty": difficulty,
    "winner": result.winner,
    "moves": result.moves,
    "estimated_player_difficulty": update["estimated_difficulty"],
    "player_move_quality": update["player_move_quality"],
    "difficulty_error": update["difficulty_error"],
    "reward": update["reward"],
    "next_difficulty_state": next_state,
}
```

Output meanings:

```text
difficulty:
  Opponent difficulty selected for this episode.

winner:
  Winner of the round. PLAYER / OPPONENT / draw.

moves:
  Number of moves, used as round duration.

estimated_player_difficulty:
  Player difficulty estimated from the recent window.

player_move_quality:
  Average Q-value percentile quality of the player's moves.

difficulty_error:
  Difference between current opponent difficulty and estimated player difficulty.

reward:
  Reward used for the DDA Q-learning update.

next_difficulty_state:
  DDA state used for the next episode.
```

---

### 8. Usage Example

```python
import random

from difficulty_scaling import (
    QLearningDDA,
    QValueDDAMetrics,
    QValueDifficultyEstimator,
    RewardConfig,
    run_online_difficulty_learning_episode,
)

rng = random.Random(13)

opponent_pool = make_rl_opponent_pool(size=7, k=5, rng=rng)

q_estimator = QValueDifficultyEstimator(opponent_pool)

reward_config = RewardConfig(
    target_duration=0.48,
    q_match_weight=1.20,
    duration_weight=0.70,
    duration_variance_weight=0.35,
    switch_weight=0.08,
)

metrics = QValueDDAMetrics(
    window=20,
    reward_config=reward_config,
    n_levels=len(opponent_pool),
)

dda_controller = QLearningDDA(
    n_levels=len(opponent_pool),
    rng=rng,
)

for episode in range(100):
    result = run_online_difficulty_learning_episode(
        game=game,
        player_policy=player_policy,
        opponent_pool=opponent_pool,
        q_estimator=q_estimator,
        metrics=metrics,
        dda_controller=dda_controller,
        player_first=(episode % 2 == 0),
    )

    print(result)
```

---

### 9. Summary

The module can be summarized as follows:

```text
Input:
  board state, player move, trained Q-policy, current difficulty, game result

Process:
  Evaluate the player move using Q-values
  Estimate player difficulty
  Compare it with current opponent difficulty
  Compute reward
  Update the down/stay/up difficulty adjustment policy using Q-learning

Output:
  selected difficulty
  estimated player difficulty
  player move quality
  difficulty error
  DDA reward
  next DDA state
```

In one sentence:

```text
difficulty_scaling.py estimates player skill online from the Q-values of the moves the player actually makes,
and uses Q-learning to adjust the opponent difficulty to match that estimated skill.
```
