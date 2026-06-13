"""
MNK Linear Function Approximation Agent.

Replaces the tabular Q-table with a weight vector:
    Q(board, move, marker) = weights · φ(board, move, marker, k)

where φ is the 18-dimensional feature vector from feature_extractor.py.
This scales to any board size because the weight vector has fixed size (18),
whereas a Q-table would need a separate entry for every possible board state
(which grows exponentially with board size).

Update rule (semi-gradient TD):
    δ = r + γ · max_a' Q(s', a') − Q(s, a)
    weights += α · δ · φ(s, a)
"""

from __future__ import annotations

import os
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mnk_game import (
    OPPONENT, PLAYER, MNKGame, RandomPolicy,
    _check_winner, legal_moves,
)
from rl_training.feature_extractor import N_FEATURES, extract


class MNKQLearningAgent:
    """
    Linear FA agent compatible with the MNK policy interface.

    Set epsilon=0.0 before using as an opponent pool policy so the
    agent always picks its best known move.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        discount_factor: float = 0.99,
        epsilon: float = 0.3,
        seed: int | None = None,
        td_clip: float = 5.0,
        weight_clip: float = 10.0,
        use_tactical_rules: bool = True,
    ) -> None:
        self.weights = np.zeros(N_FEATURES, dtype=np.float64)
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.td_clip = td_clip
        self.weight_clip = weight_clip
        self.use_tactical_rules = use_tactical_rules
        self._rng = random.Random(seed)
        self.stats = {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "episodes": 0,
            "clipped_updates": 0,
            "skipped_updates": 0,
        }

    # ------------------------------------------------------------------
    # Policy interface — called by MNKGame and QValueDifficultyEstimator
    # ------------------------------------------------------------------

    def q_value(
        self,
        board: list[list[int]],
        move: tuple[int, int],
        marker: int,
        enemy_marker: int,
        k: int,
    ) -> float:
        value = float(np.dot(self.weights, extract(board, move, marker, k)))
        if not np.isfinite(value):
            return 0.0
        return value

    def choose_move(
        self,
        board: list[list[int]],
        marker: int,
        enemy_marker: int,
        k: int,
    ) -> tuple[int, int]:
        moves = legal_moves(board)
        if not moves:
            raise ValueError("choose_move called with no legal moves")

        if self._should_use_tactical_rule():
            own_winners = self._winning_moves(board, marker, k, moves)
            if own_winners:
                return self._best_q_move(board, own_winners, marker, enemy_marker, k)

            enemy_winners = self._winning_moves(board, enemy_marker, k, moves)
            if enemy_winners:
                safe_blocks = self._moves_that_prevent_immediate_loss(
                    board,
                    moves,
                    marker,
                    enemy_marker,
                    k,
                )
                defensive_moves = safe_blocks if safe_blocks else enemy_winners
                return self._best_q_move(board, defensive_moves, marker, enemy_marker, k)

            own_open_threat_moves = self._open_threat_blocks(
                board,
                marker,
                k,
                min_stones=max(1, k - 2),
            )
            legal_attacks = [move for move in own_open_threat_moves if move in moves]
            if legal_attacks:
                return self._best_q_move(board, legal_attacks, marker, enemy_marker, k)

            enemy_open_threat_blocks = self._open_threat_blocks(
                board,
                enemy_marker,
                k,
                min_stones=max(1, k - 2),
            )
            legal_blocks = [move for move in enemy_open_threat_blocks if move in moves]
            if legal_blocks:
                return self._best_q_move(board, legal_blocks, marker, enemy_marker, k)

        if self._rng.random() < self.epsilon:
            return self._rng.choice(moves)

        return self._best_q_move(board, moves, marker, enemy_marker, k)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_episode(
        self,
        game: MNKGame,
        opponent=None,
        rewards: dict | None = None,
    ) -> int:
        """
        Play one training game and update weights via semi-gradient TD.

        The agent is randomly assigned PLAYER or OPPONENT each episode so
        it learns both sides. Weights are updated at every agent turn using
        the state seen after the opponent responded, and at terminal states.

        Args:
            game:     MNKGame instance
            opponent: policy with choose_move interface (default: RandomPolicy)
            rewards:  {"win": float, "loss": float, "draw": float}

        Returns:
            winner marker (PLAYER, OPPONENT, or 0 for draw)
        """
        if opponent is None:
            opponent = RandomPolicy()
        if rewards is None:
            rewards = {"win": 1.0, "loss": -1.0, "draw": 0.0}

        board = [[0] * game.n for _ in range(game.m)]
        agent_marker = PLAYER if self._rng.random() < 0.5 else OPPONENT
        enemy_marker = -agent_marker
        current = PLAYER

        prev_phi: np.ndarray | None = None
        prev_q: float = 0.0
        winner = 0

        while True:
            moves = legal_moves(board)

            if not moves:
                winner = 0
                if prev_phi is not None:
                    self._td_update(prev_phi, prev_q, rewards["draw"])
                break

            if current == agent_marker:
                # Update the previous agent step now that we can see the next state
                if prev_phi is not None:
                    max_next_q = max(
                        self.q_value(board, m, agent_marker, enemy_marker, game.k)
                        for m in moves
                    )
                    self._td_update(prev_phi, prev_q, self.gamma * max_next_q)

                move = self.choose_move(board, agent_marker, enemy_marker, game.k)
                phi = extract(board, move, agent_marker, game.k)
                prev_q = self.q_value(board, move, agent_marker, enemy_marker, game.k)
                prev_phi = phi

                board[move[0]][move[1]] = agent_marker

                if _check_winner(board, game.m, game.n, game.k, move, agent_marker):
                    winner = agent_marker
                    self._td_update(prev_phi, prev_q, rewards["win"])
                    break

            else:
                opp_move = opponent.choose_move(board, current, -current, game.k)
                board[opp_move[0]][opp_move[1]] = current

                if _check_winner(board, game.m, game.n, game.k, opp_move, current):
                    winner = current
                    if prev_phi is not None:
                        self._td_update(prev_phi, prev_q, rewards["loss"])
                    break

            current = -current

        self.stats["episodes"] += 1
        if winner == agent_marker:
            self.stats["wins"] += 1
        elif winner == 0:
            self.stats["draws"] += 1
        else:
            self.stats["losses"] += 1

        return winner

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        if not self.is_healthy():
            raise ValueError("Refusing to save agent with non-finite weights.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "weights": self.weights,
                    "stats": self.stats,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "td_clip": self.td_clip,
                    "weight_clip": self.weight_clip,
                    "use_tactical_rules": self.use_tactical_rules,
                },
                f,
            )
        print(f"Saved -> {path}  (non-zero weights: {np.count_nonzero(self.weights)})")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.weights = np.asarray(data["weights"], dtype=np.float64)
        if not self.is_healthy():
            raise ValueError(
                f"Loaded model has non-finite weights and cannot be used: {path}"
            )
        self.stats = data.get(
            "stats", {"wins": 0, "losses": 0, "draws": 0, "episodes": 0}
        )
        self.stats.setdefault("clipped_updates", 0)
        self.stats.setdefault("skipped_updates", 0)
        self.use_tactical_rules = data.get("use_tactical_rules", self.use_tactical_rules)
        print(f"Loaded <- {path}")

    def win_rate(self) -> float:
        total = self.stats["wins"] + self.stats["losses"] + self.stats["draws"]
        return self.stats["wins"] / total if total > 0 else 0.0

    def copy_for_play(self, epsilon: float | None = None) -> "MNKQLearningAgent":
        """Return a frozen play-only copy of the current learned weights."""
        clone = MNKQLearningAgent(
            learning_rate=self.alpha,
            discount_factor=self.gamma,
            epsilon=self.epsilon if epsilon is None else epsilon,
            td_clip=self.td_clip,
            weight_clip=self.weight_clip,
            use_tactical_rules=self.use_tactical_rules,
        )
        clone.weights = self.weights.copy()
        clone.stats = self.stats.copy()
        return clone

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _td_update(self, phi: np.ndarray, q: float, reward: float) -> None:
        """Apply a bounded TD update and keep the weight vector finite."""
        if not np.isfinite(q) or not np.isfinite(reward) or not np.all(np.isfinite(phi)):
            self.stats["skipped_updates"] += 1
            return

        delta = reward - q
        if not np.isfinite(delta):
            self.stats["skipped_updates"] += 1
            return

        clipped_delta = float(np.clip(delta, -self.td_clip, self.td_clip))
        if clipped_delta != delta:
            self.stats["clipped_updates"] += 1

        new_weights = self.weights + self.alpha * clipped_delta * phi
        if not np.all(np.isfinite(new_weights)):
            self.stats["skipped_updates"] += 1
            return

        self.weights = np.clip(new_weights, -self.weight_clip, self.weight_clip)

    def is_healthy(self) -> bool:
        return bool(np.all(np.isfinite(self.weights)))

    def _should_use_tactical_rule(self) -> bool:
        """Apply tactical guards with probability 1 - epsilon.

        This keeps the same epsilon-based difficulty scale for immediate wins,
        immediate blocks, and open-threat attack/defense.
        """
        return self.use_tactical_rules and self._rng.random() >= self.epsilon

    def _best_q_move(
        self,
        board: list[list[int]],
        moves: list[tuple[int, int]],
        marker: int,
        enemy_marker: int,
        k: int,
    ) -> tuple[int, int]:
        return max(moves, key=lambda m: self.q_value(board, m, marker, enemy_marker, k))

    def _winning_moves(
        self,
        board: list[list[int]],
        marker: int,
        k: int,
        moves: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        winners = []
        m = len(board)
        n = len(board[0])
        for move in moves:
            row, col = move
            board[row][col] = marker
            try:
                if _check_winner(board, m, n, k, move, marker):
                    winners.append(move)
            finally:
                board[row][col] = 0
        return winners

    def _moves_that_prevent_immediate_loss(
        self,
        board: list[list[int]],
        moves: list[tuple[int, int]],
        marker: int,
        enemy_marker: int,
        k: int,
    ) -> list[tuple[int, int]]:
        safe_moves = []
        for move in moves:
            row, col = move
            board[row][col] = marker
            try:
                enemy_replies = legal_moves(board)
                if not self._winning_moves(board, enemy_marker, k, enemy_replies):
                    safe_moves.append(move)
            finally:
                board[row][col] = 0
        return safe_moves

    def _open_threat_blocks(
        self,
        board: list[list[int]],
        marker: int,
        k: int,
        min_stones: int,
    ) -> list[tuple[int, int]]:
        """Return endpoints that block open runs of at least min_stones.

        For Gomoku k=5, min_stones=3 captures open-three. This is intentionally
        an inference-time safety rule, not a learned feature.
        """
        height = len(board)
        width = len(board[0])
        blocks = set()
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for row in range(height):
            for col in range(width):
                if board[row][col] != marker:
                    continue

                for dr, dc in directions:
                    prev_r = row - dr
                    prev_c = col - dc
                    if (
                        0 <= prev_r < height
                        and 0 <= prev_c < width
                        and board[prev_r][prev_c] == marker
                    ):
                        continue

                    run = 0
                    r = row
                    c = col
                    while 0 <= r < height and 0 <= c < width and board[r][c] == marker:
                        run += 1
                        r += dr
                        c += dc

                    if run < min_stones or run >= k:
                        continue

                    before = (row - dr, col - dc)
                    after = (r, c)
                    before_open = (
                        0 <= before[0] < height
                        and 0 <= before[1] < width
                        and board[before[0]][before[1]] == 0
                    )
                    after_open = (
                        0 <= after[0] < height
                        and 0 <= after[1] < width
                        and board[after[0]][after[1]] == 0
                    )

                    if before_open and after_open:
                        blocks.add(before)
                        blocks.add(after)

        return sorted(blocks)


# ---------------------------------------------------------------------------
# Epsilon-based difficulty pool
# ---------------------------------------------------------------------------

# Epsilon values for each difficulty level (index 0 = level 1 = easiest).
# Level 1 plays almost randomly; level 5 always picks the greedy best move.
EPSILON_LEVELS = [1.0, 0.7, 0.4, 0.15, 0.0]


def build_opponent_pool(
    path: str,
    num_levels: int = 5,
    epsilon_levels: list[float] | None = None,
    use_tactical_rules: bool | None = None,
) -> list[MNKQLearningAgent]:
    """
    Load one trained agent and return `num_levels` copies with decreasing
    epsilon values.

    Difficulty is controlled entirely through epsilon rather than training
    snapshots, which guarantees a monotonic gradient regardless of when the
    weights converged during training.

    Args:
        path:          Path to the single trained .pkl file.
        num_levels:    Number of difficulty levels to create.
        epsilon_levels: Override the default EPSILON_LEVELS list.

    Returns:
        List of agents ordered from weakest (high ε) to strongest (ε=0).
    """
    if epsilon_levels is None:
        if num_levels == len(EPSILON_LEVELS):
            epsilon_levels = EPSILON_LEVELS
        else:
            # Evenly space from 1.0 down to 0.0
            step = 1.0 / (num_levels - 1) if num_levels > 1 else 1.0
            epsilon_levels = [round(1.0 - i * step, 4) for i in range(num_levels)]

    pool = []
    for eps in epsilon_levels:
        agent = MNKQLearningAgent()
        agent.load(path)
        agent.epsilon = eps
        if use_tactical_rules is not None:
            agent.use_tactical_rules = use_tactical_rules
        pool.append(agent)
    return pool


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_mnk_agent(
    n: int = 3,
    k: int = 3,
    num_episodes: int = 30_000,
    save_dir: str = "agents",
    name: str = "mnk",
    use_tactical_rules: bool = True,
    opponent_strategy: str = "mixed",
) -> MNKQLearningAgent:
    """
    Train one agent to convergence and save it as a single .pkl file.

    Epsilon decays linearly from 0.3 → 0.05 over training. Difficulty
    levels are created at inference time via build_opponent_pool(), not
    by saving multiple snapshots.

    Returns:
        The trained agent.
    """
    game = MNKGame(n, n, k)
    agent = MNKQLearningAgent(
        learning_rate=0.01,
        epsilon=0.3,
        use_tactical_rules=use_tactical_rules,
    )

    print(f"Training MNK({n},{n},{k}) for {num_episodes:,} episodes")
    print(f"Tactical rules: {'on' if use_tactical_rules else 'off'}")
    print(f"Opponent strategy: {opponent_strategy}")
    print("-" * 60)

    random_opponent = RandomPolicy()
    opponent_snapshots: list[MNKQLearningAgent] = []
    log_interval = max(num_episodes // 5, 1)
    for episode in range(num_episodes):
        agent.epsilon = 0.3 - 0.25 * (episode / num_episodes)
        progress = episode / max(num_episodes, 1)
        opponent = _training_opponent(
            agent=agent,
            random_opponent=random_opponent,
            snapshots=opponent_snapshots,
            progress=progress,
            strategy=opponent_strategy,
        )
        agent.train_episode(game, opponent=opponent)

        if (episode + 1) % log_interval == 0:
            opponent_snapshots.append(agent.copy_for_play(epsilon=0.05))
            opponent_snapshots = opponent_snapshots[-4:]
            print(f"  Episode {episode + 1:,} / {num_episodes:,} | "
                  f"Win rate: {agent.win_rate():.1%} | "
                  f"Clipped: {agent.stats['clipped_updates']:,} | "
                  f"Skipped: {agent.stats['skipped_updates']:,}")

    path = os.path.join(save_dir, f"{name}_{n}x{n}_k{k}_trained.pkl")
    agent.save(path)
    print("-" * 60)
    print(f"Done. Saved to {path}")
    return agent


def _training_opponent(
    agent: MNKQLearningAgent,
    random_opponent: RandomPolicy,
    snapshots: list[MNKQLearningAgent],
    progress: float,
    strategy: str,
):
    """Choose a training opponent for curriculum self-play.

    random:
        Always train against RandomPolicy.
    self:
        Train against a frozen copy of the current agent.
    mixed:
        Start with RandomPolicy, then mix random, current self-play, and older
        snapshots so the agent sees both weak and increasingly competent play.
    """
    if strategy == "random":
        return random_opponent

    if strategy == "self":
        return agent.copy_for_play(epsilon=max(0.05, agent.epsilon))

    if strategy != "mixed":
        raise ValueError(f"unknown opponent_strategy: {strategy}")

    if progress < 0.20:
        return random_opponent

    roll = agent._rng.random()
    if roll < 0.20:
        return random_opponent
    if roll < 0.65 or not snapshots:
        return agent.copy_for_play(epsilon=max(0.05, agent.epsilon))
    return agent._rng.choice(snapshots)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    game = MNKGame(3, 3, 3)
    agent = MNKQLearningAgent(learning_rate=0.05, epsilon=0.3, seed=1)

    for _ in range(5_000):
        agent.train_episode(game)

    agent.epsilon = 0.0
    rand = RandomPolicy()
    wins = draws = losses = 0

    for i in range(500):
        board = [[0] * 3 for _ in range(3)]
        current = PLAYER
        agent_marker = PLAYER if i % 2 == 0 else OPPONENT
        enemy_marker = -agent_marker
        while True:
            moves = legal_moves(board)
            if not moves:
                draws += 1
                break
            if current == agent_marker:
                move = agent.choose_move(board, agent_marker, enemy_marker, 3)
            else:
                move = rand.choose_move(board, current, -current, 3)
            board[move[0]][move[1]] = current
            if _check_winner(board, 3, 3, 3, move, current):
                if current == agent_marker:
                    wins += 1
                else:
                    losses += 1
                break
            current = -current

    total = wins + draws + losses
    print(f"After 5,000 training episodes (linear FA):")
    print(f"  Win rate vs random: {wins/total:.1%} "
          f"(wins={wins}, draws={draws}, losses={losses})")
    assert wins / total > 0.35, f"Win rate too low: {wins/total:.1%}"

    board = [[0] * 3 for _ in range(3)]
    val = agent.q_value(board, (1, 1), PLAYER, OPPONENT, 3)
    assert isinstance(val, float)
    print(f"  q_value((1,1) on empty board): {val:.4f}")
    print("All checks passed.")
