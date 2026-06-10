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
        learning_rate: float = 0.05,
        discount_factor: float = 0.99,
        epsilon: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self.weights = np.zeros(N_FEATURES, dtype=np.float64)
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self.stats = {"wins": 0, "losses": 0, "draws": 0, "episodes": 0}

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
        return float(np.dot(self.weights, extract(board, move, marker, k)))

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

        if self._rng.random() < self.epsilon:
            return self._rng.choice(moves)

        return max(moves, key=lambda m: np.dot(self.weights, extract(board, m, marker, k)))

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
                        np.dot(self.weights, extract(board, m, agent_marker, game.k))
                        for m in moves
                    )
                    delta = self.gamma * float(max_next_q) - prev_q
                    self.weights += self.alpha * delta * prev_phi

                move = self.choose_move(board, agent_marker, enemy_marker, game.k)
                phi = extract(board, move, agent_marker, game.k)
                prev_q = float(np.dot(self.weights, phi))
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
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"weights": self.weights, "stats": self.stats}, f)
        print(f"Saved -> {path}  (non-zero weights: {np.count_nonzero(self.weights)})")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.weights = data["weights"]
        self.stats = data.get(
            "stats", {"wins": 0, "losses": 0, "draws": 0, "episodes": 0}
        )
        print(f"Loaded <- {path}")

    def win_rate(self) -> float:
        total = self.stats["wins"] + self.stats["losses"] + self.stats["draws"]
        return self.stats["wins"] / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _td_update(self, phi: np.ndarray, q: float, reward: float) -> None:
        """Terminal update — no next state, so max Q(s') = 0."""
        delta = reward - q
        self.weights += self.alpha * delta * phi


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
    agent = MNKQLearningAgent(learning_rate=0.05, epsilon=0.3)

    print(f"Training MNK({n},{n},{k}) for {num_episodes:,} episodes")
    print("-" * 60)

    log_interval = max(num_episodes // 5, 1)
    for episode in range(num_episodes):
        agent.epsilon = 0.3 - 0.25 * (episode / num_episodes)
        agent.train_episode(game)

        if (episode + 1) % log_interval == 0:
            print(f"  Episode {episode + 1:,} / {num_episodes:,} | "
                  f"Win rate: {agent.win_rate():.1%}")

    path = os.path.join(save_dir, f"{name}_{n}x{n}_k{k}_trained.pkl")
    agent.save(path)
    print("-" * 60)
    print(f"Done. Saved to {path}")
    return agent


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
