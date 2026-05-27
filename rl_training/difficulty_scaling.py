from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass

from envs.mnk_game import GameResult, MNKGame, OPPONENT, PLAYER, legal_moves


class QValueDifficultyEstimator:
    def __init__(self, policies):
        self.policies = policies
        self.max_level = max(1, len(policies) - 1)

    def estimate_move(
        self,
        board: list[list[int]],
        move: tuple[int, int],
        marker: int,
        enemy_marker: int,
        k: int,
    ) -> dict[str, float]:
        level_scores = []
        legal = legal_moves(board)

        for level, policy in enumerate(self.policies):
            values = [
                policy.q_value(board, candidate, marker, enemy_marker, k)
                for candidate in legal
            ]
            chosen = policy.q_value(board, move, marker, enemy_marker, k)
            lower_or_equal = sum(1 for value in values if value <= chosen)
            percentile = lower_or_equal / len(values)
            level_scores.append((level, percentile))

        quality = sum(score for _, score in level_scores) / len(level_scores)
        strongest_quality = level_scores[-1][1]
        estimated = strongest_quality * self.max_level

        return {
            "difficulty": estimated,
            "quality": quality,
        }


@dataclass(frozen=True)
class RewardConfig:
    target_duration: float = 0.55
    duration_weight: float = 0.70
    duration_variance_weight: float = 0.35
    switch_weight: float = 0.08
    q_match_weight: float = 1.20


class QValueDDAMetrics:
    def __init__(self, window: int, reward_config: RewardConfig, n_levels: int):
        self.window = window
        self.reward_config = reward_config
        self.n_levels = n_levels
        self.estimated_difficulties = deque(maxlen=window)
        self.move_qualities = deque(maxlen=window)
        self.outcomes = deque(maxlen=window)
        self.durations = deque(maxlen=window)
        self.difficulties = deque(maxlen=window)
        self.last_difficulty = 0

    def state(self) -> tuple[int, int, int, int]:
        estimated = self._mean(self.estimated_difficulties, default=self.last_difficulty)
        duration = self._mean(self.durations, default=self.reward_config.target_duration)
        estimate_var = self._variance(self.estimated_difficulties)

        return (
            self._bucket(estimated, [0.75, 1.5, 2.5, 3.25]),
            self._bucket(duration, [0.30, 0.45, 0.65, 0.80]),
            self._bucket(estimate_var, [0.10, 0.35, 0.75]),
            self.last_difficulty,
        )

    def update(self, result: GameResult, difficulty: int) -> dict[str, float]:
        player_score = 1.0 if result.winner == PLAYER else 0.0 if result.winner == OPPONENT else 0.5
        self.outcomes.append(player_score)
        self.estimated_difficulties.append(result.estimated_player_difficulty)
        self.move_qualities.append(result.player_move_quality)
        self.durations.append(result.normalized_duration)
        self.difficulties.append(difficulty)
        self.last_difficulty = difficulty

        estimated = self._mean(self.estimated_difficulties, default=float(difficulty))
        win_rate = self._mean(self.outcomes, default=0.5)
        quality = self._mean(self.move_qualities, default=0.0)
        duration = self._mean(self.durations, default=self.reward_config.target_duration)
        duration_var = self._variance(self.durations)
        estimate_var = self._variance(self.estimated_difficulties)

        switch = 0.0
        if len(self.difficulties) >= 2 and self.difficulties[-1] != self.difficulties[-2]:
            switch = 1.0

        max_level = max(1, self.n_levels - 1)
        difficulty_error = abs(difficulty - estimated) / max_level
        duration_penalty = abs(duration - self.reward_config.target_duration)

        reward = -(
            self.reward_config.q_match_weight * difficulty_error
            + self.reward_config.duration_weight * duration_penalty
            + self.reward_config.duration_variance_weight * estimate_var / max_level
            + self.reward_config.switch_weight * switch
        )

        return {
            "player_score": player_score,
            "win_rate": win_rate,
            "estimated_difficulty": estimated,
            "player_move_quality": quality,
            "difficulty_error": difficulty_error,
            "duration": duration,
            "duration_var": duration_var,
            "estimate_var": estimate_var,
            "switch": switch,
            "reward": reward,
        }

    @staticmethod
    def _mean(values, default: float) -> float:
        return sum(values) / len(values) if values else default

    @staticmethod
    def _variance(values) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / len(values)

    @staticmethod
    def _bucket(value: float, thresholds: list[float]) -> int:
        for idx, threshold in enumerate(thresholds):
            if value < threshold:
                return idx
        return len(thresholds)


class QLearningDDA:
    def __init__(
        self,
        n_levels: int,
        rng: random.Random,
        alpha: float = 0.18,
        gamma: float = 0.90,
        epsilon: float = 0.25,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.04,
    ):
        self.n_levels = n_levels
        self.rng = rng
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = [-1, 0, 1]
        self.q = defaultdict(lambda: [0.0 for _ in self.actions])
        self._last_action_idx = 1

    def select_difficulty(self, state) -> int:
        current_difficulty = state[3]
        valid_actions = [
            idx
            for idx, delta in enumerate(self.actions)
            if 0 <= current_difficulty + delta < self.n_levels
        ]

        if self.rng.random() < self.epsilon:
            self._last_action_idx = self.rng.choice(valid_actions)
            return current_difficulty + self.actions[self._last_action_idx]

        values = self.q[state]
        best = max(values[idx] for idx in valid_actions)
        best_actions = [
            idx
            for idx in valid_actions
            if math.isclose(values[idx], best)
        ]
        self._last_action_idx = self.rng.choice(best_actions)
        return current_difficulty + self.actions[self._last_action_idx]

    def observe(self, state, difficulty, reward, next_state) -> None:
        action_idx = self._last_action_idx
        current = self.q[state][action_idx]
        target = reward + self.gamma * max(self.q[next_state])
        self.q[state][action_idx] = current + self.alpha * (target - current)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def run_online_difficulty_learning_episode(
    game: MNKGame,
    player_policy,
    opponent_pool,
    q_estimator: QValueDifficultyEstimator,
    metrics: QValueDDAMetrics,
    dda_controller: QLearningDDA,
    player_first: bool = True,
) -> dict[str, float]:
    state = metrics.state()
    difficulty = dda_controller.select_difficulty(state)

    result = game.run_round(
        player_policy=player_policy,
        opponent_policy=opponent_pool[difficulty],
        player_first=player_first,
        player_move_observer=q_estimator.estimate_move,
    )

    update = metrics.update(result, difficulty)
    next_state = metrics.state()
    dda_controller.observe(state, difficulty, update["reward"], next_state)

    return {
        "difficulty": difficulty,
        "winner": result.winner,
        "moves": result.moves,
        "estimated_player_difficulty": update["estimated_difficulty"],
        "player_move_quality": update["player_move_quality"],
        "difficulty_error": update["difficulty_error"],
        "reward": update["reward"],
        "next_difficulty_state": next_state,
    }
