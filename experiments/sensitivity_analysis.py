"""Hyperparameter sensitivity analysis for MNK linear FA agents."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mnk_game import MNKGame, OPPONENT, PLAYER, RandomPolicy, _check_winner, legal_moves
from rl_training.mnk_q_agent import MNKQLearningAgent


def train_and_evaluate(
    lr: float,
    gamma: float,
    epsilon: float,
    n: int = 3,
    k: int = 3,
    num_episodes: int = 15_000,
    eval_games: int = 200,
) -> float:
    game = MNKGame(n, n, k)
    agent = MNKQLearningAgent(learning_rate=lr, discount_factor=gamma, epsilon=epsilon)
    rand = RandomPolicy()

    for _ in range(num_episodes):
        agent.train_episode(game)

    agent.epsilon = 0.0
    wins = 0

    for i in range(eval_games):
        board = [[0] * n for _ in range(n)]
        agent_marker = PLAYER if i % 2 == 0 else OPPONENT
        enemy_marker = -agent_marker
        current = PLAYER
        while True:
            moves = legal_moves(board)
            if not moves:
                break
            if current == agent_marker:
                move = agent.choose_move(board, agent_marker, enemy_marker, k)
            else:
                move = rand.choose_move(board, current, -current, k)
            board[move[0]][move[1]] = current
            if _check_winner(board, n, n, k, move, current):
                if current == agent_marker:
                    wins += 1
                break
            current = -current

    return wins / eval_games


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep hyperparameters for the MNK linear FA agent."
    )
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    n, k = args.n, args.k
    print(f"Sensitivity analysis on MNK({n},{n},{k})")

    print("\nLearning Rate  (gamma=0.99, epsilon=0.3)")
    print("-" * 40)
    for lr in [0.01, 0.05, 0.1, 0.2]:
        wr = train_and_evaluate(lr=lr, gamma=0.99, epsilon=0.3, n=n, k=k)
        print(f"  lr={lr:.2f}: {wr:.1%}")

    print("\nDiscount Factor  (lr=0.05, epsilon=0.3)")
    print("-" * 40)
    for gamma in [0.9, 0.95, 0.99]:
        wr = train_and_evaluate(lr=0.05, gamma=gamma, epsilon=0.3, n=n, k=k)
        print(f"  gamma={gamma}: {wr:.1%}")

    print("\nEpsilon  (lr=0.05, gamma=0.99)")
    print("-" * 40)
    for eps in [0.1, 0.2, 0.3, 0.5]:
        wr = train_and_evaluate(lr=0.05, gamma=0.99, epsilon=eps, n=n, k=k)
        print(f"  epsilon={eps}: {wr:.1%}")
