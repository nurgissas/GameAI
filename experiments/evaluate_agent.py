"""Evaluate each trained difficulty level against a random opponent."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mnk_game import OPPONENT, PLAYER, RandomPolicy, _check_winner, legal_moves
from rl_training.mnk_q_agent import MNKQLearningAgent


def evaluate_agent(agent_path: str, n: int = 3, k: int = 3, num_games: int = 100) -> float:
    """Return win rate of the agent at agent_path vs a random opponent."""
    agent = MNKQLearningAgent()
    agent.load(agent_path)
    agent.epsilon = 0.0

    rand = RandomPolicy()
    wins = 0

    for i in range(num_games):
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

    return wins / num_games


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained MNK difficulty levels vs random opponent."
    )
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--levels", type=int, default=5)
    args = parser.parse_args()

    print(f"Evaluating MNK({args.n},{args.n},{args.k}) — {args.games} games per level")
    print("-" * 50)

    for level in range(1, args.levels + 1):
        path = f"agents/mnk_{args.n}x{args.n}_k{args.k}_level_{level}.pkl"
        try:
            win_rate = evaluate_agent(path, n=args.n, k=args.k, num_games=args.games)
            print(f"Level {level}: {win_rate:.1%} win rate")
        except FileNotFoundError:
            print(f"Level {level}: Not found — run train_base_agent.py first")
