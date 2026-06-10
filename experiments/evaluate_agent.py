"""Evaluate each difficulty level of the opponent pool against a random opponent."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mnk_game import OPPONENT, PLAYER, RandomPolicy, _check_winner, legal_moves
from rl_training.mnk_q_agent import EPSILON_LEVELS, build_opponent_pool


def evaluate_agent(agent, n: int, k: int, num_games: int = 100) -> float:
    """Return win rate of agent vs a random opponent over num_games."""
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
        description="Evaluate all difficulty levels vs random opponent."
    )
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--levels", type=int, default=5)
    args = parser.parse_args()

    path = f"agents/mnk_{args.n}x{args.n}_k{args.k}_trained.pkl"
    if not os.path.exists(path):
        print(f"Model not found: {path}")
        print("Run: python rl_training/train_base_agent.py first")
        sys.exit(1)

    pool = build_opponent_pool(path, num_levels=args.levels)

    print(f"Evaluating MNK({args.n},{args.n},{args.k}) — {args.games} games per level")
    print(f"{'Level':<8} {'Epsilon':<10} {'Win Rate'}")
    print("-" * 35)
    for level, agent in enumerate(pool, start=1):
        win_rate = evaluate_agent(agent, n=args.n, k=args.k, num_games=args.games)
        print(f"  {level:<6} ε={agent.epsilon:<8.2f} {win_rate:.1%}")
