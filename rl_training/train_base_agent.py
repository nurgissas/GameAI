"""Train the MNK opponent pool for a given board size and win condition."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_training.mnk_q_agent import train_mnk_agents


def main():
    parser = argparse.ArgumentParser(
        description="Train the MNK opponent pool (5 difficulty levels)."
    )
    parser.add_argument("--n", type=int, default=3,
                        help="Board size n×n (default: 3)")
    parser.add_argument("--k", type=int, default=3,
                        help="Win condition: k pieces in a row (default: 3)")
    parser.add_argument("--episodes", type=int, default=50_000,
                        help="Total training episodes (default: 50000)")
    parser.add_argument("--levels", type=int, default=5,
                        help="Number of difficulty levels to save (default: 5)")
    parser.add_argument("--save-dir", default="agents",
                        help="Directory for saved agent files (default: agents)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    train_mnk_agents(
        n=args.n,
        k=args.k,
        num_episodes=args.episodes,
        num_levels=args.levels,
        save_dir=args.save_dir,
        name="mnk",
    )


if __name__ == "__main__":
    main()
