"""Train a single MNK agent to convergence for a given board size and win condition.

Difficulty levels are created at inference time by build_opponent_pool(),
which loads this model and returns copies with varying epsilon values.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_training.mnk_q_agent import train_mnk_agent


def main():
    parser = argparse.ArgumentParser(
        description="Train a single MNK agent (difficulty levels come from epsilon, not snapshots)."
    )
    parser.add_argument("--n", type=int, default=3,
                        help="Board size n×n (default: 3)")
    parser.add_argument("--k", type=int, default=3,
                        help="Win condition: k pieces in a row (default: 3)")
    parser.add_argument("--episodes", type=int, default=30_000,
                        help="Training episodes (default: 30000)")
    parser.add_argument("--save-dir", default="agents",
                        help="Directory for the saved model (default: agents)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    train_mnk_agent(
        n=args.n,
        k=args.k,
        num_episodes=args.episodes,
        save_dir=args.save_dir,
        name="mnk",
    )


if __name__ == "__main__":
    main()
