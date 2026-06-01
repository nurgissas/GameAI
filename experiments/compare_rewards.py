"""Compare the effect of different terminal reward formulations on agent training."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mnk_game import OPPONENT, PLAYER, RandomPolicy, _check_winner, legal_moves
from rl_training.mnk_q_agent import MNKQLearningAgent
from envs.mnk_game import MNKGame

REWARD_SCHEMES = {
    "sparse":       {"win":  1.0, "loss": -1.0, "draw":  0.0},
    "draw_penalty": {"win":  1.0, "loss": -1.0, "draw": -0.5},
    "draw_reward":  {"win":  1.0, "loss": -1.0, "draw":  0.5},
    "asymmetric":   {"win":  2.0, "loss": -1.0, "draw":  0.0},
}


def train_with_scheme(scheme_name: str, n: int = 3, k: int = 3, num_episodes: int = 20_000):
    game = MNKGame(n, n, k)
    agent = MNKQLearningAgent(learning_rate=0.05, epsilon=0.3)
    rewards = REWARD_SCHEMES[scheme_name]

    for episode in range(num_episodes):
        progress = episode / num_episodes
        agent.epsilon = 0.3 - 0.2 * progress
        agent.train_episode(game, rewards=rewards)

    return agent


def evaluate(agent: MNKQLearningAgent, n: int = 3, k: int = 3, num_games: int = 200) -> dict:
    agent.epsilon = 0.0
    rand = RandomPolicy()
    wins = draws = 0

    for i in range(num_games):
        board = [[0] * n for _ in range(n)]
        agent_marker = PLAYER if i % 2 == 0 else OPPONENT
        enemy_marker = -agent_marker
        current = PLAYER

        while True:
            moves = legal_moves(board)
            if not moves:
                draws += 1
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

    return {"win_rate": wins / num_games, "draw_rate": draws / num_games}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare reward formulations for MNK agent training."
    )
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=20_000)
    args = parser.parse_args()

    print(f"Comparing reward formulations on MNK({args.n},{args.n},{args.k}) "
          f"— {args.episodes:,} episodes each")
    print(f"{'Scheme':<20} {'Win Rate':>10} {'Draw Rate':>10}")
    print("-" * 45)

    for name in REWARD_SCHEMES:
        agent = train_with_scheme(name, n=args.n, k=args.k, num_episodes=args.episodes)
        result = evaluate(agent, n=args.n, k=args.k)
        print(f"{name:<20} {result['win_rate']:>9.1%} {result['draw_rate']:>9.1%}")
