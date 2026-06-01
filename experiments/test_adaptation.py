"""End-to-end test of the rule-based adaptive difficulty system.

A random-move "player" faces pre-trained opponents. The MetaAgent watches
win/loss history and adjusts which difficulty level is used each game.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mnk_game import MNKGame, OPPONENT, PLAYER, RandomPolicy, _check_winner, legal_moves
from rl_training.mnk_q_agent import MNKQLearningAgent
from rl_training.meta_agent import MetaAgent


def test_adaptive_gameplay(n: int = 3, k: int = 3, num_games: int = 50) -> dict:
    """
    Simulate adaptive gameplay between a random "player" and a trained opponent pool.

    Loads 5 pre-trained agents from agents/mnk_{n}x{n}_k{k}_level_*.pkl.
    """
    agents = []
    for level in range(1, 6):
        path = f"agents/mnk_{n}x{n}_k{k}_level_{level}.pkl"
        agent = MNKQLearningAgent()
        agent.load(path)
        agent.epsilon = 0.0
        agents.append(agent)

    meta = MetaAgent(num_levels=5, target_win_rate=0.5)
    player = RandomPolicy()

    stats: dict = {
        "player_wins": 0,
        "player_losses": 0,
        "player_draws": 0,
        "difficulty_levels_used": [],
    }

    for _ in range(num_games):
        level = meta.select_difficulty()
        stats["difficulty_levels_used"].append(level + 1)
        opponent = agents[level]

        board = [[0] * n for _ in range(n)]
        current = PLAYER
        result = "draw"

        while True:
            moves = legal_moves(board)
            if not moves:
                break
            if current == PLAYER:
                move = player.choose_move(board, PLAYER, OPPONENT, k)
            else:
                move = opponent.choose_move(board, OPPONENT, PLAYER, k)
            board[move[0]][move[1]] = current
            if _check_winner(board, n, n, k, move, current):
                result = "player_win" if current == PLAYER else "opponent_win"
                break
            current = -current

        if result == "player_win":
            meta.update_performance(True)
            stats["player_wins"] += 1
        elif result == "opponent_win":
            meta.update_performance(False)
            stats["player_losses"] += 1
        else:
            meta.update_performance(False)
            stats["player_draws"] += 1

    total = stats["player_wins"] + stats["player_losses"] + stats["player_draws"]
    stats["player_win_rate"] = stats["player_wins"] / total if total else 0.0
    stats["avg_difficulty"] = (
        sum(stats["difficulty_levels_used"]) / len(stats["difficulty_levels_used"])
    )
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test rule-based adaptive difficulty on MNK."
    )
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--games", type=int, default=50)
    args = parser.parse_args()

    print(f"Testing adaptive difficulty on MNK({args.n},{args.n},{args.k})...")
    stats = test_adaptive_gameplay(n=args.n, k=args.k, num_games=args.games)

    total = stats["player_wins"] + stats["player_losses"] + stats["player_draws"]
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Total games     : {total}")
    print(f"Player wins     : {stats['player_wins']} ({stats['player_win_rate']:.1%})")
    print(f"Player losses   : {stats['player_losses']}")
    print(f"Draws           : {stats['player_draws']}")
    print(f"Avg difficulty  : {stats['avg_difficulty']:.1f} / 5")
    print(f"Levels used     : {sorted(set(stats['difficulty_levels_used']))}")
    print("=" * 50)
