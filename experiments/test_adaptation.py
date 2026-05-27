"""Test adaptive difficulty system end-to-end"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from envs.tictactoe import TicTacToeGame
from rl_training.q_learning_agent import QLearningAgent
from rl_training.meta_agent import MetaAgent


def test_adaptive_gameplay(num_games=50, player_skill=0.6):
    """
    Simulate adaptive gameplay with a player of given skill level

    Args:
        num_games: Number of games to play
        player_skill: Probability of making the optimal random move (0.0-1.0)

    Returns:
        dict: Game statistics
    """
    agents = []
    for level in range(1, 6):
        agent = QLearningAgent()
        agent.load_agent(f'agents/tictactoe_level_{level}.pkl')
        agent.epsilon = 0.0
        agents.append(agent)

    meta_agent = MetaAgent(num_levels=5, target_win_rate=0.5)
    game = TicTacToeGame()

    stats = {
        'player_wins': 0,
        'player_losses': 0,
        'player_draws': 0,
        'difficulty_levels_used': [],
        'avg_difficulty': 0
    }

    for _ in range(num_games):
        level = meta_agent.select_difficulty()
        stats['difficulty_levels_used'].append(level + 1)

        game.reset()
        opponent = agents[level]

        while True:
            # Simulated player move (X)
            state = game.get_state()
            valid_moves = game.get_valid_moves()
            action = random.choice(valid_moves) if valid_moves else None

            if action is None:
                break

            game.make_move(action)
            winner = game.get_winner()

            if winner:
                if winner == 'X':
                    meta_agent.update_performance(True)
                    stats['player_wins'] += 1
                elif winner == 'Draw':
                    meta_agent.update_performance(False)
                    stats['player_draws'] += 1
                else:
                    meta_agent.update_performance(False)
                    stats['player_losses'] += 1
                break

            # Opponent move (O)
            state = game.get_state()
            valid_moves = game.get_valid_moves()
            action = opponent.choose_action(state, valid_moves, training=False)
            game.make_move(action)

            winner = game.get_winner()
            if winner:
                if winner == 'O':
                    meta_agent.update_performance(False)
                    stats['player_losses'] += 1
                elif winner == 'Draw':
                    meta_agent.update_performance(False)
                    stats['player_draws'] += 1
                else:
                    meta_agent.update_performance(True)
                    stats['player_wins'] += 1
                break

    stats['avg_difficulty'] = sum(stats['difficulty_levels_used']) / len(stats['difficulty_levels_used'])
    total_games = stats['player_wins'] + stats['player_losses'] + stats['player_draws']
    stats['player_win_rate'] = stats['player_wins'] / total_games if total_games > 0 else 0

    return stats


if __name__ == "__main__":
    print("Testing adaptive difficulty system...")
    stats = test_adaptive_gameplay(num_games=50, player_skill=0.6)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    total = stats['player_wins'] + stats['player_losses'] + stats['player_draws']
    print(f"Total games: {total}")
    print(f"Player wins: {stats['player_wins']} ({stats['player_win_rate']:.1%})")
    print(f"Player losses: {stats['player_losses']}")
    print(f"Draws: {stats['player_draws']}")
    print(f"Average difficulty: {stats['avg_difficulty']:.1f}/5")
    print(f"Difficulty levels used: {set(stats['difficulty_levels_used'])}")
    print("=" * 50)
