"""Compare different reward formulations and their effect on training"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from envs.tictactoe import TicTacToeGame
from rl_training.q_learning_agent import QLearningAgent


REWARD_SCHEMES = {
    'sparse': {'win': 1.0, 'loss': -1.0, 'draw': 0.0},
    'draw_penalty': {'win': 1.0, 'loss': -1.0, 'draw': -0.5},
    'draw_reward': {'win': 1.0, 'loss': -1.0, 'draw': 0.5},
    'asymmetric': {'win': 2.0, 'loss': -1.0, 'draw': 0.0},
}


def train_with_scheme(scheme_name, num_episodes=20000):
    rewards = REWARD_SCHEMES[scheme_name]
    agent = QLearningAgent(learning_rate=0.1, epsilon=0.3)
    game = TicTacToeGame()

    for episode in range(num_episodes):
        game.reset()
        agent.epsilon = 0.3 if episode < num_episodes // 2 else 0.1

        while True:
            state = game.get_state()
            valid_moves = game.get_valid_moves()
            action = agent.choose_action(state, valid_moves, training=True)
            game.make_move(action)

            winner = game.get_winner()
            if winner:
                if winner == 'X':
                    reward = rewards['win']
                    agent.training_stats['wins'] += 1
                elif winner == 'Draw':
                    reward = rewards['draw']
                    agent.training_stats['draws'] += 1
                else:
                    reward = rewards['loss']
                    agent.training_stats['losses'] += 1

                agent.update_q_value(state, action, reward, game.get_state(), [])
                agent.training_stats['episodes'] += 1
                break

            # Random opponent
            valid_moves = game.get_valid_moves()
            game.make_move(random.choice(valid_moves))

            winner = game.get_winner()
            if winner:
                break

    return agent


def evaluate(agent, num_games=200):
    game = TicTacToeGame()
    agent.epsilon = 0.0
    wins = draws = 0

    for _ in range(num_games):
        game.reset()
        while True:
            state = game.get_state()
            valid_moves = game.get_valid_moves()
            action = agent.choose_action(state, valid_moves, training=False)
            game.make_move(action)

            winner = game.get_winner()
            if winner:
                if winner == 'X':
                    wins += 1
                elif winner == 'Draw':
                    draws += 1
                break

            valid_moves = game.get_valid_moves()
            if valid_moves:
                game.make_move(random.choice(valid_moves))
            winner = game.get_winner()
            if winner:
                if winner == 'Draw':
                    draws += 1
                break

    return {'win_rate': wins / num_games, 'draw_rate': draws / num_games}


if __name__ == "__main__":
    print("Comparing reward formulations (20,000 episodes each)...")
    print("=" * 55)
    print(f"{'Scheme':<20} {'Win Rate':>10} {'Draw Rate':>10} {'States':>10}")
    print("-" * 55)

    for scheme_name in REWARD_SCHEMES:
        agent = train_with_scheme(scheme_name, num_episodes=20000)
        result = evaluate(agent)
        states = len(agent.q_table)
        print(f"{scheme_name:<20} {result['win_rate']:>9.1%} {result['draw_rate']:>9.1%} {states:>10}")

    print("=" * 55)
