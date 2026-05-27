"""Hyperparameter sensitivity analysis for Q-Learning agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from envs.tictactoe import TicTacToeGame
from rl_training.q_learning_agent import QLearningAgent


def train_and_evaluate(lr, gamma, epsilon, num_episodes=15000, eval_games=200):
    agent = QLearningAgent(learning_rate=lr, discount_factor=gamma, epsilon=epsilon)
    game = TicTacToeGame()

    for episode in range(num_episodes):
        game.reset()

        while True:
            state = game.get_state()
            valid_moves = game.get_valid_moves()
            action = agent.choose_action(state, valid_moves, training=True)
            game.make_move(action)

            winner = game.get_winner()
            if winner:
                if winner == 'X':
                    reward, key = 1.0, 'wins'
                elif winner == 'Draw':
                    reward, key = 0.0, 'draws'
                else:
                    reward, key = -1.0, 'losses'

                agent.update_q_value(state, action, reward, game.get_state(), [])
                agent.training_stats[key] += 1
                agent.training_stats['episodes'] += 1
                break

            valid_moves = game.get_valid_moves()
            game.make_move(random.choice(valid_moves))
            if game.get_winner():
                break

    # Evaluate
    agent.epsilon = 0.0
    wins = 0
    for _ in range(eval_games):
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
                break
            valid_moves = game.get_valid_moves()
            if valid_moves:
                game.make_move(random.choice(valid_moves))
            if game.get_winner():
                break

    return wins / eval_games


if __name__ == "__main__":
    learning_rates = [0.05, 0.1, 0.2, 0.5]
    discount_factors = [0.9, 0.95, 0.99]
    epsilons = [0.1, 0.2, 0.3, 0.5]

    print("Sensitivity Analysis: Learning Rate")
    print("-" * 40)
    for lr in learning_rates:
        win_rate = train_and_evaluate(lr=lr, gamma=0.99, epsilon=0.3)
        print(f"  lr={lr:.2f}: {win_rate:.1%} win rate")

    print("\nSensitivity Analysis: Discount Factor")
    print("-" * 40)
    for gamma in discount_factors:
        win_rate = train_and_evaluate(lr=0.1, gamma=gamma, epsilon=0.3)
        print(f"  gamma={gamma}: {win_rate:.1%} win rate")

    print("\nSensitivity Analysis: Epsilon")
    print("-" * 40)
    for eps in epsilons:
        win_rate = train_and_evaluate(lr=0.1, gamma=0.99, epsilon=eps)
        print(f"  epsilon={eps}: {win_rate:.1%} win rate")
