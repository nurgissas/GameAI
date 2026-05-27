"""Evaluate trained agent performance against a random opponent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from envs.tictactoe import TicTacToeGame
from rl_training.q_learning_agent import QLearningAgent


def evaluate_agent(agent_path, num_games=100):
    """
    Test agent against random opponent

    Returns:
        float: win rate
    """
    agent = QLearningAgent()
    agent.load_agent(agent_path)
    agent.epsilon = 0.0  # No exploration during evaluation

    game = TicTacToeGame()
    wins = 0

    for _ in range(num_games):
        game.reset()

        while True:
            # Agent (X) move
            state = game.get_state()
            valid_moves = game.get_valid_moves()
            action = agent.choose_action(state, valid_moves, training=False)
            game.make_move(action)

            winner = game.get_winner()
            if winner:
                if winner == 'X':
                    wins += 1
                break

            # Random opponent (O)
            state = game.get_state()
            valid_moves = game.get_valid_moves()
            action = random.choice(valid_moves)
            game.make_move(action)

            winner = game.get_winner()
            if winner:
                break

    return wins / num_games


if __name__ == "__main__":
    print("Evaluating difficulty levels...")

    for level in range(1, 6):
        agent_path = f'agents/tictactoe_level_{level}.pkl'
        try:
            win_rate = evaluate_agent(agent_path, num_games=50)
            print(f"Level {level}: {win_rate:.1%} win rate")
        except FileNotFoundError:
            print(f"Level {level}: Not found (run train_base_agent.py first)")
