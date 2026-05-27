"""Train base Q-Learning agent from scratch"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from envs.tictactoe import TicTacToeGame
from rl_training.q_learning_agent import QLearningAgent
from rl_training.utils import play_game


def train_agent(num_episodes=10000, save_interval=2000, save_path_template='agents/tictactoe_level_{}.pkl'):
    """
    Train Q-Learning agent and save snapshots at intervals

    Args:
        num_episodes: Total games to play
        save_interval: Save every N episodes
        save_path_template: Path template for saving levels
    """
    agent = QLearningAgent(learning_rate=0.1, epsilon=0.3)
    game = TicTacToeGame()

    print(f"Starting training for {num_episodes} episodes...")
    print("-" * 50)

    for episode in range(num_episodes):
        game.reset()
        agent.epsilon = 0.3 if episode < num_episodes // 2 else 0.1

        while True:
            # Agent (X) move
            state = game.get_state()
            valid_moves = game.get_valid_moves()
            action = agent.choose_action(state, valid_moves, training=True)
            game.make_move(action)

            winner = game.get_winner()
            if winner:
                if winner == 'X':
                    reward = 1.0
                elif winner == 'Draw':
                    reward = 0.0
                else:
                    reward = -1.0

                next_state = game.get_state()
                agent.update_q_value(state, action, reward, next_state, [])

                if winner == 'X':
                    agent.training_stats['wins'] += 1
                elif winner == 'Draw':
                    agent.training_stats['draws'] += 1
                else:
                    agent.training_stats['losses'] += 1
                agent.training_stats['episodes'] += 1
                break

            # Random opponent (O)
            state = game.get_state()
            valid_moves = game.get_valid_moves()
            action = random.choice(valid_moves)
            game.make_move(action)

            winner = game.get_winner()
            if winner:
                break

        if (episode + 1) % save_interval == 0:
            level = (episode + 1) // save_interval
            save_path = save_path_template.format(level)
            agent.save_agent(save_path)

            stats = agent.get_stats()
            total = stats['total_games']
            win_rate = agent.training_stats['wins'] / total * 100 if total > 0 else 0
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"States: {stats['states_learned']} | "
                  f"Win rate: {win_rate:.1f}%")

    print("-" * 50)
    print("Training complete!")
    return agent


if __name__ == "__main__":
    train_agent(num_episodes=50000, save_interval=10000)
