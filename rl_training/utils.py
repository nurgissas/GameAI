"""Utility functions for training"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from envs.tictactoe import TicTacToeGame


def play_game(agent1, agent2, game, training=True):
    """
    Play one game between two agents

    Returns:
        tuple: (winner, last_agent, last_action, last_state)
    """
    game.reset()

    while True:
        # Agent 1 (X) move
        state = game.get_state()
        valid_moves = game.get_valid_moves()
        action = agent1.choose_action(state, valid_moves, training=training)
        game.make_move(action)

        winner = game.get_winner()
        if winner:
            return winner, agent1, action, state

        # Agent 2 (O) move
        state = game.get_state()
        valid_moves = game.get_valid_moves()
        action = agent2.choose_action(state, valid_moves, training=training)
        game.make_move(action)

        winner = game.get_winner()
        if winner:
            return winner, agent2, action, state


def train_one_game(agent, game, opponent=None):
    """
    Train agent through one game

    Returns:
        tuple: (winner, reward_for_agent)
    """
    if opponent is None:
        from rl_training.q_learning_agent import QLearningAgent
        opponent = QLearningAgent()

    winner, last_agent, last_action, last_state = play_game(
        agent, opponent, game, training=True
    )

    if winner == 'X':
        reward = 1.0 if last_agent == agent else -1.0
    elif winner == 'Draw':
        reward = 0.0
    else:
        reward = -1.0 if last_agent == agent else 1.0

    agent.update_q_value(last_state, last_action, reward, game.get_state(), [])

    return winner, reward
