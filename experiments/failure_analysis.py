"""Identify and analyze failure modes of trained agents"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from envs.tictactoe import TicTacToeGame
from rl_training.q_learning_agent import QLearningAgent


def collect_failure_cases(agent_path, num_games=500):
    """
    Play games and collect cases where the agent lost or missed a win.

    Returns:
        dict: failure categories with counts and examples
    """
    agent = QLearningAgent()
    agent.load_agent(agent_path)
    agent.epsilon = 0.0

    game = TicTacToeGame()

    failures = {
        'missed_win': [],       # Agent could have won but didn't
        'missed_block': [],     # Agent didn't block opponent's winning move
        'losses': 0,
        'wins': 0,
        'draws': 0,
    }

    WINNING_COMBOS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]

    def find_winning_move(board, symbol):
        for combo in WINNING_COMBOS:
            a, b, c = combo
            cells = [board[a], board[b], board[c]]
            if cells.count(symbol) == 2 and cells.count(' ') == 1:
                return combo[cells.index(' ')]
        return None

    for _ in range(num_games):
        game.reset()
        move_history = []

        while True:
            board_before = list(game.board)
            state = game.get_state()
            valid_moves = game.get_valid_moves()

            # Check if agent could win
            winning_move = find_winning_move(list(board_before), 'X')
            # Check if agent should block
            blocking_move = find_winning_move(list(board_before), 'O')

            action = agent.choose_action(state, valid_moves, training=False)
            move_history.append((state, action))

            if winning_move is not None and action != winning_move:
                failures['missed_win'].append({
                    'board': board_before,
                    'winning_move': winning_move,
                    'chosen_move': action
                })

            if blocking_move is not None and winning_move is None and action != blocking_move:
                failures['missed_block'].append({
                    'board': board_before,
                    'blocking_move': blocking_move,
                    'chosen_move': action
                })

            game.make_move(action)
            winner = game.get_winner()
            if winner:
                if winner == 'X':
                    failures['wins'] += 1
                elif winner == 'Draw':
                    failures['draws'] += 1
                else:
                    failures['losses'] += 1
                break

            # Random opponent
            valid_moves = game.get_valid_moves()
            if valid_moves:
                game.make_move(random.choice(valid_moves))
            winner = game.get_winner()
            if winner:
                if winner == 'O':
                    failures['losses'] += 1
                elif winner == 'Draw':
                    failures['draws'] += 1
                break

    return failures


def print_report(failures, agent_path):
    total = failures['wins'] + failures['draws'] + failures['losses']
    print(f"\nFailure Analysis Report: {agent_path}")
    print("=" * 50)
    print(f"Games played : {total}")
    print(f"Wins         : {failures['wins']} ({failures['wins']/total:.1%})")
    print(f"Draws        : {failures['draws']} ({failures['draws']/total:.1%})")
    print(f"Losses       : {failures['losses']} ({failures['losses']/total:.1%})")
    print(f"Missed wins  : {len(failures['missed_win'])}")
    print(f"Missed blocks: {len(failures['missed_block'])}")


if __name__ == "__main__":
    for level in range(1, 6):
        path = f'agents/tictactoe_level_{level}.pkl'
        try:
            failures = collect_failure_cases(path, num_games=300)
            print_report(failures, path)
        except FileNotFoundError:
            print(f"Level {level}: Not found (run train_base_agent.py first)")
