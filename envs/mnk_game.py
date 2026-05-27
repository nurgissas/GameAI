"""
MNK Game environment — generalized m×n board, k-in-a-row to win.
Covers Tic-Tac-Toe (3,3,3) and Gomoku (15,15,5).

TODO: implement MNKGame, GameResult, and legal_moves fully.
      difficulty_scaling.py depends on this module.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

PLAYER = 1
OPPONENT = -1


@dataclass
class GameResult:
    winner: int                         # PLAYER, OPPONENT, or 0 (draw)
    moves: int                          # total moves played
    normalized_duration: float          # moves / board_cells
    estimated_player_difficulty: float  # set by QValueDifficultyEstimator
    player_move_quality: float          # set by QValueDifficultyEstimator


def legal_moves(board: List[List[int]]) -> List[Tuple[int, int]]:
    """Return all empty cells as (row, col) tuples."""
    raise NotImplementedError("legal_moves not implemented yet")


class MNKGame:
    """
    Generalized m×n board game where k consecutive marks win.

    Args:
        m: board rows
        n: board columns
        k: marks in a row needed to win
    """

    def __init__(self, m: int, n: int, k: int):
        self.m = m
        self.n = n
        self.k = k

    def run_round(
        self,
        player_policy,
        opponent_policy,
        player_first: bool = True,
        player_move_observer: Optional[Callable] = None,
    ) -> GameResult:
        """
        Play one full game between player_policy and opponent_policy.

        Args:
            player_policy:        policy with a choose_move(board, marker) method
            opponent_policy:      same interface
            player_first:         True if PLAYER moves first
            player_move_observer: callback(board, move, marker, enemy, k) called
                                  after each player move — used by
                                  QValueDifficultyEstimator

        Returns:
            GameResult
        """
        raise NotImplementedError("MNKGame.run_round not implemented yet")
