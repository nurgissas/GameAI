"""
MNK Game environment — generalized m×n board, k-in-a-row to win.

Tic-Tac-Toe : MNKGame(3, 3, 3)
Gomoku       : MNKGame(15, 15, 5)

Board representation
--------------------
  2-D list of ints: 0 = empty, PLAYER = 1, OPPONENT = -1
  Moves are (row, col) tuples.

Policy interface (duck-typed)
-----------------------------
Any object used as player_policy or opponent_policy must implement:

  choose_move(board, marker, enemy_marker, k) -> (row, col)
  q_value(board, move, marker, enemy_marker, k) -> float

RandomPolicy (defined below) is a minimal baseline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

PLAYER = 1
OPPONENT = -1


@dataclass
class GameResult:
    winner: int            # PLAYER, OPPONENT, or 0 (draw)
    moves: int             # total half-moves played
    normalized_duration: float          # moves / board_cells
    estimated_player_difficulty: float  # mean across player moves; 0 if no observer
    player_move_quality: float          # mean across player moves; 0 if no observer


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def legal_moves(board: List[List[int]]) -> List[Tuple[int, int]]:
    """Return all empty cells as (row, col) tuples."""
    return [
        (r, c)
        for r, row in enumerate(board)
        for c, cell in enumerate(row)
        if cell == 0
    ]


def _check_winner(
    board: List[List[int]],
    m: int,
    n: int,
    k: int,
    last_move: Tuple[int, int],
    marker: int,
) -> bool:
    """
    Check whether `marker` has k-in-a-row passing through `last_move`.
    Runs in O(4k) — only examines lines through the last placed stone.
    """
    row, col = last_move
    # (delta_row, delta_col) for the four axis directions
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in directions:
        count = 1

        # Walk forward along the direction
        r, c = row + dr, col + dc
        while 0 <= r < m and 0 <= c < n and board[r][c] == marker:
            count += 1
            r += dr
            c += dc

        # Walk backward along the same axis
        r, c = row - dr, col - dc
        while 0 <= r < m and 0 <= c < n and board[r][c] == marker:
            count += 1
            r -= dr
            c -= dc

        if count >= k:
            return True

    return False


def _mean(values: list, default: float) -> float:
    return sum(values) / len(values) if values else default


# ---------------------------------------------------------------------------
# MNKGame
# ---------------------------------------------------------------------------

class MNKGame:
    """
    Generalized m×n board game where k consecutive marks in a row/col/diagonal win.

    Args:
        m: number of rows
        n: number of columns
        k: marks in a row needed to win
    """

    def __init__(self, m: int, n: int, k: int) -> None:
        if k > max(m, n):
            raise ValueError(f"k={k} can never be achieved on a {m}×{n} board")
        self.m = m
        self.n = n
        self.k = k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_round(
        self,
        player_policy,
        opponent_policy,
        player_first: bool = True,
        player_move_observer: Optional[Callable] = None,
    ) -> GameResult:
        """
        Play one full game between player_policy and opponent_policy.

        player_move_observer is called BEFORE each player move is placed on
        the board so that legal_moves() inside the observer still includes
        the chosen cell.  Signature:

            observer(board, move, marker, enemy_marker, k) -> {"difficulty": float, "quality": float}

        Returns:
            GameResult with winner, move count, duration, and aggregated
            player difficulty / quality estimates.
        """
        board = [[0] * self.n for _ in range(self.m)]
        total_cells = self.m * self.n
        move_count = 0

        player_difficulties: list[float] = []
        player_qualities: list[float] = []

        # Build alternating turn sequence
        if player_first:
            turns = [(PLAYER, player_policy), (OPPONENT, opponent_policy)]
        else:
            turns = [(OPPONENT, opponent_policy), (PLAYER, player_policy)]

        turn_idx = 0

        while True:
            marker, policy = turns[turn_idx % 2]
            enemy = OPPONENT if marker == PLAYER else PLAYER

            moves = legal_moves(board)
            if not moves:
                # Board full — draw
                return GameResult(
                    winner=0,
                    moves=move_count,
                    normalized_duration=move_count / total_cells,
                    estimated_player_difficulty=_mean(player_difficulties, 0.0),
                    player_move_quality=_mean(player_qualities, 0.0),
                )

            # Snapshot board BEFORE the move for the observer
            if marker == PLAYER and player_move_observer is not None:
                board_snapshot = [row[:] for row in board]

            move = policy.choose_move(board, marker, enemy, self.k)
            row, col = move

            # Validate move (guard against buggy policies)
            if board[row][col] != 0:
                raise ValueError(
                    f"Policy chose occupied cell {move} for marker {marker}"
                )

            board[row][col] = marker
            move_count += 1

            # Notify observer with pre-move snapshot
            if marker == PLAYER and player_move_observer is not None:
                obs = player_move_observer(board_snapshot, move, marker, enemy, self.k)
                player_difficulties.append(obs["difficulty"])
                player_qualities.append(obs["quality"])

            # Check win
            if _check_winner(board, self.m, self.n, self.k, move, marker):
                return GameResult(
                    winner=marker,
                    moves=move_count,
                    normalized_duration=move_count / total_cells,
                    estimated_player_difficulty=_mean(player_difficulties, 0.0),
                    player_move_quality=_mean(player_qualities, 0.0),
                )

            turn_idx += 1


# ---------------------------------------------------------------------------
# Baseline policy
# ---------------------------------------------------------------------------

class RandomPolicy:
    """Uniform-random legal move selector. q_value always returns 0."""

    def choose_move(
        self,
        board: List[List[int]],
        marker: int,
        enemy_marker: int,
        k: int,
    ) -> Tuple[int, int]:
        return random.choice(legal_moves(board))

    def q_value(
        self,
        board: List[List[int]],
        move: Tuple[int, int],
        marker: int,
        enemy_marker: int,
        k: int,
    ) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- legal_moves ---
    board = [[0, 1, 0], [-1, 0, 1], [0, 0, 0]]
    assert len(legal_moves(board)) == 6

    # --- _check_winner: row ---
    b = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    assert _check_winner(b, 3, 3, 3, (0, 2), 1)
    assert not _check_winner(b, 3, 3, 3, (0, 2), -1)

    # --- _check_winner: column ---
    b = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
    assert _check_winner(b, 3, 3, 3, (2, 0), 1)

    # --- _check_winner: diagonal ---
    b = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert _check_winner(b, 3, 3, 3, (1, 1), 1)

    # --- _check_winner: anti-diagonal ---
    b = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    assert _check_winner(b, 3, 3, 3, (1, 1), 1)

    # --- full game: Tic-Tac-Toe ---
    game = MNKGame(3, 3, 3)
    p = RandomPolicy()
    results = [game.run_round(p, p, player_first=(i % 2 == 0)) for i in range(200)]
    winners = [r.winner for r in results]
    assert all(w in (PLAYER, OPPONENT, 0) for w in winners)
    assert all(0 < r.moves <= 9 for r in results)
    assert all(0.0 < r.normalized_duration <= 1.0 for r in results)
    print(f"TicTacToe (3,3,3): "
          f"P={winners.count(PLAYER)}, O={winners.count(OPPONENT)}, "
          f"Draw={winners.count(0)} over 200 games")

    # --- full game: Gomoku (small 7×7 for speed) ---
    game5 = MNKGame(7, 7, 5)
    results5 = [game5.run_round(p, p) for _ in range(50)]
    assert all(r.moves <= 49 for r in results5)
    print(f"Gomoku-like (7,7,5): "
          f"P={sum(r.winner==PLAYER for r in results5)}, "
          f"O={sum(r.winner==OPPONENT for r in results5)}, "
          f"Draw={sum(r.winner==0 for r in results5)} over 50 games")

    # --- invalid k ---
    try:
        MNKGame(3, 3, 10)
        assert False, "should have raised"
    except ValueError:
        pass

    print("All tests passed.")
