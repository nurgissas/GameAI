"""Fixed-size feature vector for MNK board positions.

Extracts 18 floats from (board, move, marker, k), independent of board size.
This allows a weight vector to generalize across board positions without
storing a separate Q-value for every possible state.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
N_FEATURES = 18


def _count_line(
    board: List[List[int]],
    r: int, c: int,
    dr: int, dc: int,
    marker: int,
    m: int, n: int,
) -> Tuple[int, int, int, int]:
    """
    Walk both directions from (r, c) counting consecutive marker pieces.
    (r, c) is assumed EMPTY — the move not yet placed.

    Returns:
        (fwd_count, bwd_count, fwd_open, bwd_open)
        where *_open is 1 if the end of the run is an empty cell.
    """
    fwd = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < m and 0 <= nc < n and board[nr][nc] == marker:
        fwd += 1
        nr += dr
        nc += dc
    fwd_open = int(0 <= nr < m and 0 <= nc < n and board[nr][nc] == 0)

    bwd = 0
    nr, nc = r - dr, c - dc
    while 0 <= nr < m and 0 <= nc < n and board[nr][nc] == marker:
        bwd += 1
        nr -= dr
        nc -= dc
    bwd_open = int(0 <= nr < m and 0 <= nc < n and board[nr][nc] == 0)

    return fwd, bwd, fwd_open, bwd_open


def extract(
    board: List[List[int]],
    move: Tuple[int, int],
    marker: int,
    k: int,
) -> np.ndarray:
    """
    Build an 18-float feature vector for placing `marker` at `move`.

    `board` is the state BEFORE the move is placed.

    Layout (4 directions × 4 feature groups = 16, plus 2 global):
      0-3   my run length after placing / k  (one per direction, clamped to 1)
      4-7   my open ends / 2                 (one per direction)
      8-11  opponent pieces in forward dir / k
      12-15 opponent pieces in backward dir / k
      16    normalized distance from board center  (0 = center, 1 = corner)
      17    board fill ratio
    """
    m = len(board)
    n = len(board[0])
    r, c = move
    enemy = -marker

    phi = np.zeros(N_FEATURES, dtype=np.float32)

    for d, (dr, dc) in enumerate(DIRECTIONS):
        my_fwd, my_bwd, my_fwd_open, my_bwd_open = _count_line(
            board, r, c, dr, dc, marker, m, n
        )
        opp_fwd, opp_bwd, _, _ = _count_line(
            board, r, c, dr, dc, enemy, m, n
        )

        phi[d]      = min((1 + my_fwd + my_bwd) / k, 1.0)
        phi[4 + d]  = (my_fwd_open + my_bwd_open) / 2.0
        phi[8 + d]  = min(opp_fwd / k, 1.0)
        phi[12 + d] = min(opp_bwd / k, 1.0)

    cr, cc = (m - 1) / 2.0, (n - 1) / 2.0
    max_dist = (cr ** 2 + cc ** 2) ** 0.5 or 1.0
    phi[16] = ((r - cr) ** 2 + (c - cc) ** 2) ** 0.5 / max_dist

    filled = sum(1 for row in board for cell in row if cell != 0)
    phi[17] = filled / (m * n)

    return phi


if __name__ == "__main__":
    board = [[0] * 3 for _ in range(3)]
    phi = extract(board, (1, 1), 1, 3)
    assert phi.shape == (N_FEATURES,)
    assert phi[16] == 0.0, "center cell should have distance 0"

    # place one piece and check fill ratio updates
    board[0][0] = 1
    phi2 = extract(board, (1, 1), 1, 3)
    assert phi2[17] > 0.0

    print(f"Feature vector (center of empty 3x3): {phi}")
    print("feature_extractor: all checks passed.")
