"""Tic-Tac-Toe Game Implementation"""

from .base_game import BaseGame

class TicTacToeGame(BaseGame):
    """Tic-Tac-Toe game environment"""
    
    def __init__(self):
        """Initialize game board"""
        self.board = [' '] * 9  # Positions 0-8
        self.current_player = 'X'
    
    def reset(self):
        """Reset board for new game"""
        self.board = [' '] * 9
        self.current_player = 'X'
    
    def get_state(self):
        """Return hashable board state"""
        return tuple(self.board)
    
    def get_valid_moves(self):
        """Return list of empty positions"""
        return [i for i in range(9) if self.board[i] == ' ']
    
    def make_move(self, position):
        """Place symbol at position"""
        if position < 0 or position > 8:
            return False
        if self.board[position] != ' ':
            return False
        
        self.board[position] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True
    
    def get_winner(self):
        """Check for winner"""
        winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]               # Diagonals
        ]
        
        for combo in winning_combos:
            a, b, c = combo
            if (self.board[a] != ' ' and 
                self.board[a] == self.board[b] == self.board[c]):
                return self.board[a]
        
        if ' ' not in self.board:
            return 'Draw'
        
        return None
    
    def display(self):
        """Print board"""
        print("\n")
        for i in range(0, 9, 3):
            print(f" {self.board[i]} | {self.board[i+1]} | {self.board[i+2]}")
            if i < 6:
                print("-----------")
        print("\n")

# TEST
if __name__ == "__main__":
    game = TicTacToeGame()
    game.make_move(4)
    game.make_move(0)
    print(f"State: {game.get_state()}")
    print(f"Valid moves: {game.get_valid_moves()}")
    print("✓ Game logic works!")