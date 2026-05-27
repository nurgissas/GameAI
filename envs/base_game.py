"""Base class for game environments"""

from abc import ABC, abstractmethod

class BaseGame(ABC):
    """Abstract base class for all games"""
    
    @abstractmethod
    def reset(self):
        """Reset game to initial state"""
        pass
    
    @abstractmethod
    def get_state(self):
        """Get current game state"""
        pass
    
    @abstractmethod
    def get_valid_moves(self):
        """Get list of valid moves"""
        pass
    
    @abstractmethod
    def make_move(self, position):
        """Make a move, return True if valid"""
        pass
    
    @abstractmethod
    def get_winner(self):
        """Get winner: 'X', 'O', 'Draw', or None"""
        pass