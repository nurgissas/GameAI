"""Q-Learning Agent Implementation"""

import random
import pickle
import numpy as np

class QLearningAgent:
    """Q-Learning agent for board games"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.99, epsilon=0.3):
        """
        Initialize Q-Learning agent
        
        Args:
            learning_rate: How fast to update Q-values (0.0-1.0)
            discount_factor: Value of future rewards (0.0-1.0)
            epsilon: Exploration rate (0.0-1.0)
        """
        self.q_table = {}  # state -> {action: q_value}
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.training_stats = {
            'episodes': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0
        }
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        if state not in self.q_table:
            self.q_table[state] = {}
        return self.q_table[state].get(action, 0.0)
    
    def choose_action(self, state, valid_moves, training=True):
        """
        Choose action using epsilon-greedy strategy
        """
        if training and random.random() < self.epsilon:
            # Explore: random move
            return random.choice(valid_moves)
        else:
            # Exploit: best known move
            q_values = [self.get_q_value(state, move) for move in valid_moves]
            max_q = max(q_values)
            best_moves = [m for m, q in zip(valid_moves, q_values) if q == max_q]
            return random.choice(best_moves)
    
    def update_q_value(self, state, action, reward, next_state, next_valid_moves):
        """
        Update Q-value using Bellman equation:
        Q(s,a) = Q(s,a) + α [R + γ max Q(s',a') - Q(s,a)]
        """
        current_q = self.get_q_value(state, action)
        
        if next_valid_moves:
            max_next_q = max(self.get_q_value(next_state, a) 
                           for a in next_valid_moves)
        else:
            max_next_q = 0
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q
    
    def save_agent(self, filepath):
        """Save Q-table to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"✓ Agent saved to {filepath}")
    
    def load_agent(self, filepath):
        """Load Q-table from file"""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"✓ Agent loaded from {filepath}")
    
    def get_stats(self):
        """Return training statistics"""
        return {
            'episodes_trained': self.training_stats['episodes'],
            'states_learned': len(self.q_table),
            'total_games': sum([self.training_stats['wins'], 
                              self.training_stats['losses'],
                              self.training_stats['draws']])
        }

# TEST
if __name__ == "__main__":
    agent = QLearningAgent()
    print("✓ Agent created successfully!")