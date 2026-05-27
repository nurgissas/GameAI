"""Meta-Agent for Dynamic Difficulty Adjustment"""


class MetaAgent:
    """Selects difficulty level based on player performance"""

    def __init__(self, num_levels=5, target_win_rate=0.5, window_size=5):
        """
        Args:
            num_levels: Number of difficulty levels (1-5)
            target_win_rate: Target win rate for player (0.0-1.0)
            window_size: Games to average over
        """
        self.num_levels = num_levels
        self.target_win_rate = target_win_rate
        self.current_level = num_levels // 2  # Start at medium
        self.window_size = window_size
        self.performance_history = []

        self.metrics = {
            'total_games': 0,
            'total_wins': 0,
            'switches': 0,
            'convergence_time': 0
        }

    def update_performance(self, player_won):
        """
        Update meta-agent with game result

        Args:
            player_won: Boolean, whether player won
        """
        self.performance_history.append(1 if player_won else 0)
        self.metrics['total_games'] += 1
        if player_won:
            self.metrics['total_wins'] += 1

        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)

    def select_difficulty(self):
        """
        Select difficulty level based on recent performance

        Returns:
            int: difficulty level index (0-indexed)
        """
        if len(self.performance_history) < self.window_size:
            return self.current_level

        recent_win_rate = sum(self.performance_history) / self.window_size
        old_level = self.current_level

        if recent_win_rate > self.target_win_rate + 0.1:
            self.current_level = min(self.current_level + 1, self.num_levels - 1)
            print(f"Difficulty adjusted UP to level {self.current_level + 1}")
        elif recent_win_rate < self.target_win_rate - 0.1:
            self.current_level = max(self.current_level - 1, 0)
            print(f"Difficulty adjusted DOWN to level {self.current_level + 1}")

        if old_level != self.current_level:
            self.metrics['switches'] += 1

        return self.current_level

    def get_metrics(self):
        """Return performance metrics"""
        win_rate = (self.metrics['total_wins'] / self.metrics['total_games']
                    if self.metrics['total_games'] > 0 else 0)

        return {
            'total_games': self.metrics['total_games'],
            'win_rate': win_rate,
            'difficulty_switches': self.metrics['switches'],
            'current_level': self.current_level + 1
        }


if __name__ == "__main__":
    meta = MetaAgent(num_levels=5)

    for i in range(20):
        player_won = i % 10 < 6  # 60% win rate
        meta.update_performance(player_won)
        level = meta.select_difficulty()

    print(meta.get_metrics())
    print("Meta-agent works!")
