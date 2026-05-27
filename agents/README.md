# Trained Agents

This folder stores pre-trained Q-Learning agents as `.pkl` files.

## Files

| File | Episodes | Expected Win Rate |
|------|----------|-------------------|
| `tictactoe_level_1.pkl` | 10,000 | ~25% |
| `tictactoe_level_2.pkl` | 20,000 | ~50% |
| `tictactoe_level_3.pkl` | 30,000 | ~70% |
| `tictactoe_level_4.pkl` | 40,000 | ~85% |
| `tictactoe_level_5.pkl` | 50,000 | ~95% |

## How to Generate

```bash
cd adaptive_difficulty_rl
python rl_training/train_base_agent.py
```

This trains for 50,000 episodes and saves a snapshot every 10,000 episodes.

## How to Load

```python
from rl_training.q_learning_agent import QLearningAgent

agent = QLearningAgent()
agent.load_agent('agents/tictactoe_level_3.pkl')
agent.epsilon = 0.0  # Disable exploration for inference
```
