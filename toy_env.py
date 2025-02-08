
import numpy as np

class ToyEnv:
    """
    A simple 1D environment.
    The agent starts at the middle position of a line [0, size-1].
    Actions:
      0 - move left
      1 - move right
    The episode ends when the agent reaches either end.
    Rewards:
      -1 for reaching the left boundary (position 0).
      +1 for reaching the right boundary (position size-1).
    """
    def __init__(self, size=5):
        self.size = size
        self.start = size // 2
        self.reset()

    def reset(self):
        self.position = self.start
        return self.position

    def step(self, action):
        if action == 0:
            self.position -= 1
        elif action == 1:
            self.position += 1
        else:
            raise ValueError("Invalid action")
        
        done = False
        reward = 0
        
        if self.position <= 0:
            done = True
            reward = -1
            self.position = 0
        elif self.position >= self.size - 1:
            done = True
            reward = 1
            self.position = self.size - 1
        
        return self.position, reward, done, {}
