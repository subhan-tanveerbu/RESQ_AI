import numpy as np
class RandomAgent:
    def __init__(self, action_size=7, seed=None):
        self.rng = np.random.default_rng(seed)
    def act(self, observation, env=None, explore=False):
        return int(self.rng.integers(0, 7))
