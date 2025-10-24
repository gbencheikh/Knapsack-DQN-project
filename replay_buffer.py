from collections import deque
import random
import numpy as np

class ReplayBuffer:
    """
    Experience replay buffer for DQN training.

    Attributes
    ----------
    buffer : deque
        Stores tuples (state, action, reward, next_state, done)

    Methods
    -------
    push(s, a, r, s2, done)
        Add a transition to the buffer.
    sample(batch_size)
        Randomly sample a batch of transitions.
    __len__()
        Return current size of buffer.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = map(np.array, zip(*batch))
        return s, a, r, s2, done
    
    def __len__(self):
        return len(self.buffer)