import numpy as np
from utils import generate_instance

class KnapsackEnvRandom:
    """
    Environment for 0/1 Knapsack problem.
    
    Attributes
    ----------
    items : list of tuples
        List of (weight, value) items.
    capacity : int
        Maximum allowed weight.
    index : int
        Current item index.
    remaining : int
        Remaining capacity.
    
    Methods
    -------
    reset(num_items=None)
        Reset environment with a new random instance.
    step(action)
        Take action (0=skip, 1=take) and return next state, reward, done.
    _get_state()
        Return normalized state representation for current item.
    """
    
    def __init__(self, num_items=12, penalty=-5, w_max=15, v_max=20, capacity_scale=True):
        self.num_items_cfg = num_items
        self.penalty = penalty
        self.w_max = w_max
        self.v_max = v_max
        self.capacity_scale = capacity_scale
        self.reset()
    
    def reset(self, num_items=None):
        if num_items is None:
            num_items = self.num_items_cfg
        self.items, self.capacity = generate_instance(num_items=num_items,
                                                      w_low=1, w_high=self.w_max,
                                                      v_low=1, v_high=self.v_max,
                                                      cap_low=max(5, num_items), cap_high=self.w_max * num_items // 2 + 5)
        self.num_items = len(self.items)
        self.index = 0
        self.remaining = self.capacity
        return self._get_state()
    
    def _get_state(self):
        # if index out of range, return dummy (shouldn't be queried)
        if self.index >= self.num_items:
            # zeros
            return np.zeros(5, dtype=np.float32)
        w, v = self.items[self.index]
        # normalize
        w_n = w / float(self.w_max)
        v_n = v / float(self.v_max)
        rem_n = self.remaining / float(self.capacity) if self.capacity > 0 else 0.0
        idx_n = self.index / float(self.num_items)
        cap_n = self.capacity / float(self.w_max * self.num_items)  # rough normalization
        return np.array([w_n, v_n, rem_n, idx_n, cap_n], dtype=np.float32)
    
    def step(self, action):
        done = False
        if self.index >= self.num_items:
            return self._get_state(), 0.0, True
        w, v = self.items[self.index]
        reward = 0.0
        if action == 1:
            if w <= self.remaining:
                reward = float(v)
                self.remaining -= w
            else:
                reward = float(self.penalty)
        self.index += 1
        if self.index == self.num_items:
            done = True
        return self._get_state(), reward, done