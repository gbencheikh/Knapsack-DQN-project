import numpy as np

def generate_instance(num_items, w_low, w_high, v_low, v_high, cap_low, cap_high):
    """
    Generate a random knapsack instance.

    Parameters
    ----------
    num_items : int
        Number of items in the instance.
    w_low, w_high : int
        Minimum and maximum weights of items.
    v_low, v_high : int
        Minimum and maximum values of items.
    cap_low, cap_high : int
        Minimum and maximum knapsack capacity.

    Returns
    -------
    items : list of tuples
        List of (weight, value) pairs.
    capacity : int
        Knapsack capacity.
    """
    
    weights = np.random.randint(w_low, w_high + 1, size=num_items)
    values = np.random.randint(v_low, v_high + 1, size=num_items)
    capacity = np.random.randint(cap_low, cap_high + 1)
    items = list(zip(weights.tolist(), values.tolist()))
    return items, int(capacity)

def dp_knapsack(items, capacity):
    """
    Solve the 0/1 Knapsack problem exactly using dynamic programming.

    Parameters
    ----------
    items : list of tuples
        Each tuple is (weight, value) representing an item.
    capacity : int
        Maximum weight the knapsack can carry.

    Returns
    -------
    int
        Maximum achievable value for the given items and capacity.

    Notes
    -----
    Uses a bottom-up DP approach with O(n*capacity) complexity,
    where n is the number of items.
    """
    n = len(items)
    # dp array of size (capacity+1)
    dp = [0] * (capacity + 1)
    for w, v in items:
        # iterate backwards to avoid reuse of item
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[capacity]