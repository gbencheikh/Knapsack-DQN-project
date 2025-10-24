import torch.nn as nn

class DQNNet(nn.Module):
    """
    Deep Q-Network for 0/1 Knapsack.

    Architecture:
    - Input layer: size 5 (state features)
    - Two hidden layers (default 128 units, ReLU)
    - Output layer: size 2 (action values: skip/take)

    Methods
    -------
    forward(x)
        Compute Q-values for given input states.
    """

    def __init__(self, input_dim, output_dim, hidden=128):
        super(DQNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Batch of states of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Q-values of shape (batch_size, 2)
        """
        return self.net(x)