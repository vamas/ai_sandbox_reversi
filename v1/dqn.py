import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim):
        """
        Initialize the Deep Q-Network.

        Args:
            input_dim (int): Number of input features (e.g., board size: 8x8 = 64).
            output_dim (int): Number of possible actions (legal moves).
            hidden_dim (int): Number of units in the hidden layers.
        """
        super(DQN, self).__init__()
        # self.network = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, (hidden_dim >> 2)),
        #     nn.ReLU(),
        #     nn.Linear((hidden_dim >> 2), output_dim)
        # )
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input state.

        Returns:
            Tensor: Q-values for all possible actions.
        """
        return self.network(x)