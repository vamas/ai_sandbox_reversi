import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 dropout_rate=0.2):
        """
        Initialize the Deep Q-Network with neuron drop out rate

        Args:
            input_dim (int): Number of input features (e.g., board size: 8x8 = 64).
            output_dim (int): Number of possible actions (legal moves).
            hidden_dim (int): Number of units in the hidden layers.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input state.

        Returns:
            Tensor: Q-values for all possible actions.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x