import torch
import torch.nn as nn
import torch.nn.functional as F
from v1.gamestate import BOARD_SHAPE

class NeuralNet(nn.Module):
    def __init__(self, dropout_rate=0.5, initial_weights=""):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256 * BOARD_SHAPE * BOARD_SHAPE, 512)  # Flatten
        self.fc2 = nn.Linear(512, BOARD_SHAPE * BOARD_SHAPE + 1)  # Output Q-values for each action

        if initial_weights != "":
            # init model from pre-trained model
            self.load_state_dict(torch.load(initial_weights, weights_only=True))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.shape[0], -1)  # Flatten before FC layer
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Q-values for all possible actions
