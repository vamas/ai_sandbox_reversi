import os
import sys
import torch
from tqdm import tqdm

from v1.dqn_agent import DQNAgent
from v1.minimax_agent import MinimaxAgent
from v1.player import Player
from v1.gamemanager_console import GameManager
from v1.random_agent import RandomAgent



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DEFAULT_Q_VALUE = 0.0

models_path = "models/4x4/"

if __name__ == "__main__":
    sys.exit()


# W9HPASFA95AH