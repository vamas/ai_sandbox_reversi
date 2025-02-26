import os
import sys
import socket

import torch
import asyncio

from infrastructure.metric_logger import periodic_task
from v1.dqn_agent import DQNAgent
from v1.dqn_train_new import DQNTrain
from v1.minimax_agent import MinimaxAgent
from v1.player import Player
from v1.random_agent import RandomAgent
from v1.gamestate import BOARD_SHAPE
from v1.gamemanager_gui import GameManager

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DEFAULT_Q_VALUE = 0.0

models_path = "models/{}x{}/".format(BOARD_SHAPE,BOARD_SHAPE)

def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

if __name__ == "__main__":
    black_model = torch.load("{}dqn_model_full_100000.pth".format(models_path))
    white_model = torch.load("{}dqn_model_full_100000.pth".format(models_path))
    black = DQNAgent(Player.BLACK, black_model, torch_device=get_device())
    white = DQNAgent(Player.WHITE, white_model, torch_device=get_device())
    # black = RandomAgent(Player.BLACK)
    # white = RandomAgent(Player.WHITE)
    white = MinimaxAgent(Player.WHITE, 0)
    # black = MinimaxAgent(Player.BLACK, 3)

    game = GameManager(black, white)
    game.run()