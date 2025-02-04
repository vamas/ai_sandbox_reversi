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

games = 100
if __name__ == "__main__":

    b_performance = []
    black_model = torch.load("{}dqn_model_full_1000.pth".format(models_path))
    white_model = torch.load("{}dqn_model_full_1000.pth".format(models_path))
    black = DQNAgent(Player.BLACK, black_model)
    white = DQNAgent(Player.WHITE, white_model)
    # black = RandomAgent(Player.BLACK)
    # white = RandomAgent(Player.WHITE)
    # white = MinimaxAgent(Player.WHITE, 9)
    # black = MinimaxAgent(Player.BLACK, 5)
    wins = {Player.BLACK: 0, Player.WHITE: 0, Player.NONE: 0}
    for i in tqdm(range(games), desc="Playing games"):
        game_manager = GameManager(black, white)
        winner = game_manager.run()
        wins[winner] += 1
    print(f"BLACK wins: ", wins[Player.BLACK])
    print(f"WHITE wins: ", wins[Player.WHITE])
    b_performance.append(wins[Player.BLACK]/games)

    print("=============================================================")
    print(b_performance)
    print(f"BLACK performance: ", sum(b_performance) / len(b_performance))

    sys.exit()


# W9HPASFA95AH