import os
import sys
import torch
from tqdm import tqdm

from v1.dqn_agent import DQNAgent
from v1.minimax_agent import MinimaxAgent
from v1.player import Player
from v1.gamemanager_console import GameManager

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DEFAULT_Q_VALUE = 0.0

games = 1
if __name__ == "__main__":

    b_performance = []
    black_model = torch.load("dqn_black_model_full_10000_plus.pth")
    white_model = torch.load("dqn_white_model_full_10000.pth")
    black = DQNAgent(Player.BLACK, black_model)
    white = DQNAgent(Player.WHITE, white_model)
    # black = MinimaxAgent(Player.BLACK, 3)
    # white = RandomAgent(Player.WHITE)
    white = MinimaxAgent(Player.WHITE, 3)
    wins = {Player.BLACK: 0, Player.WHITE: 0, Player.NONE: 0}
    for i in tqdm(range(games), desc="Playing games"):
        game_manager = GameManager(black, white)
        winner = game_manager.run()
        wins[winner] += 1
    print(f"WHITE wins: ", wins[Player.WHITE])
    print(f"BLACK wins: ", wins[Player.BLACK])
    b_performance.append(wins[Player.BLACK]/games)

    print("=============================================================")
    print(b_performance)
    print(f"BLACK performance: ", sum(b_performance) / len(b_performance))

    sys.exit()


# W9HPASFA95AH