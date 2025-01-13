import os
import pickle
import sys
import torch
from tqdm import tqdm

from v1.dqn_agent import DQNAgent
from v1.minimax_agent import MinimaxAgent
from v1.player import Player
from v1.gamemanager_console import GameManager
from v1.qtable_agent import QTableAgent

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DEFAULT_Q_VALUE = 0.0

games = 100
if __name__ == "__main__":

    b_performance = []
    wins = {Player.BLACK: 0, Player.WHITE: 0, Player.NONE: 0}

    black_model = torch.load("dqn_black_model_full_10000.pth")
    with open('qtable_white_10000.pkl', 'rb') as file:
        restored_qtable = pickle.load(file)
    black = DQNAgent(Player.BLACK, black_model)
    white = QTableAgent(Player.WHITE, restored_qtable)

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