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
from v1.random_agent import RandomAgent

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DEFAULT_Q_VALUE = 0.0

games = 100
if __name__ == "__main__":

    with open('model.pkl', 'rb') as file:
        restored_qtable = pickle.load(file)

    black = QTableAgent(Player.BLACK, restored_qtable)
    white = RandomAgent(Player.WHITE)
    white = MinimaxAgent(Player.WHITE, 3)
    wins = {Player.BLACK: 0, Player.WHITE: 0, Player.NONE: 0}
    for i in tqdm(range(games), desc="Playing games"):
        game_manager = GameManager(black, white)
        winner = game_manager.run()
        wins[winner] += 1
    print(f"BLACK wins: ", wins[Player.BLACK])
    print(f"WHITE wins: ", wins[Player.WHITE])

    sys.exit()


# W9HPASFA95AH