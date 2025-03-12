import os
import sys
import torch
from tqdm import tqdm

from v1.dqn import DQN
from v1.dqn_agent import DQNAgent
from v1.gamestate import BOARD_SHAPE
from v1.minimax_agent import MinimaxAgent
from v1.player import Player
from v1.gamemanager import GameManager
from v1.random_agent import RandomAgent

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DEFAULT_Q_VALUE = 0.0

def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

models_path = "models/{}x{}/".format(BOARD_SHAPE,BOARD_SHAPE)
hidden_dim = 0
if BOARD_SHAPE == 8:
    hidden_dim = BOARD_SHAPE * BOARD_SHAPE * 128
elif BOARD_SHAPE == 4:
    hidden_dim = BOARD_SHAPE * BOARD_SHAPE * 32
elif BOARD_SHAPE == 6:
    hidden_dim = BOARD_SHAPE * BOARD_SHAPE * 64

games = 100
if __name__ == "__main__":
    b_performance = []
    w_performance = []
    model_b = DQN(BOARD_SHAPE * BOARD_SHAPE + BOARD_SHAPE * BOARD_SHAPE, BOARD_SHAPE * BOARD_SHAPE + 1, hidden_dim)
    model_w = DQN(BOARD_SHAPE * BOARD_SHAPE + BOARD_SHAPE * BOARD_SHAPE, BOARD_SHAPE * BOARD_SHAPE + 1, hidden_dim)
    model_b.load_state_dict(torch.load("{}dqn_model_full_20000_9.pth".format(models_path), weights_only=True))
    model_w.load_state_dict(torch.load("{}dqn_model_full_20000_0.pth".format(models_path), weights_only=True))
    black = DQNAgent(Player.BLACK, model_b, torch_device=get_device())
    white = DQNAgent(Player.WHITE, model_w, torch_device=get_device())
    # black = RandomAgent(Player.BLACK)
    # white = RandomAgent(Player.WHITE)
    white = MinimaxAgent(Player.WHITE, 6)
    # black = MinimaxAgent(Player.BLACK, 7)
    wins = {Player.BLACK: 0, Player.WHITE: 0, Player.NONE:0}
    for i in tqdm(range(games), desc="Playing games"):
        game_manager = GameManager(black, white)
        winner = game_manager.run()
        wins[winner] += 1
    print(f"BLACK wins: ", wins[Player.BLACK])
    print(f"WHITE wins: ", wins[Player.WHITE])
    b_performance.append(wins[Player.BLACK]/games)
    w_performance.append(wins[Player.WHITE] / games)

    print("=============================================================")
    print(b_performance)
    print(f"BLACK performance: ", sum(b_performance) / len(b_performance))
    print(f"WHITE performance: ", sum(w_performance) / len(w_performance))

    sys.exit()


# W9HPASFA95AH