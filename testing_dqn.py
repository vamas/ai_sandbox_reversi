import os
import sys
import torch
from tqdm import tqdm

from v1.dqn import NeuralNet
from v1.dqn_agent import NeuralNetworkAgent
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
    hidden_dim = BOARD_SHAPE * BOARD_SHAPE * 64
elif BOARD_SHAPE == 6:
    hidden_dim = BOARD_SHAPE * BOARD_SHAPE * 64

games = 100
if __name__ == "__main__":

    model_b = NeuralNet(hidden_dim)
    model_w = NeuralNet(hidden_dim)
    opponent_model = NeuralNet(hidden_dim)

    # Result: horizonals -  agent levels, verticals - opponent levels
    # test each generated model
    result = []
    for level in range(0, 3):
        model_b.load_state_dict(torch.load("{}dqn_model_full_60000_{}.pth".format(models_path, level), weights_only=True))
        model_w.load_state_dict(torch.load("{}dqn_model_full_60000_{}.pth".format(models_path, level), weights_only=True))
        black_agent = NeuralNetworkAgent(Player.BLACK, model_b, torch_device=get_device())
        white_agent = NeuralNetworkAgent(Player.WHITE, model_w, torch_device=get_device())

        # against opponent level
        opponent_level_results = []
        for opponent_level in range(0, 8):

            # flip black and white
            my_wins = 0
            for agent in [black_agent, white_agent]:

                wins = {Player.BLACK: 0, Player.WHITE: 0, Player.NONE: 0}

                # opponent_model.load_state_dict(torch.load("{}dqn_model_full_40000_{}.pth".format(models_path, opponent_level), weights_only=True))

                # white = NeuralNetworkAgent(Player.WHITE, opponent_model, torch_device=get_device()) if agent.player == Player.BLACK else white_agent
                # black = NeuralNetworkAgent(Player.BLACK, opponent_model, torch_device=get_device()) if agent.player == Player.WHITE else black_agent

                white = MinimaxAgent(Player.WHITE, opponent_level) if agent.player == Player.BLACK else white_agent
                black = MinimaxAgent(Player.BLACK, opponent_level) if agent.player == Player.WHITE else black_agent

                # play N games
                for i in tqdm(range(games), desc="Playing games"):
                    game_manager = GameManager(black, white)
                    winner = game_manager.run()
                    if winner == agent.player:
                        my_wins += 1
                    wins[winner] += 1

            opponent_level_results.append(int(my_wins/(games*2)*100))
            print(f"\nPlayed agent/{level} against opponent/{opponent_level}: {int(my_wins/(games*2)*100)}%")

        result.append(opponent_level_results)
    print(result)

    sys.exit()


# W9HPASFA95AH

# [[98, 72, 93, 73, 78, 78]]