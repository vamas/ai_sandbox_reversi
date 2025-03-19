import os
import sys
import torch
from tqdm import tqdm

from v1.cnn import NeuralNet
from v1.cnn_agent import NeuralNetworkAgent
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

    model_b = NeuralNet()
    model_w = NeuralNet()
    opponent_model = NeuralNet()

    # Result: horizonals -  agent levels, verticals - opponent levels
    # test each generated model
    result = []
    for level in range(3, 4):
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
                #
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
    #     # white = MinimaxAgent(Player.WHITE, 7)
    #     black = MinimaxAgent(Player.BLACK, 7)
    #
    #     b_performance = []
    #     w_performance = []
    #     wins = {Player.BLACK: 0, Player.WHITE: 0, Player.NONE:0}
    #     for i in tqdm(range(games), desc="Playing games"):
    #         game_manager = GameManager(black, white)
    #         winner = game_manager.run()
    #         wins[winner] += 1
    #     print(f"BLACK wins: ", wins[Player.BLACK])
    #     print(f"WHITE wins: ", wins[Player.WHITE])
    #     b_performance.append(wins[Player.BLACK]/games)
    #     w_performance.append(wins[Player.WHITE] / games)
    #
    #     print("=============================================================")
    #     print(b_performance)
    #     print(f"BLACK performance: ", sum(b_performance) / len(b_performance))
    #     print(f"WHITE performance: ", sum(w_performance) / len(w_performance))
    #
    # model_b.load_state_dict(torch.load("{}dqn_model_full_40000_0.pth".format(models_path), weights_only=True))
    # model_w.load_state_dict(torch.load("{}dqn_model_full_40000_0.pth".format(models_path), weights_only=True))
    # black = DQNAgent(Player.BLACK, model_b, torch_device=get_device())
    # white = DQNAgent(Player.WHITE, model_w, torch_device=get_device())
    # # black = RandomAgent(Player.BLACK)
    # # white = RandomAgent(Player.WHITE)
    # # white = MinimaxAgent(Player.WHITE, 7)
    # black = MinimaxAgent(Player.BLACK, 7)
    #
    # b_performance = []
    # w_performance = []
    # wins = {Player.BLACK: 0, Player.WHITE: 0, Player.NONE:0}
    # for i in tqdm(range(games), desc="Playing games"):
    #     game_manager = GameManager(black, white)
    #     winner = game_manager.run()
    #     wins[winner] += 1
    # print(f"BLACK wins: ", wins[Player.BLACK])
    # print(f"WHITE wins: ", wins[Player.WHITE])
    # b_performance.append(wins[Player.BLACK]/games)
    # w_performance.append(wins[Player.WHITE] / games)
    #
    # print("=============================================================")
    # print(b_performance)
    # print(f"BLACK performance: ", sum(b_performance) / len(b_performance))
    # print(f"WHITE performance: ", sum(w_performance) / len(w_performance))

    sys.exit()


# W9HPASFA95AH

# minimax
# [[99, 98, 96, 93, 100, 95, 52, 27], [99, 51, 97, 94, 91, 95, 50, 26], [100, 47, 98, 41, 94, 94, 52, 38], [100, 49, 96, 78, 83, 94, 51, 30], [99, 92, 98, 82, 93, 88, 92, 17]]
# self
# [[45, 75, 68, 79, 46], [19, 48, 48, 50, 4], [27, 50, 48, 49, 20], [20, 50, 48, 49, 15], [49, 96, 77, 80, 51]]