import os
import pickle
import sys
import torch

from v1.dqn_agent import DQNAgent
from v1.dqn_helpers import print_stats
from v1.dqn_train_new import DQNTrain
from v1.minimax_agent import MinimaxAgent
from v1.player import Player
from v1.qtable_agent import QTableAgent
from v1.random_agent import RandomAgent

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DEFAULT_Q_VALUE = 0.0

models_path = "models/8x8/"

games = 1
if __name__ == "__main__":

    # Best config so far - 92%
    # epsilon = 1.0
    # discount_factor = 0.1
    # reward_decay = 0.9
    # learning_rate = 0.0001
    # epsilon_min = 0.001
    # memory_size = 64
    # epochs = 1
    # batch_size = 64

    total_games_ar = [50000]
    epsilon = 1.0
    discount_factor = 0.1
    reward_decay = 0.9
    learning_rate = 0.001
    epsilon_min = 0.001
    memory_size = 64
    epochs = 1
    batch_size = 64
    hidden_dim = 512

    for total_games in total_games_ar:
        # print("Create baseline models. BLACK {}".format(total_games))
        # white_model = torch.load("{}dqn_model_full_100000.pth".format(models_path))
        # with open('{}qtable_white_40000.pkl'.format(models_path), 'rb') as file:
        #     restored_qtable = pickle.load(file)

        # Initialize the model for an 8x8 Othello board black
        training = DQNTrain(total_games=total_games,
                               epsilon=epsilon,
                               learning_rate=learning_rate,
                               discount_factor=discount_factor,
                               opponent_agents=[
                                                RandomAgent(Player.WHITE),
                                                # RandomAgent(Player.WHITE),
                                                # DQNAgent(Player.WHITE, white_model),
                                                # QTableAgent(Player.WHITE, restored_qtable),
                                                # MinimaxAgent(Player.WHITE, 0),
                                                # MinimaxAgent(Player.WHITE, 1),
                                                # MinimaxAgent(Player.WHITE, 2),
                                                # MinimaxAgent(Player.WHITE, 4),
                                                ],
                               testing_agents=[
                                   RandomAgent(Player.WHITE),
                                   MinimaxAgent(Player.WHITE, 0, "Heuristic"),
                                   MinimaxAgent(Player.WHITE, 1, "Minimax1"),
                                   # MinimaxAgent(Player.WHITE, 2, "Minimax2"),
                                   # MinimaxAgent(Player.WHITE, 3, "Minimax3"),
                               ],
                               reward_decay=reward_decay,
                               memory_size=memory_size,
                               batch_size=batch_size,
                               model=None,
                               epochs=epochs,
                               epsilon_min=epsilon_min,
                               hidden_dim=hidden_dim,
                               self_instances=1)
        model_black = training.train_dqn()
        stats = training.get_stats()
        print_stats(stats[0], stats[1], stats[2], stats[3], stats[4], stats[5])
        # Save the model
        print("Saving model to {}".format("dqn_model_full_{}.pth".format(total_games)))
        torch.save(model_black, "{}dqn_model_full_{}.pth".format(models_path, total_games))

    sys.exit()


# W9HPASFA95AH