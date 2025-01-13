import os
import sys
import torch

from v1.dqn_agent import DQNAgent
from v1.dqn_train_new import DQNTrain
from v1.minimax_agent import MinimaxAgent
from v1.player import Player
from v1.random_agent import RandomAgent

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DEFAULT_Q_VALUE = 0.0

games = 1
if __name__ == "__main__":

    total_games_ar = [100000]
    iterations = 1
    learning_rate_ratio = 20
    epsilon = 1.0
    discount_factor = 0.1
    reward_decay = 0.5
    for total_games in total_games_ar:
        learning_rate = total_games / 1000000 / learning_rate_ratio
        print("Create baseline models. BLACK {}".format(total_games))
        # Initialize the model for an 8x8 Othello board black
        training = DQNTrain(total_games=100,
                               epsilon=epsilon,
                               learning_rate=learning_rate,
                               discount_factor=discount_factor,
                               train_agent=RandomAgent(Player.BLACK),
                               opponent_agent=RandomAgent(Player.WHITE),
                               reward_decay=reward_decay,
                               epochs=total_games)
        model_black = training.train_dqn()
        training.print_stats()
        # Save the model
        print("Saving model to {}".format("dqn_black_model_full_{}.pth".format(total_games)))
        torch.save(model_black, "dqn_black_model_full_{}.pth".format(total_games))

        # # Initialize the model for an 8x8 Othello board white
        # print("Create baseline models. WHITE {}".format(total_games))
        # training = DQNTrain(total_games=total_games,
        #                     epsilon=epsilon,
        #                     learning_rate=learning_rate,
        #                     discount_factor=discount_factor,
        #                     train_agent=RandomAgent(Player.WHITE),
        #                     opponent_agent=RandomAgent(Player.BLACK),
        #                     reward_decay=reward_decay)
        # model_white = training.train_dqn()
        # training.print_stats()
        # # Save the model.
        # torch.save(model_white, "dqn_white_model_full_{}.pth".format(total_games))



    sys.exit()


# W9HPASFA95AH