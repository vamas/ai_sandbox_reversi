import os
import pickle
import sys
import torch

from v1.player import Player
from v1.qtable import QTable
from v1.qtable_single import SingleQTable
from v1.qtable_single_db import SingleQTableDB
from v1.minimax_agent import MinimaxAgent
from v1.qtable_single_db_plus import SingleQTableDBPlus
from v1.qtable_train import QTableTrain
from v1.random_agent import RandomAgent

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DEFAULT_Q_VALUE = 0.0

games = 10
if __name__ == "__main__":

    total_games = 40000

    training = QTableTrain(total_games=total_games,
                           epsilon=0.9999,
                           learning_rate=0.1,
                           discount_factor=0.5,
                           train_agent=RandomAgent(Player.WHITE),
                           opponent_agent=RandomAgent(Player.BLACK),
                           qtable=SingleQTable(DEFAULT_Q_VALUE),
                           reward_decay=0.5)
    qtable = training.train()
    training.print_stats()
    with open("qtable_white_{}.pkl".format(total_games), 'wb') as file:
        pickle.dump(qtable, file)
    print("Training completed!. Size of qtable: {}".format(qtable.qtable_size()))

    sys.exit()


# W9HPASFA95AH