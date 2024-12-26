import sys
from random import Random

import torch
from tqdm import tqdm
import pickle

from v1.dqn_agent import DQNAgent
from v1.dqn_train import DQNTrain
from v1.manual_agent import ManualAgent
from v1.player import Player
from v1.qtable_agent import QTableAgent
from v1.qtable_agent_double import DoubleQTableAgent
from v1.qtable_double import DoubleQTable
from v1.qtable_single import SingleQTable
from v1.qtable_train import QTableTrain
from v1.random_agent import RandomAgent
from v1.minimax_agent import MinimaxAgent
from v1.manual_agent import ManualAgent
from v1.gamemanager_console import GameManager
# from v1.gamemanager_gui import GameManager

DEFAULT_Q_VALUE = 0.0

games = 5
if __name__ == "__main__":

    # black = MinimaxAgent(Player.BLACK, 2)
    # white = RandomAgent(Player.WHITE)
    #
    # wins = {Player.BLACK: 0, Player.WHITE: 0, Player.NONE: 0}
    #
    # for i in tqdm(range(games), desc="Playing games"):
    #     game_manager = GameManager(black, white)
    #     winner = game_manager.run()
    #     wins[winner] += 1
    #
    # print(f"BLACK wins: ", wins[Player.BLACK])
    # print(f"WHITE wins: ", wins[Player.WHITE])

    # training = QTableTrain(total_games=10000,
    #                        epsilon=0.9,
    #                        learning_rate=0.1,
    #                        discount_factor=0.5,
    #                        train_agent=RandomAgent(Player.BLACK),
    #                        opponent_agent=RandomAgent(Player.WHITE),
    #                        qtable=SingleQTable(DEFAULT_Q_VALUE),
    #                        reward_decay=0.2)
    # qtable = training.train()
    #
    # training.print_stats()
    # with open('model.pkl', 'wb') as file:
    #     pickle.dump(qtable, file)
    # print("Training completed!. Size of qtable: {}".format(qtable.qtable_size()))
    # # print(qtable)
    #
    #
    # with open('model.pkl', 'rb') as file:
    #     restored_qtable = pickle.load(file)
    #
    # black = QTableAgent(Player.BLACK, restored_qtable)
    # white = RandomAgent(Player.WHITE)
    # wins = {Player.BLACK: 0, Player.WHITE: 0, Player.NONE: 0}
    # for i in tqdm(range(games), desc="Playing games"):
    #     game_manager = GameManager(black, white)
    #     winner = game_manager.run()
    #     wins[winner] += 1
    # print(f"WHITE wins: ", wins[Player.WHITE])
    # print(f"BLACK wins: ", wins[Player.BLACK])

    # with open('model.pkl', 'rb') as file:
    #     restored_qtable = pickle.load(file)
    #
    # training = QTableTrain(total_games=100000,
    #                        epsilon=0.99,
    #                        learning_rate=0.01,
    #                        discount_factor=0.6,
    #                        train_agent=RandomAgent(Player.BLACK),
    #                        opponent_agent=RandomAgent(Player.WHITE),
    #                        qtable=restored_qtable)
    # qtable = training.train()
    #
    # training.print_stats()
    # with open('model.pkl', 'wb') as file:
    #     pickle.dump(qtable, file)
    # print("Training completed!. Size of qtable: {}".format(qtable.qtable_size()))
    # # print(qtable)
    #
    # with open('model.pkl', 'rb') as file:
    #     restored_qtable = pickle.load(file)
    #
    # black = QTableAgent(Player.BLACK, restored_qtable)
    # white = RandomAgent(Player.WHITE)
    # wins = {Player.BLACK: 0, Player.WHITE: 0, Player.NONE: 0}
    # for i in tqdm(range(games), desc="Playing games"):
    #     game_manager = GameManager(black, white)
    #     winner = game_manager.run()
    #     wins[winner] += 1
    # print(f"WHITE wins: ", wins[Player.WHITE])
    # print(f"BLACK wins: ", wins[Player.BLACK])

    iterations = 1
    learning_rate_ratio = 20
    epsilon = 1.0
    discount_factor = 0.1
    reward_decay = 0.5
    total_games = 30000
    enhancement_epsilon = 0.3
    learning_rate = total_games / 1000000 / learning_rate_ratio
    b_performance = []
    for i in range(iterations):

        # print("Create baseline models")
        # # Initialize the model for an 8x8 Othello board black
        # training = DQNTrain(total_games=total_games,
        #                        epsilon=epsilon,
        #                        learning_rate=learning_rate,
        #                        discount_factor=discount_factor,
        #                        train_agent=RandomAgent(Player.BLACK),
        #                        opponent_agent=RandomAgent(Player.WHITE),
        #                        reward_decay=reward_decay)
        # model_black = training.train_dqn()
        # # training.print_stats()
        # # Save the model
        # torch.save(model_black, "dqn_black_model_full.pth")
        #
        # # Initialize the model for an 8x8 Othello board white
        # training = DQNTrain(total_games=total_games,
        #                     epsilon=epsilon,
        #                     learning_rate=learning_rate,
        #                     discount_factor=discount_factor,
        #                     train_agent=RandomAgent(Player.WHITE),
        #                     opponent_agent=RandomAgent(Player.BLACK),
        #                     reward_decay=reward_decay)
        # model_black = training.train_dqn()
        # # training.print_stats()
        # # Save the model.
        # torch.save(model_black, "dqn_white_model_full.pth")

        # # Enhance with Minimax-3
        # model_black = torch.load("dqn_black_model_full.pth")
        # model_white = torch.load("dqn_white_model_full.pth")
        # training = DQNTrain(total_games=total_games,
        #                     epsilon=enhancement_epsilon,
        #                     learning_rate=learning_rate,
        #                     discount_factor=discount_factor,
        #                     train_agent=RandomAgent(Player.BLACK),
        #                     opponent_agent=MinimaxAgent(Player.WHITE, 3),
        #                     reward_decay=reward_decay,
        #                     model = model_black)
        # model = training.train_dqn()
        # torch.save(model, "dqn_black_model_full.pth")
        #
        # training = DQNTrain(total_games=total_games,
        #                     epsilon=enhancement_epsilon,
        #                     learning_rate=learning_rate,
        #                     discount_factor=discount_factor,
        #                     train_agent=RandomAgent(Player.WHITE),
        #                     opponent_agent=MinimaxAgent(Player.BLACK, 3),
        #                     reward_decay=reward_decay,
        #                     model = model_white)
        # model = training.train_dqn()
        # torch.save(model, "dqn_white_model_full.pth")


        # Enhance with against DQN opponents

        # for iteration in range(3):
        #
        #     model_black = torch.load("dqn_black_model_full.pth")
        #     model_white = torch.load("dqn_white_model_full.pth")
        #
        #     training = DQNTrain(total_games=total_games,
        #                         epsilon=enhancement_epsilon,
        #                         learning_rate=learning_rate,
        #                         discount_factor=discount_factor,
        #                         train_agent=RandomAgent(Player.BLACK),
        #                         opponent_agent=DQNAgent(Player.WHITE, model_white),
        #                         reward_decay=reward_decay,
        #                         model = model_black)
        #     model = training.train_dqn()
        #     torch.save(model, "dqn_black_model_full.pth")
        #
        #     training = DQNTrain(total_games=total_games,
        #                         epsilon=enhancement_epsilon,
        #                         learning_rate=learning_rate,
        #                         discount_factor=discount_factor,
        #                         train_agent=RandomAgent(Player.WHITE),
        #                         opponent_agent=DQNAgent(Player.BLACK, model_black),
        #                         reward_decay=reward_decay,
        #                         model=model_white)
        #     model = training.train_dqn()
        #     torch.save(model, "dqn_white_model_full.pth")


        black_model = torch.load("dqn_black_model_full_20000.pth")
        white_model = torch.load("dqn_white_model_full_30000.pth")
        black = DQNAgent(Player.BLACK, black_model)
        white = DQNAgent(Player.WHITE, white_model)
        # black = MinimaxAgent(Player.BLACK, 3)
        # white = RandomAgent(Player.WHITE)
        # white = MinimaxAgent(Player.WHITE, 3)
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