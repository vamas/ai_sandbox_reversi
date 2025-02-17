import os
import sys
import socket

import torch
import asyncio

from anyio import sleep

from infrastructure.metric_logger import periodic_task
from v1.dqn_train_new import DQNTrain
from v1.minimax_agent import MinimaxAgent
from v1.player import Player
from v1.random_agent import RandomAgent
from v1.gamestate import BOARD_SHAPE

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DEFAULT_Q_VALUE = 0.0

models_path = "models/4x4/"

games = 1

log_interval = 0.1

print(socket.gethostname())

async def main():
    metric_logger = asyncio.create_task(periodic_task(log_interval))

    # Best config so far - 92%
    # epsilon = 1.0
    # discount_factor = 0.15
    # reward_decay = 0.9
    # learning_rate = 0.001
    # epsilon_min = 0.001
    # memory_size = batch_size * 10
    # epochs = 1
    # batch_size = 64
    # hidden_dim = BOARD_SHAPE * BOARD_SHAPE * 12

    total_games_ar = [10000]
    epsilon = 1.0
    discount_factor = 0.3
    reward_decay = 0.6
    learning_rate = 0.001
    epsilon_min = 0.1
    batch_size = 64
    memory_size = batch_size * 10
    hidden_dim = BOARD_SHAPE * BOARD_SHAPE * 12

    for i in range(3):
        await train(batch_size, discount_factor, epsilon, epsilon_min, hidden_dim, learning_rate, memory_size, reward_decay,
                total_games_ar)

    # sys.exit()
    await asyncio.sleep(10)
    metric_logger.cancel()


async def train(batch_size, discount_factor, epsilon, epsilon_min, hidden_dim, learning_rate, memory_size, reward_decay,
                total_games_ar):
    for total_games in total_games_ar:
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
                                # RandomAgent(Player.WHITE),
                                # MinimaxAgent(Player.WHITE, 0, "Heuristic"),
                                # MinimaxAgent(Player.WHITE, 1, "Minimax1"),
                                # MinimaxAgent(Player.WHITE, 2, "Minimax2"),
                                MinimaxAgent(Player.WHITE, 3, "Minimax3"),
                            ],
                            reward_decay=reward_decay,
                            memory_size=memory_size,
                            batch_size=batch_size,
                            epsilon_min=epsilon_min,
                            hidden_dim=hidden_dim,
                            self_instances=1)
        await sleep(log_interval)
        model = await asyncio.to_thread(training.train_dqn)
        # Save the model
        print("Saving model to {}".format("dqn_model_full_{}.pth".format(total_games)))
        torch.save(model, "{}dqn_model_full_{}.pth".format(models_path, total_games))


# Run the async event loop
asyncio.run(main())

# W9HPASFA95AH