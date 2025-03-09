import os
import sys
import socket

import torch
import asyncio

from anyio import sleep

from infrastructure.metric_logger import periodic_task
from v1.dqn_agent import DQNAgent
from v1.dqn_train_selfplay import DQNTrain
from v1.minimax_agent import MinimaxAgent
from v1.player import Player
from v1.random_agent import RandomAgent
from v1.gamestate import BOARD_SHAPE

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEFAULT_Q_VALUE = 0.0
log_interval = 0.1

print("Host name: {}". format(socket.gethostname()))
print("Board shape: {}".format(BOARD_SHAPE))

models_path = "models/{}x{}/".format(BOARD_SHAPE,BOARD_SHAPE)
total_games = 40000
epsilon = 1.0
discount_factor = 0.3
reward_decay = 0.90
learning_rate = 0.0001
epsilon_min = 0.1
batch_size = 64
memory_size = 0
hidden_dim = 0

if BOARD_SHAPE == 8:
    memory_size = batch_size * 10
    hidden_dim = BOARD_SHAPE * BOARD_SHAPE * 128
elif BOARD_SHAPE == 4:
    memory_size = batch_size * 1
    hidden_dim = BOARD_SHAPE * BOARD_SHAPE * 32
elif BOARD_SHAPE == 6:
    memory_size = batch_size * 10
    hidden_dim = BOARD_SHAPE * BOARD_SHAPE * 64

async def main():
    torch.backends.mps.allow_tf32 = True  # Enable mixed precision
    print(torch.backends.mps.is_available())  # Should print: True®
    print(torch.backends.mps.is_built())  # Should print: True

    metric_logger = asyncio.create_task(periodic_task(log_interval))

    for i in range(1):
        await train(batch_size, discount_factor, epsilon, epsilon_min, hidden_dim, learning_rate, memory_size, reward_decay,
                total_games)
    await asyncio.sleep(10)
    metric_logger.cancel()


async def train(batch_size, discount_factor, epsilon, epsilon_min, hidden_dim, learning_rate, memory_size, reward_decay,
                total_games):
    # dqn_model = torch.load("{}dqn_model_full_100000_random_2.pth".format(models_path))

    # Initialize the model for an 8x8 Othello board black
    training = DQNTrain(total_games=total_games,
                            torch_device=get_device(True),
                            epsilon=epsilon,
                            learning_rate=learning_rate,
                            discount_factor=discount_factor,
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