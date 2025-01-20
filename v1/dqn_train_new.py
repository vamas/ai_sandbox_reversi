# Train qtable agent against random agent
import math
from idlelib.pyparse import trans
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from v1.dqn import DQN
from v1.dqn_replaybuffer import ReplayBuffer
from v1.gamestate import GameState, print_board
from v1.player import Player, opponent
from v1.position import Position, SkipPosition
from v1.random_agent import RandomAgent

WIN_VALUE = 1.0
DRAW_VALUE = 0.5
LOSS_VALUE = 0.0

DEFAULT_Q_VALUE = 1.0

BOARD_SHAPE = 4

def game_state_one_hot_encode(game_state, shape=BOARD_SHAPE):
    if game_state is None:
        return np.zeros(shape * shape + shape * shape, dtype=float)
    state = np.zeros(game_state.Rows * game_state.Cols + game_state.Rows * game_state.Cols, dtype=float)
    for row in range(game_state.Rows):
        for col in range(game_state.Cols):
            if game_state.board[row][col] != Player.NONE:
                idx = row * game_state.Cols + col
                if game_state.board[row][col] == Player.BLACK:
                    state[idx] = 1
                elif game_state.board[row][col] == Player.WHITE:
                    state[idx + shape * shape] = 1
    return state.flatten()

def game_state_one_hot_decode(encoded_state, shape=BOARD_SHAPE):
    """
    Decode a one-hot encoded game state back to a GameState object.

    Args:
        encoded_state (np.array): One-hot encoded game state.
        shape (int): The shape of the board (default is BOARD_SHAPE).

    Returns:
        GameState: The decoded GameState object.
    """
    board = np.zeros((shape, shape), dtype=int)
    half = shape * shape
    for idx in range(half):
        row = idx // shape
        col = idx % shape
        if encoded_state[idx] == 1:
            board[row][col] = Player.BLACK
        elif encoded_state[idx + half] == 1:
            board[row][col] = Player.WHITE
    return GameState(board=board)

def position_one_hot_encode(position, shape=BOARD_SHAPE):
    if position.row == -1 and position.col == -1:
        return 1 << 16
    return 1 << (position.row * shape + position.col)

def position_one_hot_decode(action_encoded, shape=BOARD_SHAPE):
    if action_encoded == (1 << 16):
        return SkipPosition()
    index = int(math.log2(action_encoded))
    x = index // shape
    y = index % shape
    return Position(x, y)

# def action_one_hot_encode(position, shape=BOARD_SHAPE):
#     if position.row == -1 and position.col == -1:
#         return
#     return (position.row << 2) | position.col
#
# def action_one_hot_decode(action_encoded, shape=BOARD_SHAPE):
#     x = (action_encoded >> 2) & 0b11
#     y = action_encoded & 0b11
#     return Position(x, y)

def one_hot_encoding_to_idx(encoded):
    if encoded == (1 << 16):
        return 16
    return int(math.log2(encoded))


# def action_encode(move_info, shape=BOARD_SHAPE):
#     if move_info.position.row == -1 and move_info.position.col == -1:
#         return shape*shape
#     return shape * move_info.position.row + move_info.position.col

def action_encode(position, shape=BOARD_SHAPE):
    if position.row == -1 and position.col == -1:
        return shape*shape
    return shape * position.row + position.col

def action_decode(action_encoded, shape=BOARD_SHAPE):
    if action_encoded == shape*shape:
        return SkipPosition()
    return Position(action_encoded // shape, action_encoded % shape)

# - Play game
# - Make first training move
#   - Play episode till the end
#   - Enrich history with reward values base on the final reward
#   - Append history to the replay buffer
#   - Check if replay buffer is full, if so, sample batch and train the model
#   - Continue loop

class DQNTrain:

    def __init__(self,
                 hidden_dim=256,
                 total_games=100,
                 learning_rate=0.4,
                 discount_factor=1.0,
                 epsilon=0.7,
                 train_agent=RandomAgent(Player.BLACK),
                 opponent_agents=[RandomAgent(Player.WHITE)],
                 reward_decay=0.9,
                 memory_size=1000,
                 batch_size=64,
                 model=None,
                 epochs=1,
                 epsilon_min=0.1):
        """
        Initialize the Deep Q-Network.

        Args:
            input_dim (int): Number of input features (e.g., board size: 8x8 = 64).
            output_dim (int): Number of possible actions (legal moves).
            hidden_dim (int): Number of units in the hidden layers.
        """
        self.total_games = total_games
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.opponent_agents = opponent_agents
        self.train_agent = train_agent
        self.agents = {
            Player.BLACK: self.train_agent if self.train_agent.player == Player.BLACK else self.opponent_agents[0],
            Player.WHITE: self.opponent_agents[0] if self.opponent_agents[0].player == Player.WHITE else self.train_agent
        }
        self.train_agent = train_agent
        self.episode = 0
        self.total_rewards = []
        self.avg_q_values = []
        self.reward_decay = reward_decay
        self.input_dim = BOARD_SHAPE * BOARD_SHAPE + BOARD_SHAPE * BOARD_SHAPE
        self.output_dim = BOARD_SHAPE * BOARD_SHAPE + 1
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.model = None
        self.optimizer =None
        self.loss_fn = None
        self.target_model = None
        self.target_update_freq = 1000. # Update target q-network every other 1000 steps (played games)
        self.replay_buffer = None
        self.epoch_q_value_changes = []
        self.is_exploration = True
        self.model = model
        self.epochs = epochs
        self.illegal_moves = []
        self.epoch_illegal_moves = 0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = np.exp(np.log(epsilon_min / epsilon) / total_games)
        self.recent_avg_reward = 0
        self.epsilon_history = []

    def initialize_model(self):
        """
            Initialize the DQN model, optimizer, and loss function.

            Args:
                board_size (int): Dimensions of the board (default: 8x8).
                action_space_size (int): Number of possible actions.
                hidden_dim (int): Number of units in hidden layers.
                learning_rate (float): Learning rate for the optimizer.

            Returns:
                model (DQN): The initialized DQN model.
                optimizer (torch.optim.Optimizer): Optimizer for training the model.
                loss_fn (nn.Module): Loss function for DQN.
            """
        input_dim = self.input_dim  # Flattened board as input
        output_dim = self.output_dim  # Total number of possible actions

        model = DQN(input_dim, output_dim, self.hidden_dim)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(self.total_games/100), gamma=0.85)

        for layer in model.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # for name, param in model.named_parameters():
        #     print(f"Layer: {name}, Weights Mean: {param.mean().item()}, Std: {param.std().item()}")

        loss_fn = nn.SmoothL1Loss()

        return model, optimizer, loss_fn, scheduler

    def train_dqn(self):
        """
        Train the DQN model.

        Return:
            model (DQN): The trained DQN model.
        """
        if self.model is None:
            self.model, self.optimizer, self.loss_fn, self.scheduler = self.initialize_model()
        else:
            _, self.optimizer, self.loss_fn, self.scheduler = self.initialize_model()
        self.target_model, _, _, _ = self.initialize_model()
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize with same weights
        self.replay_buffer = ReplayBuffer(self.memory_size)
        self.epoch_q_value_changes = []
        self.total_rewards = []
        self.avg_q_values = []
        self.is_exploration = True
        for game in tqdm(range(self.total_games), desc="Training DQN"):
            # print("Game/Total games {}/{}".format(game + 1, self.total_games))
            self.rotate_opponent_agents()
            self.play_game()
            self.update_epsilon_boltzmann(game)

            if random.random() < self.epsilon:
                self.is_exploration = True
            else:
                self.is_exploration = False

            if self.replay_buffer.is_buffer_ready:
                # print("Train")
                self.train_model(self.replay_buffer)
                self.illegal_moves.append(self.epoch_illegal_moves)
                self.epoch_illegal_moves = 0
            if game % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # print("Epsilon: {}", self.epsilon)
        return self.model

    def play_game(self):
        """
        Play a game of Othello
        When playing a game we make first exploration/exploitation move
        and then finish up episode playing according to the training player
        and opponent strategies
        """
        game_state = GameState()
        while not game_state.game_over:
            game_state_before, move_info = self.make_training_move(game_state)
            game_history = [] # List of (move_info, game_state) tuples
            # self.print_board(game_state_before)
            if move_info is not None:
                # self.print_board(game_state_before)
                game_history.append((move_info, game_state_before))
                game_episode_state = game_state.clone()
                winner = self.play_episode(game_episode_state, game_history)
                self.add_to_replay_buffer(game_history, self.game_result_reward(winner))

    def play_single_game(self):
        """
        Play a game of Othello
        When playing a game we make first exploration/exploitation move
        and then finish up episode playing according to the training player
        and opponent strategies
        """
        game_state = GameState()
        game_history = []  # List of (move_info, game_state) tuples
        while not game_state.game_over:
            game_state, move_info = self.make_training_move(game_state)
            game_history.append((move_info, game_state))
            winner = self.play_episode(game_state, game_history)
            self.add_to_replay_buffer(game_history, self.game_result_reward(winner))

    def play_episode(self, game_state, game_history):
        """
            Play an episode of the game till the end.
        Args:
            game_state: current game state
            game_history: array where the game history will be stored
        Return:
            Player: The winner of the game
        """
        while not game_state.game_over:
            # move = self.agents[game_state.current_player].get_best_move(game_state)
            if game_state.current_player == self.train_agent.player:
                move = self.choose_action(game_state, override_exploration=False)
                move_info = game_state.make_move(move)
                game_history.append((move_info, game_state.clone()))
            else:
                move = self.agents[game_state.current_player].get_best_move(game_state)
                game_state.make_move(move)
        self.episode += 1
        # self.print_board(game_state)
        return game_state.winner

    def add_to_replay_buffer(self, game_history, final_reward):
        """
            Add the game history to the replay buffer with the final reward.
        Args:
            game_history: the list of (move, game_state) tuples
            final_reward: reward received after game is over
        """
        reward = final_reward
        done = 1
        next_state = None
        for move, game_state in reversed(game_history):
            self.replay_buffer.push(game_state_one_hot_encode(game_state),
                                    position_one_hot_encode(move.position),
                                    reward,
                                    game_state_one_hot_encode(next_state),
                                    done)
            reward = reward * self.reward_decay
            done = 0
            next_state = game_state

    def make_training_move(self, game_state):
        """
            Make a move for the training agent. If the current player is the training agent, choose an action
            based on the epsilon-greedy policy. If the current player is the opponent, choose the best move based
            on the opponent's strategy.
        Args:
            game_state: current game state
        Return:
            (game state before the move, move_info): game state before the move and the move information
        """
        move_info = None
        game_state_before = None
        for i in range(2):
            if not game_state.game_over:
                if game_state.current_player == self.train_agent.player:
                    game_state_before = game_state.clone()
                    move = self.choose_action(game_state)
                    move_info = game_state.make_move(move)
                else:
                    move = self.agents[game_state.current_player].get_best_move(game_state)
                    game_state.make_move(move)
        return game_state_before, move_info

    def choose_action(self, game_state, override_exploration=False):
        """
            Choose an action based on the epsilon-greedy policy.
        Args:
            game_state: current game_state
            override_exploration: hint to override exploration and choose exploitation
        Return:
            Position: The chosen action
        """
        legal_moves = list(game_state.legal_moves.keys())
        is_exploration = self.is_exploration & (not override_exploration)
        if is_exploration:
            # Exploration with preference to unexplored moves
            if legal_moves is None:
                return SkipPosition()
            return random.choice(legal_moves)
        else:
            # Exploitation
            with torch.no_grad():
                if legal_moves is None:
                    return SkipPosition()
                if isinstance(legal_moves[0], SkipPosition):
                    return SkipPosition()
                q_values = self.model(torch.tensor(game_state_one_hot_encode(game_state), dtype=torch.float32).unsqueeze(0))
                best_move = action_decode(q_values[0].argmax(axis=0).item())
                if best_move not in legal_moves:
                    self.epoch_illegal_moves += 1
                # Select best move based on highest q-value taking into consideration only legal moves
                legal_q_values = {move: q_values[0][one_hot_encoding_to_idx(position_one_hot_encode(move))].item() for move in legal_moves}
                best_move = max(legal_q_values, key=legal_q_values.get)
                return best_move

    def train_model(self, replay_buffer):
        """
        Train the DQN model using a minibatch of replay buffer samples.

        Args:
            replay_buffer:
        """
        for epoch in range(self.epochs):
            # Sample minibatch
            states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)

            # Convert to tensors
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Compute Q-values
            f = lambda x: one_hot_encoding_to_idx(x)
            actions_unsqueeze = torch.tensor(actions.unsqueeze(1), dtype=torch.long).apply_(f)
            q_values = self.model(states).gather(1, actions_unsqueeze).squeeze(1)

            # # Next state q value Single Q learning
            # next_q_values = self.target_model(next_states).max(1)[0]

            # Next state q value Double Q learning
            next_q_values_online = self.model(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_model(next_states).gather(1, next_q_values_online).squeeze(1)

            # Compute targets using Bellmans equation
            # For the end states we don't add the discounted future rewards
            targets = rewards + (1 - dones) * self.discount_factor * next_q_values

            # During training, after computing Q-values
            with torch.no_grad():
                old_q_values = self.model(states).gather(1, actions_unsqueeze).squeeze(1)  # Shape: [B]
            # Calculate Q-value changes
            q_value_changes = torch.abs(old_q_values - targets)
            # Average Q-value change
            avg_q_value_change = q_value_changes.mean().item()
            self.recent_avg_reward = avg_q_value_change
            # Log this for analysis
            self.epoch_q_value_changes.append(avg_q_value_change)

            # Compute loss
            loss = self.loss_fn(q_values, targets)

            # print("Q-values:", q_values)
            # print("Targets:", targets)
            # print("Loss:", loss.item())

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update learning rate
            self.scheduler.step()
            # current_lr = self.optimizer.param_groups[0]['lr']
            # print(f"Current Learning Rate: {current_lr}")

        # self.replay_buffer.purge()

    def game_result_reward(self, winner):
        if winner == Player.NONE:
            return DRAW_VALUE
        elif winner == self.train_agent.player:
            return WIN_VALUE
        else:
            return LOSS_VALUE

    def print_board(self, game_state):
        for row in game_state.board:
            print(' '.join('B' if cell == Player.BLACK else 'W' if cell == Player.WHITE else '.' for cell in row))
        print("\n")

    def print_stats(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].plot(self.epoch_q_value_changes)
        axs[0, 0].set_title("Average Q-value Change Per Epoch")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Average Q-value Change")

        axs[0, 1].plot(self.illegal_moves)
        axs[0, 1].set_title("Illegal Moves")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Volume of Illegal Moves")

        axs[1, 0].plot(self.epsilon_history)
        axs[1, 0].set_title("Epsilon History")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].set_ylabel("Epsilon")

        # axs[1, 1].plot(self.learning_rates)
        # axs[1, 1].set_title("Learning Rates History")
        # axs[1, 1].set_xlabel("Epoch")
        # axs[1, 1].set_ylabel("Learning Rate")

        plt.tight_layout()
        plt.show()


    def update_epsilon(self, current_step):
        """
        Update epsilon

        Args:
           current_step (int): Current step or episode number.

        Returns:
            float: Updated epsilon value.
        """
        if (current_step + 1) % (self.total_games / 10) == 0:
            self.epsilon = max(self.epsilon_min, max(0.0, self.epsilon - 0.1))
        self.epsilon_history.append(self.epsilon)
        return self.epsilon

    def update_epsilon_boltzmann(self, current_step, decay_rate=0.0000000001):
        """
        Update epsilon using Boltzmann exploration policy.

        Args:
            min_epsilon (float): Minimum value of epsilon.
            decay_rate (float): Decay rate for epsilon.
            current_step (int): Current step or episode number.

        Returns:
            float: Updated epsilon value.
        """
        temperature = self.epsilon * math.exp(-decay_rate * current_step)
        self.epsilon = max(self.epsilon_min, temperature)
        self.epsilon_history.append(self.epsilon)
        return self.epsilon

    def update_epsilon_adaptive(self, threshold=0.2):
        if self.recent_avg_reward < threshold:
            self.epsilon = min(1.0, self.epsilon * 1.001)  # Slightly increase exploration
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

    def rotate_opponent_agents(self):
        self.opponent_agents = self.opponent_agents[1:] + [self.opponent_agents[0]]
        self.agents = {
            Player.BLACK: self.train_agent if self.train_agent.player == Player.BLACK else self.opponent_agents[0],
            Player.WHITE: self.opponent_agents[0] if self.opponent_agents[0].player == Player.WHITE else self.train_agent
        }