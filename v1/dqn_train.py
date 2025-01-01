# Train qtable agent against random agent
import math
from idlelib.pyparse import trans
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from v1.dqn import DQN
from v1.dqn_replaybuffer import ReplayBuffer
from v1.gamestate import GameState, print_board
from v1.player import Player, opponent
from v1.position import Position, SkipPosition
from v1.random_agent import RandomAgent

WIN_VALUE = 1.0
DRAW_VALUE = 0.0
LOSS_VALUE = -1.0

DEFAULT_Q_VALUE = -1.0

BOARD_SHAPE = 8

def game_state_encode(game_state, shape=BOARD_SHAPE):
    if game_state is None:
        return np.zeros(shape * shape, dtype=float)
    state = np.zeros(game_state.Rows * game_state.Cols, dtype=float)
    for row in range(game_state.Rows):
        for col in range(game_state.Cols):
            idx = row * game_state.Cols + col
            if game_state.board[row][col] == Player.BLACK:
                state[idx] = 1
            elif game_state.board[row][col] == Player.WHITE:
                state[idx] = -1
    return state.flatten()

def action_encode(move_info, shape=BOARD_SHAPE):
    if move_info.position.row == -1 and move_info.position.col == -1:
        return shape*shape
    return shape * move_info.position.row + move_info.position.col

def position_encode(position, shape=BOARD_SHAPE):
    if position.row == -1 and position.col == -1:
        return shape*shape
    return shape * position.row + position.col

def action_decode(action, shape=BOARD_SHAPE):
    if action == shape*shape:
        return SkipPosition()
    return Position(action // shape, action % shape)

# - Play game
# - Make first training move
#   - Play episode till the end
#   - Enrich history with reward values base on the final reward
#   - Append history to the replay buffer
#   - Check if replay buffer is full, if so, sample batch and train the model
#   - Continue loop

class DQNTrain:

    def __init__(self,
                 hidden_dim=128,
                 total_games=100,
                 learning_rate=0.4,
                 discount_factor=1.0,
                 epsilon=0.7,
                 train_agent=RandomAgent(Player.BLACK),
                 opponent_agent=RandomAgent(Player.WHITE),
                 reward_decay=0.9,
                 memory_size=10000,
                 batch_size=512,
                 model=None):
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
        self.agents = {
            Player.BLACK: train_agent if train_agent.player == Player.BLACK else opponent_agent,
            Player.WHITE: opponent_agent if opponent_agent.player == Player.WHITE else train_agent
        }
        self.train_agent = train_agent
        self.episode = 0
        self.total_rewards = []
        self.avg_q_values = []
        self.reward_decay = reward_decay
        self.input_dim = BOARD_SHAPE * BOARD_SHAPE
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
        loss_fn = nn.MSELoss()

        return model, optimizer, loss_fn

    def train_dqn(self):
        """
        Train the DQN model.

        Return:
            model (DQN): The trained DQN model.
        """
        if self.model is None:
            self.model, self.optimizer, self.loss_fn = self.initialize_model()
        else:
            _, self.optimizer, self.loss_fn = self.initialize_model()
        self.target_model, _, _ = self.initialize_model()
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize with same weights
        self.replay_buffer = ReplayBuffer(self.memory_size)
        self.epoch_q_value_changes = []
        epsilon = self.epsilon
        self.is_exploration = True
        for epoch in range(1):
            self.epsilon = epsilon
            self.episode = 0
            self.total_rewards = []
            self.avg_q_values = []
            for game in range(self.total_games):
                print("Game/Total games {}/{}".format(game + 1, self.total_games))
                self.play_game()
                self.update_epsilon_boltzmann(game)
                if not self.is_exploration:
                    # if len(self.replay_buffer) > self.batch_size:
                    #     self.train_model(self.replay_buffer)
                    if self.replay_buffer.is_buffer_ready:
                        print("Replay buffer is ready. Start training. Size: {}".format(self.replay_buffer.data_points))
                        self.train_model(self.replay_buffer)
                    if game % self.target_update_freq == 0:
                        self.target_model.load_state_dict(self.model.state_dict())
                if random.random() < self.epsilon:
                    self.is_exploration = True
                else:
                    self.is_exploration = False
                game += 1
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
            self.replay_buffer.push(game_state_encode(game_state),
                                    action_encode(move),
                                    reward,
                                    game_state_encode(next_state),
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
                legal_moves_idx = [position_encode(move) for move in legal_moves]
                q_values = self.model(torch.tensor(game_state_encode(game_state), dtype=torch.float32).unsqueeze(0))
                pos = q_values[0][:(game_state.Cols*game_state.Rows)][legal_moves_idx].argmax(axis=0).item()
                return legal_moves[pos]

    def train_model(self, replay_buffer):
        """
        Train the DQN model using a minibatch of replay buffer samples.

        Args:
            replay_buffer:
        """
        print("Training model")

        # Sample minibatch
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute Q-values and targets
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.discount_factor * next_q_values

        # During training, after computing Q-values
        with torch.no_grad():
            old_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: [B]
        # Calculate Q-value changes
        q_value_changes = torch.abs(old_q_values - targets)
        # Average Q-value change
        avg_q_value_change = q_value_changes.mean().item()
        # Log this for analysis
        self.epoch_q_value_changes.append(avg_q_value_change)

        # Compute loss
        loss = self.loss_fn(q_values, targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
        plt.plot(self.epoch_q_value_changes)
        plt.title("Average Q-value Change Per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Average Q-value Change")
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
            self.epsilon = max(0, self.epsilon - 0.1)
        return self.epsilon

    def update_epsilon_boltzmann(self, current_step, decay_rate=0.001, min_epsilon=0.00001):
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
        self.epsilon = max(min_epsilon, temperature)
        return self.epsilon