# Train qtable agent against random agent
import copy
import time
from collections import defaultdict, deque
import random
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from prompt_toolkit.contrib.telnet import TelnetServer
from torch.optim import lr_scheduler
from tqdm import tqdm

from v1.dqn import DQN
from v1.dqn_agent import DQNAgent
from v1.dqn_helpers import (board_one_hot_encode,
                            action_decode, action_encode,
                            legal_moves_mask, get_symmetrical_states)
from v1.dqn_replaybuffer import ReplayBuffer
from v1.dqn_replaybuffer_prioritized import PrioritizedReplayBuffer
from v1.gamestate import GameState, BOARD_SHAPE
from v1.minimax_agent import MinimaxAgent
from v1.moveinfo import MoveInfo
from v1.player import Player, opponent
from v1.position import Position, SkipPosition
from v1.random_agent import RandomAgent
from v1.gamemanager import GameManager
from infrastructure.metric_logger import training_stats, chart_colors
from v1.scoring import ELOSystem, DEFAULT_SCORE

WIN_VALUE = 1.0
DRAW_VALUE = 0.6
LOSS_VALUE = 0.0
ILLEGAL_MOVE_LOSS_VALUE = 0.0

DEFAULT_Q_VALUE = 1.0

# Best - 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)  # For MPS on Mac
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # For CUDA

# - Play game
# - Make first training move
#   - Play episode till the end
#   - Enrich history with reward values base on the final reward
#   - Append history to the replay buffer
#   - Check if replay buffer is full, if so, sample batch and train the model
#   - Continue loop

class DQNTrain:

    def __init__(self,
                 torch_device=None,
                 hidden_dim=256,
                 total_games=100,
                 learning_rate=0.4,
                 discount_factor=1.0,
                 epsilon=0.7,
                 reward_decay=0.9,
                 memory_size=1000,
                 batch_size=64,
                 epsilon_min=0.1,
                 self_instances=1,
                 pre_trained_model_path=None):
        """
        Initialize the Deep Q-Network.

        Args:
            input_dim (int): Number of input features (e.g., board size: 8x8 = 64).
            output_dim (int): Number of possible actions (legal moves).
            hidden_dim (int): Number of units in the hidden layers.
        """
        self.learner_name = "Othello DQN"

        training_stats["learner"] = self.learner_name
        training_stats["color"] = random.choice(chart_colors)
        training_stats["score"] = DEFAULT_SCORE

        self.total_games = total_games
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.reward_decay = reward_decay
        self.input_dim = BOARD_SHAPE * BOARD_SHAPE + BOARD_SHAPE * BOARD_SHAPE
        self.output_dim = BOARD_SHAPE * BOARD_SHAPE + 1
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.target_model = None
        self.torch_device = torch_device if torch_device is not None else torch.device("cpu")
        self.epsilon_min = epsilon_min
        self.epsilon_decay = np.exp(np.log(epsilon_min / epsilon) / total_games)
        self.self_instances = self_instances
        self.epsilon_max = epsilon
        self.epsilon_decay_rate = np.log(self.epsilon_min / self.epsilon) / self.total_games
        self.scheduler = None
        self.initialize_model(pre_trained_model_path)
        self.active_player = Player.BLACK
        self.run_id = str(uuid.uuid4())
        self.epoch = 0
        self.episode = 0
        self.replay_buffer = ReplayBuffer(self.memory_size)
        self.is_exploration = True
        self.scorer = ELOSystem()
        self.target_model_update_freq = 1000  # Update target q-network every other 1000 steps (played games)

        # self.opponent_update_freq = 10000
        self.opponent_agents = deque(maxlen=1)
        opponent_agent = RandomAgent(Player.WHITE)
        if pre_trained_model_path != "":
            opponent_model = DQN(BOARD_SHAPE * BOARD_SHAPE + BOARD_SHAPE * BOARD_SHAPE, BOARD_SHAPE * BOARD_SHAPE + 1, hidden_dim)
            opponent_model.load_state_dict(torch.load(pre_trained_model_path, weights_only=True))
            opponent_agent = DQNAgent(Player.WHITE, opponent_model, torch_device=self.torch_device, name="Opponent")
        self.opponent_agents.append(opponent_agent)

        # self.scoring_agents_update_freq = 10000
        self.scoring_freq = 1000
        self.scoring_agents = deque(maxlen=1)
        scoring_agent = MinimaxAgent(Player.WHITE, max_depth=0)
        if pre_trained_model_path != "":
            scoring_model = DQN(BOARD_SHAPE * BOARD_SHAPE + BOARD_SHAPE * BOARD_SHAPE, BOARD_SHAPE * BOARD_SHAPE + 1, hidden_dim)
            scoring_model.load_state_dict(torch.load(pre_trained_model_path, weights_only=True))
            scoring_agent = DQNAgent(Player.WHITE, scoring_model, torch_device=self.torch_device, name="Scoring")
        self.scoring_agents.append(scoring_agent)

    def initialize_model(self, pre_trained_model_path):
        """
            Initialize the DQN model, optimizer, and loss function.

            Args:
                pre_trained_model_path: pre-trained model weights

            Returns:
                model (DQN): The initialized DQN model.
                optimizer (torch.optim.Optimizer): Optimizer for training the model.
                loss_fn (nn.Module): Loss function for DQN.
            """
        set_seed()

        self.model = DQN(self.input_dim, self.output_dim, self.hidden_dim)
        if pre_trained_model_path != "":
            # init model from pre-trained model
            self.model.load_state_dict(torch.load(pre_trained_model_path, weights_only=True))
        else:
            # init model with xavier uniform weights and zeros for bias
            for layer in self.model.children():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=int(self.total_games/100), gamma=0.91)

        self.target_model = DQN(self.input_dim, self.output_dim, self.hidden_dim)
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target model with same weights
        self.model.to(self.torch_device)
        self.target_model.to(self.torch_device)

        self.loss_fn = nn.SmoothL1Loss()

    def train_dqn(self):
        """
        Train the DQN model.

        Return:
            model (DQN): The trained DQN model.
        """

        for game in tqdm(range(self.total_games), desc="Training DQN"):

            # Play a training game
            self.init_game()
            self.play_game()

            # Update epsilon
            self.update_epsilon_boltzmann(game)

            # Update exploration flag
            self.is_exploration = random.random() < self.epsilon

            # Update target model
            if game % self.target_model_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            self.test_model(game)
            self.epoch += 1
            training_stats["epoch"] = self.epoch

                # # Freeze model in history to test against it
                # if game % self.scoring_agents_update_freq == 0 and game > 0:
                #     self.scoring_agents.append(DQNAgent(Player.WHITE, self.target_model, torch_device=self.torch_device,
                #                  name="Self{}".format(game)))

                # # Update opponent agents
                # if game % self.opponent_update_freq == 0 and game > 0:
                #     self.opponent_agents.append(
                #         DQNAgent(Player.WHITE, self.target_model, torch_device=self.torch_device,
                #                  name="Self{}".format(game)))

        return self.model

    def init_game(self, bw_ratio=0.7):
        # Based on BLACK vs WHITE training games ration
        self.active_player = Player.BLACK if random.random() < bw_ratio else Player.WHITE
        # self.active_player = opponent(self.active_player)

    def play_game(self):
        """
        Play a game of Othello
        When playing a game we make first exploration/exploitation move
        and then finish up episode playing according to the training player
        and opponent strategies
        """
        game_state = GameState()
        while not game_state.game_over:
            game_state_before, move_info = self.play_turn(game_state)
            game_history = [] # List of (move_info, game_state) tuples
            assert move_info.player == self.active_player
            if move_info is not None:
                game_history.append((move_info, game_state_before))
                game_episode_state = game_state.clone()
                winner, illegal_move = self.play_episode(game_episode_state, game_history)
                self.add_to_replay_buffer(game_history, self.game_result_reward(winner, illegal_move))

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
            game_state_before, move_info = self.play_turn(game_state)
            assert move_info.player == self.active_player
            game_history.append((move_info, game_state_before))
        self.episode += 1
        return game_state.winner, game_state.illegal_move

    def add_to_replay_buffer(self, game_history, final_reward):
        """
            Add the game history to the replay buffer with the final reward.
        Args:
            game_history: the list of (move, game_state) tuples
            final_reward: reward received after game is over
        """
        reward = final_reward
        done = 1
        next_state_board = None
        for move, game_state in reversed(game_history):
            for symmetrical_board in get_symmetrical_states(game_state.board):
                for next_state_symmetrical_board in get_symmetrical_states(next_state_board):
                    self.replay_buffer.push(board_one_hot_encode(symmetrical_board, game_state.current_player),
                                            action_encode(move.position),
                                            reward,
                                            board_one_hot_encode(next_state_symmetrical_board,
                                                                 game_state.current_player),
                                            done,
                                            legal_moves_mask(game_state.legal_moves_list))
            reward = reward * self.reward_decay
            done = 0
            next_state_board = game_state.board

            # train if replay buffer is full
            if self.replay_buffer.is_buffer_ready:
                self.train_model(self.replay_buffer)
                self.replay_buffer.purge()

    def play_turn(self, game_state):
        """
            Play a turn e.g. BLACK then WHITE player.
            Capture the game state before training player moves and return move_info of training player.
            Take a "screenshot" of the game state and store it in the testing games for further model assesment.

            What if training player is WHITE and we encounter end of the game before it moves? What we return?
        Args:
            game_state: current game state
        Return:
            GameState: The game state before training player moves
            MoveInfo: The move info of the training player
        """
        move_info = None
        game_state_before = game_state.clone()
        for player in [Player.BLACK, Player.WHITE]:
            if not game_state.game_over:
                if player == self.active_player:
                    game_state_before = game_state.clone()
                    move = self.choose_action(game_state)
                    move_info = game_state.make_move(move)
                else:
                    move = self.get_opponents_best_move(game_state)
                    game_state.make_move(move)
        move_info = move_info if move_info is not None else MoveInfo(self.active_player, SkipPosition(), [])
        assert move_info.player == self.active_player
        return game_state_before, move_info

    def get_opponents_best_move(self, game_state):
        """
            Choose the best move for the all opponents.
        Args:
            game_state: current game_state
        Return:
            Position: The chosen action
        """
        opponent_moves = defaultdict(lambda: 0)
        for agent in self.opponent_agents:
            agent.player = opponent(self.active_player)
            move = agent.get_best_move(game_state)
            opponent_moves[move] = opponent_moves[move] + 1
        return max(opponent_moves, key=opponent_moves.get)

    def choose_action(self, game_state, override_exploration=False):
        """
            Choose an action based on the epsilon-greedy policy.
        Args:
            game_state: current game_state
            override_exploration: hint to override exploration and choose exploitation
        Return:
            Position: The chosen action
        """
        if len(game_state.legal_moves_list) == 1:
            return game_state.legal_moves_list[0]
        if self.is_exploration:
            # Exploration with preference to unexplored moves
            return random.choice(game_state.legal_moves_list)
        else:
            return self.choose_exploitation_action(game_state)

    def choose_exploitation_action(self, game_state):
        """
            Choose best action from the Q-values network.
        Args:
            game_state: current game_state
        Return:
            Position: The chosen action
        """
        with torch.no_grad():
            q_values = self.model(torch.tensor(board_one_hot_encode(game_state.board, game_state.current_player),
                                               dtype=torch.float32).to(self.torch_device).unsqueeze(0))
            q_values[torch.tensor(legal_moves_mask(game_state.legal_moves_list),dtype=torch.float32).
                     to(self.torch_device).unsqueeze(0) == 0] = LOSS_VALUE  # -float("inf")
            best_move = action_decode(q_values[0].argmax(axis=0).item())
            return best_move

    def train_model(self, replay_buffer):
        """
        Train the DQN model using a minibatch of replay buffer samples.

        Args:
            replay_buffer:
        """

        start_time = time.time()

        # Sample minibatch
        # Simple replay buffer
        states, actions, rewards, next_states, dones, valid_moves = replay_buffer.sample(self.batch_size)

        # # Prioritized replay buffer
        # beta = 0.4  # Compensation factor for importance sampling
        # batch, indices, weights = replay_buffer.sample(self.batch_size, beta)
        # states, actions, rewards, next_states, dones, valid_moves = zip(*batch)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.torch_device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.torch_device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.torch_device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.torch_device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.torch_device)
        valid_moves = torch.tensor(valid_moves, dtype=torch.float32).to(self.torch_device)

        # Compute Q-values
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # # Next state q value Single Q learning
        # next_q_values = self.target_model(next_states)
        # next_q_values[valid_moves == 0] = LOSS_VALUE # -float("inf")  # Mask invalid actions
        # next_q_values = next_q_values.max(1)[0]

        # Next state q value Double Q learning
        next_q_values_online = self.model(next_states)
        next_q_values_online[valid_moves == 0] = LOSS_VALUE # -float("inf")  # Mask invalid actions
        next_q_values = self.target_model(next_states).gather(1, next_q_values_online.argmax(1).unsqueeze(1)).squeeze(1)

        # Compute targets using Bellmans equation
        # For the end states we don't add the discounted future rewards
        targets = rewards + (1 - dones) * self.discount_factor * next_q_values

        # During training, after computing Q-values
        with torch.no_grad():
            old_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: [B]
        # Calculate Q-value changes
        q_value_changes = torch.abs(old_q_values - targets)
        # Average Q-value change
        avg_q_value_change = q_value_changes.mean().item()
        # Log this for analysis
        training_stats["avg_q_value_change"] = avg_q_value_change

        # Compute loss
        loss = self.loss_fn(q_values, targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update learning rate
        self.scheduler.step()
        training_stats["learning_rate"] = self.optimizer.param_groups[0]['lr']

        episode_time = time.time() - start_time
        training_stats["training_time"] = episode_time
        # self.replay_buffer.purge()

    def game_result_reward(self, winner, illegal_move=False):
        if winner == Player.NONE:
            return DRAW_VALUE
        elif winner == self.active_player:
            return WIN_VALUE
        else:
            if illegal_move:
                return ILLEGAL_MOVE_LOSS_VALUE
            return LOSS_VALUE

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
        training_stats["epsilon"] = self.epsilon
        return self.epsilon

    def update_epsilon_boltzmann(self, current_step):
        """
        Update epsilon using Boltzmann exploration policy.

        Args:
            min_epsilon (float): Minimum value of epsilon.
            decay_rate (float): Decay rate for epsilon.
            current_step (int): Current step or episode number.

        Returns:
            float: Updated epsilon value.
        """
        self.epsilon = (self.epsilon_min + (self.epsilon_max - self.epsilon_min) *
                        np.exp(self.epsilon_decay_rate * current_step))
        training_stats["epsilon"] = self.epsilon

    def test_model(self, game, total_games=10):

        def play_test_game():
            game_mgr = GameManager(black_agent, white_agent)
            return game_mgr.run()

        if game % self.scoring_freq == 0:
            for scoring_agent in self.scoring_agents:
                # wins = 0
                # Score against earlier versions of self
                for i in range(total_games):

                    # Score as BLACK
                    black_agent = DQNAgent(Player.BLACK, self.target_model,
                                              torch_device=self.torch_device, name="self_b")
                    white_agent = scoring_agent
                    white_agent.player = Player.WHITE
                    winner = play_test_game()
                    self.scorer.update_ratings("self", "scoring_model",
                                               result_a=(winner == Player.BLACK))
                    # wins = wins + (winner == Player.BLACK)

                    # Score as WHITE
                    white_agent = DQNAgent(Player.WHITE, self.target_model,
                                             torch_device=self.torch_device, name="self_w")
                    black_agent = scoring_agent
                    black_agent.player = Player.BLACK
                    winner = play_test_game()
                    self.scorer.update_ratings("self", "scoring_model",
                                               result_a=(winner == Player.WHITE))
                    # wins = wins + (winner == Player.WHITE)


                training_stats["score"] = float(self.scorer.ratings["self"])
                # training_stats["score"] = wins / (total_games * 2)


