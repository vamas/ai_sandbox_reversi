# Train qtable agent against random agent
import copy
from collections import defaultdict
import random
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from v1.dqn import DQN
from v1.dqn_agent import DQNAgent
from v1.dqn_helpers import (game_state_one_hot_encode,
                            action_decode, action_encode,
                            legal_moves_mask)
from v1.dqn_replaybuffer import ReplayBuffer
from v1.dqn_replaybuffer_prioritized import PrioritizedReplayBuffer
from v1.gamestate import GameState, BOARD_SHAPE
from v1.moveinfo import MoveInfo
from v1.player import Player, opponent
from v1.position import Position, SkipPosition
from v1.random_agent import RandomAgent
from v1.gamemanager import GameManager
from infrastructure.metric_logger import training_stats, chart_colors

WIN_VALUE = 1.0
DRAW_VALUE = 0.6
LOSS_VALUE = 0.0
ILLEGAL_MOVE_LOSS_VALUE = 0.0

DEFAULT_Q_VALUE = 1.0

def play_test_game(training_agent, testing_agent, board, current_player):
    """
    Play a game of Othello
    When playing a game we make first exploration/exploitation move
    and then finish up episode playing according to the training player
    and opponent strategies
    """

    def play_test_game_turn():
        agents = [training_agent, testing_agent]
        agents.sort(key=lambda x: x.player == current_player, reverse=True)
        for agent in agents:
            if not game_state.game_over:
                move = agent.get_best_move(game_state)
                game_state.make_move(move)

    game_state = GameState(board=copy.deepcopy(board))
    while not game_state.game_over:
        play_test_game_turn()
    return game_state.winner

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
                 opponent_agents=[],
                 testing_agents=[RandomAgent(Player.WHITE)],
                 reward_decay=0.9,
                 memory_size=1000,
                 batch_size=64,
                 epsilon_min=0.1,
                 self_instances=1):
        """
        Initialize the Deep Q-Network.

        Args:
            input_dim (int): Number of input features (e.g., board size: 8x8 = 64).
            output_dim (int): Number of possible actions (legal moves).
            hidden_dim (int): Number of units in the hidden layers.
        """
        self.learner_name = "Othello DQN"
        self.total_games = total_games
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.opponent_agents = opponent_agents
        self.reward_decay = reward_decay
        self.input_dim = BOARD_SHAPE * BOARD_SHAPE + BOARD_SHAPE * BOARD_SHAPE + BOARD_SHAPE * BOARD_SHAPE
        self.output_dim = BOARD_SHAPE * BOARD_SHAPE + 1
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.target_model = None
        self.target_update_freq = 1000 # Update target q-network every other 1000 steps (played games)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = np.exp(np.log(epsilon_min / epsilon) / total_games)
        self.self_instances = self_instances
        self.testing_agents = testing_agents
        self.testing_stats = {e.name:[] for e in testing_agents}
        self.epsilon_max = epsilon
        self.epsilon_decay_rate = np.log(self.epsilon_min / self.epsilon) / self.total_games
        self.training_agent = RandomAgent(Player.BLACK)
        self.scheduler = None
        self.initialize_model()
        self.add_trained_model_to_opponents(self.target_model)
        self.active_player = Player.BLACK
        self.run_id = str(uuid.uuid4())
        self.epoch = 0
        self.episode = 0
        self.replay_buffer = ReplayBuffer(self.memory_size)
        self.is_exploration = True
        training_stats["learner"] = "Othello DQN"
        training_stats["color"] = random.choice(chart_colors)

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

        self.model = DQN(input_dim, output_dim, self.hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=int(self.total_games/10), gamma=0.85)

        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.target_model = DQN(input_dim, output_dim, self.hidden_dim)
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize with same weights

        self.loss_fn = nn.SmoothL1Loss()

    def train_dqn(self):
        """
        Train the DQN model.

        Return:
            model (DQN): The trained DQN model.
        """

        for game in tqdm(range(self.total_games), desc="Training DQN"):

            # Play a training game
            self.init_game(game, self.training_agent.player)
            self.play_game()

            # Update epsilon
            self.update_epsilon_boltzmann(game)

            # Update exploration flag
            self.is_exploration = random.random() < self.epsilon

            # Replay buffer is ready to sample
            if self.replay_buffer.is_buffer_ready:
                self.train_model(self.replay_buffer)

            # Update target model
            if game % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            self.test_model(game)
            self.epoch += 1
            training_stats["epoch"] = self.epoch

        return self.model

    def init_game(self, game, training_agent_color=Player.BLACK):
        self.active_player = opponent(self.active_player)

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

    def play_single_game(self):
        """
        Play a game of Othello
        When playing a game we make first exploration/exploitation move
        and then finish up episode playing according to the training player
        and opponent strategies
        """
        game_state = GameState()
        game_history = []
        winner, illegal_move = self.play_episode(game_state, game_history)
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
        next_state = None
        for move, game_state in reversed(game_history):
            self.replay_buffer.push(game_state_one_hot_encode(game_state, self.training_agent.player),
                                    action_encode(move.position),
                                    reward,
                                    game_state_one_hot_encode(next_state, self.training_agent.player),
                                    done,
                                    legal_moves_mask(game_state.legal_moves_list))
            reward = reward * self.reward_decay
            done = 0
            next_state = game_state

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
                    move = self.get_opponents_best_move(game_state, player)
                    game_state.make_move(move)
        move_info = move_info if move_info is not None else MoveInfo(self.active_player, SkipPosition(), [])
        assert move_info.player == self.active_player
        return game_state_before, move_info

    def get_opponents_best_move(self, game_state, player):
        opponent_moves = defaultdict(lambda: 0)
        for agent in self.opponent_agents:
            agent.player = player
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
        legal_moves = list(game_state.legal_moves_list)
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
                q_values = self.model(torch.tensor(game_state_one_hot_encode(game_state, game_state.current_player),
                                                   dtype=torch.float32).unsqueeze(0))
                q_values[torch.tensor(legal_moves_mask(game_state.legal_moves_list), dtype=torch.float32).unsqueeze(0) == 0] = -float("inf")
                best_move = action_decode(q_values[0].argmax(axis=0).item())
                return best_move

    def train_model(self, replay_buffer):
        """
        Train the DQN model using a minibatch of replay buffer samples.

        Args:
            replay_buffer:
        """
        # Sample minibatch
        # Simple replay buffer
        states, actions, rewards, next_states, dones, valid_moves = replay_buffer.sample(self.batch_size)

        # # Prioritized replay buffer
        # beta = 0.4  # Compensation factor for importance sampling
        # batch, indices, weights = replay_buffer.sample(self.batch_size, beta)
        # states, actions, rewards, next_states, dones, valid_moves = zip(*batch)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        valid_moves = torch.tensor(valid_moves, dtype=torch.float32)

        # Compute Q-values
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # # Next state q value Single Q learning
        # next_q_values = self.target_model(next_states)
        # next_q_values[valid_moves == 0] = -float("inf")  # Mask invalid actions
        # next_q_values = next_q_values.max(1)[0]

        # Next state q value Double Q learning
        next_q_values_online = self.model(next_states)
        next_q_values_online[valid_moves == 0] = -float("inf")  # Mask invalid actions
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

        # self.replay_buffer.purge()

    def game_result_reward(self, winner, illegal_move=False):
        if winner == Player.NONE:
            return DRAW_VALUE
        elif winner == self.training_agent.player:
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

    def add_trained_model_to_opponents(self, model):
        if model is not None:
            self.opponent_agents.extend([DQNAgent(Player.WHITE, model, "Self")] * self.self_instances)

    def test_model(self, game, freq=1000, total_games=15):
        def play_random_game(agent, player):
            agent.player = opponent(player)
            agent_black = DQNAgent(player, self.target_model) if player == Player.BLACK else agent
            agent_white = DQNAgent(player, self.target_model) if player == Player.WHITE else agent
            game_mgr = GameManager(agent_black, agent_white)
            return game_mgr.run()

        if game % freq == 0:
            wins = 0
            for testing_agent in self.testing_agents:
                for i in range(total_games):
                    wins = wins + (play_random_game(testing_agent, Player.BLACK) == Player.BLACK)
                    wins = wins + (play_random_game(testing_agent, Player.WHITE) == Player.WHITE)
            score = wins / (total_games * 2 * len(self.testing_agents))
            training_stats["score"] = score


