# Train qtable agent against random agent
from idlelib.pyparse import trans
import random
import matplotlib.pyplot as plt
from sympy.printing.precedence import precedence

from v1.agent import AgentType
from v1.gamestate import GameState, print_board
from v1.player import Player, opponent
from v1.position import Position, SkipPosition
from v1.qtable_single import SingleQTable
from v1.qtable_double import DoubleQTable
from v1.random_agent import RandomAgent

WIN_VALUE = 1.0
DRAW_VALUE = -0.1
LOSS_VALUE = -1.0

DEFAULT_Q_VALUE = 0.0

cost_matrix = [
    [1000,100,100,1000],
    [100,1,1,100],
    [100,1,1,100],
    [1000,100,100,1000]]

# cost_matrix = [
#     [100,-25,10,5,5,10,-25,100],
#     [-25,-50,5,1,1,5,-50,-25],
#     [10,5,1,1,1,1,5,10],
#     [5,1,1,0,0,1,1,5],
#     [5,1,1,0,0,1,1,5],
#     [10,5,1,1,1,1,5,10],
#     [-25,-50,5,1,1,5,-50,-25],
#     [100,-25,10,5,5,10,-25,100]]

def game_state_hash(game_state):
    # # State aggregation to reduce the state space
    # state_representation = 0
    # for row in range(game_state.Rows):
    #     for column in range(game_state.Cols):
    #         if game_state.board[row][column] == game_state.current_player:
    #             state_representation += cost_matrix[row][column]
    #         elif game_state.board[row][column] == opponent(game_state.current_player):
    #             state_representation -= cost_matrix[row][column]
    # return game_state.current_player.__str__() + '_' + str(state_representation)
    return game_state.hash()

def calculate_q_value(learning_rate, discount_factor, reward,
                        current_q_value, max_next_state_q_value):
    # Q(s, a) = (1 - alpha) * Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
    return ((1 - learning_rate) * current_q_value +
            learning_rate * (reward + discount_factor * max_next_state_q_value))

class QTableTrain:

    def __init__(self,
                 qtable,
                 total_games=100,
                 learning_rate=0.4,
                 discount_factor=1.0,
                 epsilon=0.7,
                 train_agent=RandomAgent(Player.BLACK),
                 opponent_agent=RandomAgent(Player.WHITE),
                 reward_decay=0.9):
        # We use RandomAgent to choose random moves when exploring
        self.qtable = qtable
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

    def train(self):
        epsilon = self.epsilon
        for epoch in range(1):
            self.epsilon = epsilon
            self.episode = 0
            self.total_rewards = []
            self.avg_q_values = []
            for game in range(self.total_games):
                print("Game {}/{}".format(game + 1, self.total_games))
                self.play_game()
                if (game + 1) % (self.total_games / 10) == 0:
                    self.epsilon = max(0, self.epsilon - 0.1)
        return self.qtable

    def play_game(self):
        # When playing a game we make first exploration/exploitation move
        # and then finish up episode playing according to the training player
        # and opponent strategies
        game_state = GameState()
        while not game_state.game_over:
            game_state_before, move_info = self.make_training_move(game_state)
            game_history = [] # List of (move_info, game_state) tuples
            if move_info is not None:
                # self.print_board(game_state_before)
                game_history.append((move_info.position, game_state_before, move_info.player))
                game_episode_state = game_state.clone()
                winner = self.play_episode(game_episode_state, game_history)
                self.update_training(winner, game_history)

    def play_episode(self, game_state, game_history):
        while not game_state.game_over:
            move = self.agents[game_state.current_player].get_best_move(game_state)
            if game_state.current_player == self.train_agent.player:
                # move = self.choose_action(game_state)
                player = game_state.current_player
                move_info = game_state.make_move(move)
                game_history.append((move_info.position, game_state.clone(), player))
            else:
                # move = self.agents[game_state.current_player].get_best_move(game_state)
                game_state.make_move(move)
        self.episode += 1
        # self.print_board(game_state)
        return game_state.winner

    def make_training_move(self, game_state):
        # Making two moves, training player move and opponent move
        move_info = None
        game_state_before = None
        for i in range(0, 2):
            if not game_state.game_over:
                if game_state.current_player == self.train_agent.player:
                    game_state_before = game_state.clone()
                    move = self.choose_action(game_state)
                    move_info = game_state.make_move(move)
                else:
                    move = self.agents[game_state.current_player].get_best_move(game_state)
                    game_state.make_move(move)
        return game_state_before, move_info

    def choose_action(self, game_state):
        legal_moves = list(game_state.legal_moves.keys())
        unexplored_moves = [move for move in legal_moves if
                            game_state_hash(game_state) not in self.qtable.all_states()
                             # or move not in self.qtable.qtable[game_state_hash(game_state)]
                            ]
        if random.random() < self.epsilon:
            # Exploration with preference to unexplored moves
            if unexplored_moves:
                return random.choice(unexplored_moves)
            else:
                return random.choice(legal_moves)
        else:
            # Exploitation
            return self.train_agent.get_best_move(game_state)

    def update_training(self, winner, game_history):
        updated_qvalues_count = 0
        updated_qvalue_updates_total = 0
        total_reward = 0
        reward = self.game_result_reward(winner)
        next_state = None
        # Moving backwards and updating Q-values
        for game_history_element in reversed(game_history):
            current_q_value = self.qtable.get_q_value(game_state_hash(game_history_element[1]),
                                        game_history_element[0])
            max_next_state_q_value = self.qtable.get_max_q_value(game_state_hash(game_history_element[1]))
            q_value = calculate_q_value(learning_rate=self.learning_rate,
                                        discount_factor=self.discount_factor,
                                        reward=reward,
                                        current_q_value=current_q_value,
                                        max_next_state_q_value=max_next_state_q_value)

            #
            # q_value = self.calculate_q_value(game_state_hash(game_history_element[1]),
            #                             game_history_element[0],
            #                             reward,
            #                             game_state_hash(next_state) if next_state is not None else None)
            self.qtable.update_q_value(game_state_hash(game_history_element[1]),
                                       game_history_element[0],
                                       q_value)
            updated_qvalues_count += 1
            updated_qvalue_updates_total += abs(q_value - current_q_value)
            # updated_qvalue_updates_total += q_value
            total_reward += reward
            next_state = game_history_element[1]

            # print("=======================================================")
            # print("State:")
            # print_board(game_history_element[1])
            # print("Next state:")
            # if next_state is not None:
            #     print("End game")
            # else :
            #     print_board(next_state)

            # Reward = max q-value of the next state
            # reward = self.qtable.get_q_value(game_state_hash(game_history_element[1]),
            #                             game_history_element[0])
            # reward = 0

            reward = reward * self.discount_factor

        self.total_rewards.append(total_reward)
        self.avg_q_values.append(updated_qvalue_updates_total / updated_qvalues_count)

    def game_result_reward(self, winner):
        if winner == Player.NONE:
            return DRAW_VALUE
        elif winner == self.train_agent.player:
            return WIN_VALUE
        else:
            return LOSS_VALUE

    def print_stats(self):
        # # Plot the total rewards per episode
        # plt.figure(figsize=(10, 5))
        # plt.plot(range(self.episode), self.total_rewards, label='Total Reward per Episode')
        # plt.xlabel('Episode')
        # plt.ylabel('Total Reward')
        # plt.title('Convergence Chart: Total Reward Over Time')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # Plot the average Q-values per episode
        plt.figure(figsize=(10, 5))
        plt.plot(range(self.episode), self.avg_q_values, label='Average Q-value per Episode', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Average Q-value')
        plt.title('Convergence Chart: Average Q-value Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def print_board(self, game_state):
        for row in game_state.board:
            print(' '.join('B' if cell == Player.BLACK else 'W' if cell == Player.WHITE else '.' for cell in row))
        print("\n")

