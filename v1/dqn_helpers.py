import math

import numpy as np
import matplotlib.pyplot as plt

from v1.gamestate import GameState
from v1.player import Player, opponent
from v1.position import SkipPosition, Position


BOARD_SHAPE = 4

def game_state_one_hot_encode(game_state, training_player, shape=BOARD_SHAPE):
    """
        Encode a game state to a one-hot encoded vector.
    :param game_state: input game state
    :param shape: board size
    :return:
        [My pieces BOARD_SHAPExBOARD_SHAPE] +
        [Opponent pieces BOARD_SHAPExBOARD_SHAPE] +
        [Legal moves BOARD_SHAPExBOARD_SHAPE]
    """
    if game_state is None:
        return np.zeros(shape * shape + shape * shape + shape * shape, dtype=float)
    state = np.zeros(game_state.Rows * game_state.Cols + game_state.Rows * game_state.Cols, dtype=float)
    for row in range(game_state.Rows):
        for col in range(game_state.Cols):
            if game_state.board[row][col] != Player.NONE:
                idx = row * game_state.Cols + col
                if game_state.board[row][col] == training_player:
                    state[idx] = 1
                elif game_state.board[row][col] == opponent(training_player):
                    state[idx + shape * shape] = 1
    final_state = np.append(state, legal_moves_one_hot_encode(game_state.legal_moves.keys(), shape))
    # final_state = np.append(final_state, [game_state.current_player == training_player])
    # final_state = state
    return final_state.flatten()

def legal_moves_one_hot_encode(legal_moves, shape=BOARD_SHAPE):
    state = np.zeros(shape * shape, dtype=float)
    for move in legal_moves:
        if not isinstance(move, SkipPosition):
            idx = move.row * shape + move.col
            state[idx] = 1
    return state

def legal_moves_mask(legal_moves, shape=BOARD_SHAPE):
    state = np.zeros(shape * shape + 1, dtype=float)
    for move in legal_moves:
        if isinstance(move, SkipPosition):
            state[shape * shape - 1] = 1
        else:
            idx = move.row * shape + move.col
            state[idx] = 1
    return state

#
# def game_state_one_hot_decode(encoded_state, shape=BOARD_SHAPE):
#     """
#     Decode a one-hot encoded game state back to a GameState object.
#
#     Args:
#         encoded_state (np.array): One-hot encoded game state.
#         shape (int): The shape of the board (default is BOARD_SHAPE).
#
#     Returns:
#         GameState: The decoded GameState object.
#     """
#     board = np.zeros((shape, shape), dtype=int)
#     half = shape * shape
#     for idx in range(half):
#         row = idx // shape
#         col = idx % shape
#         if encoded_state[idx] == 1:
#             board[row][col] = Player.BLACK
#         elif encoded_state[idx + half] == 1:
#             board[row][col] = Player.WHITE
#     return GameState(board=board)

def legal_actions_to_binary_mask(legal_actions, shape=BOARD_SHAPE):
    """
    Convert legal actions to a binary mask.

    Args:
        legal_actions (list): List of legal actions (positions).
        shape (int): The shape of the board (default is BOARD_SHAPE).

    Returns:
        np.array: Binary mask of legal actions.
    """
    mask = np.zeros(shape * shape + 1, dtype=int)
    for action in legal_actions:
        if isinstance(action, SkipPosition):
            mask[-1] = 1
        else:
            idx = action.row * shape + action.col
            mask[idx] = 1
    return mask

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


def print_stats(epoch_q_value_changes,
                illegal_moves,
                epsilon_history,
                learning_rate_history,
                training_scores,
                legal_moves):

    window = 10

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    average_illegal_moves = []
    for ind in range(len(illegal_moves) - window + 1):
        average_illegal_moves.append(np.mean(illegal_moves[ind:ind + window]))

    axs[0, 0].plot(epoch_q_value_changes)
    axs[0, 0].set_title("Average Q-value Change Per Epoch")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Average Q-value Change")

    # axs[0, 1].plot(range(len(average_illegal_moves)), average_illegal_moves, label="Illegal moves")
    # axs[0, 1].set_title("Illegal vs Legal moves")
    # axs[0, 1].set_xlabel("Epoch")
    # axs[0, 1].set_ylabel("Illegal vs Legal moves")
    # axs[0, 1].legend()

    axs[0, 1].plot(epsilon_history)
    axs[0, 1].set_title("Epsilon History")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Epsilon")

    axs[1, 0].plot(learning_rate_history)
    axs[1, 0].set_title("Learning Rates History")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Learning Rate")

    # plot lines
    for agent in training_scores.keys():
        training_score = []
        for ind in range(len(training_scores[agent]) - window + 1):
            training_score.append(np.mean(training_scores[agent][ind:ind + window]))
        axs[1, 1].plot(range(len(training_score)), training_score, label=agent)
    axs[1, 1].set_title("Training Scores")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
