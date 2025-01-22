import math

import numpy as np

from v1.gamestate import GameState
from v1.player import Player
from v1.position import SkipPosition, Position

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