from enum import Enum
from collections import defaultdict

import numpy as np

from v1.player import Player
from v1.player import opponent
from v1.position import Position, SkipPosition
from v1.moveinfo import MoveInfo

BOARD_SHAPE = 4

# Mid1 = (3, 3)
# Mid2 = (4, 4)
# Mid3 = (3, 4)
# Mid4 = (4, 3)

Mid1 = (2, 1)
Mid2 = (1, 2)
Mid3 = (2, 2)
Mid4 = (1, 1)


def print_board(game_state):
    for row in game_state.board:
        print(' '.join('B' if cell == Player.BLACK else 'W' if cell == Player.WHITE else '.' for cell in row))
    print("\n")

class GameState:
    Rows = BOARD_SHAPE
    Cols = BOARD_SHAPE

    def __init__(self, board=None, current_player=Player.BLACK):
        if board is None:
            self.init_normal_game()
        else:
            self.board = board
            self.current_player = current_player
            self.piece_count = defaultdict(int)
            for row in board:
                for cell in row:
                    if cell != Player.NONE:
                        self.piece_count[cell] += 1
            self.game_over = False
            self.winner = Player.NONE
            self.legal_moves = {}
            self.update_legal_moves()
        self.turn_count = 0
        self.double_skip_turns = 0

    def init_normal_game(self):
        self.board = [[Player.NONE for _ in range(self.Cols)] for _ in range(self.Rows)]
        self.board[Mid1[0]][Mid1[1]] = Player.BLACK
        self.board[Mid2[0]][Mid2[1]] = Player.BLACK
        self.board[Mid3[0]][Mid3[1]] = Player.WHITE
        self.board[Mid4[0]][Mid4[1]] = Player.WHITE
        self.piece_count = {Player.BLACK: 2, Player.WHITE: 2}
        self.current_player = Player.BLACK
        self.game_over = False
        self.winner = Player.NONE
        self.legal_moves = {}
        self.update_legal_moves()

    def hash_float32(self):
        hash_value = 0
        for row in range(self.Rows):
            for col in range(self.Cols):
                if self.board[row][col] != Player.NONE:
                    hash_value = hash_value * 3 + (1 if self.board[row][col] == Player.BLACK else 2)
        return np.float32(hash_value)

    def reverse_hash_float32(self, hash_value):
        hash_value = int(hash_value)
        board = [[Player.NONE for _ in range(self.Cols)] for _ in range(self.Rows)]
        for row in range(self.Rows - 1, -1, -1):
            for col in range(self.Cols - 1, -1, -1):
                if hash_value == 0:
                    return board
                value = hash_value % 3
                hash_value //= 3
                if value == 1:
                    board[row][col] = Player.BLACK
                elif value == 2:
                    board[row][col] = Player.WHITE
        return board


    def hash(self):
        hash_value = 17
        for row in range(self.Rows):
            for col in range(self.Cols):
                if self.board[row][col] != Player.NONE:
                    hash_value = hash_value * 31 + hash((row, col, self.board[row][col]))
        return str(hash_value)

        # hash_value = 17
        # hash_value = hash_value * 31 + hash(self.current_player)
        # hash_value = hash_value * 31 + hash(self.game_over)
        # hash_value = hash_value * 31 + hash(self.winner)
        # hash_value = hash_value * 31 + hash(self.turn_count)
        # hash_value = hash_value * 31 + hash(self.double_skip_turns)
        #
        # for player, count in self.piece_count.items():
        #     hash_value = hash_value * 31 + hash(player)
        #     hash_value = hash_value * 31 + hash(count)
        #
        # for pos, flips in self.legal_moves.items():
        #     hash_value = hash_value * 31 + hash(pos)
        #     for flip in flips:
        #         hash_value = hash_value * 31 + hash(flip)
        #
        # for row in self.board:
        #     for cell in row:
        #         hash_value = hash_value * 31 + hash(cell)
        #
        # return str(hash_value)


    def clone(self):
        clone = GameState()
        clone.current_player = self.current_player
        clone.game_over = self.game_over
        clone.winner = self.winner
        clone.double_skip_turns = self.double_skip_turns
        clone.turn_count = self.turn_count
        clone.board = [row[:] for row in self.board]
        clone.piece_count = self.piece_count.copy()
        clone.legal_moves = {pos: flips[:] for pos, flips in self.legal_moves.items()}
        return clone

    def make_move(self, pos):
        moving_player = self.current_player

        if isinstance(pos, SkipPosition) or pos is None:
            self.double_skip_turns += 1
            self.check_winner()
            self.change_player()
            return MoveInfo(moving_player, pos, [])

        self.double_skip_turns = 0
        flips = self.legal_moves[pos]
        self.board[pos.row][pos.col] = moving_player

        self.flip_disks(flips)
        self.update_piece_count(moving_player, len(flips))
        self.check_winner()
        self.change_player()

        return MoveInfo(moving_player, pos, flips)

    def check_winner(self):
        if self.double_skip_turns >= 2:
            if self.piece_count[Player.BLACK] > self.piece_count[Player.WHITE]:
                self.winner = Player.BLACK
            elif self.piece_count[Player.BLACK] < self.piece_count[Player.WHITE]:
                self.winner = Player.WHITE
            else:
                self.winner = Player.NONE
            self.game_over = True
        elif (self.piece_count[Player.BLACK] + self.piece_count[Player.WHITE]) == 64:
            if self.piece_count[Player.BLACK] > self.piece_count[Player.WHITE]:
                self.winner = Player.BLACK
            elif self.piece_count[Player.BLACK] < self.piece_count[Player.WHITE]:
                self.winner = Player.WHITE
            else:
                self.winner = Player.NONE
            self.game_over = True

    def flip_disks(self, flipped):
        for pos in flipped:
            self.board[pos.row][pos.col] = opponent(self.board[pos.row][pos.col])

    def update_piece_count(self, player, flips):
        self.piece_count[player] += flips + 1
        self.piece_count[opponent(player)] -= flips

    def change_player(self):
        if self.current_player == Player.BLACK:
            self.turn_count += 1
        self.current_player = opponent(self.current_player)
        self.update_legal_moves()

    def occupied_positions(self):
        occupied = []
        for r in range(self.Rows):
            for c in range(self.Cols):
                if self.board[r][c] != Player.NONE:
                    occupied.append(Position(r, c))
        return occupied

    def is_inside_board(self, r, c):
        return 0 <= r < self.Rows and 0 <= c < self.Cols

    def get_flips(self, pos, player):
        flips = []
        for dir in Position.Directions:
            flips_in_dir = []
            current = Position(pos.row + dir[0], pos.col + dir[1])
            while self.is_inside_board(current.row, current.col) and self.board[current.row][current.col] == opponent(player):
                flips_in_dir.append(current)
                current = Position(current.row + dir[0], current.col + dir[1])
            if self.is_inside_board(current.row, current.col) and self.board[current.row][current.col] == player:
                flips.extend(flips_in_dir)
        return flips

    def update_legal_moves(self):
        self.legal_moves.clear()
        for r in range(self.Rows):
            for c in range(self.Cols):
                pos = Position(r, c)
                if self.board[r][c] == Player.NONE:
                    flips = self.get_flips(pos, self.current_player)
                    if flips:
                        self.legal_moves[pos] = flips
        if not self.legal_moves:
            self.legal_moves = {SkipPosition(): []}
        return self.legal_moves

    def grid_shape(self):
        return self.Rows

    def lookup_legal_action(self, action):
        for key in self.legal_moves.keys():
            if key == action:
                return key
        return None