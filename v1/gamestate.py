from collections import defaultdict

import numpy as np

from v1.player import Player
from v1.player import opponent
from v1.position import Position, SkipPosition
from v1.moveinfo import MoveInfo

BOARD_SHAPE = 6

if BOARD_SHAPE == 8:
    Mid1 = (3, 4) # B
    Mid2 = (4, 3) # B
    Mid3 = (3, 3) # W
    Mid4 = (4, 4) # W
elif BOARD_SHAPE == 4:
    Mid1 = (2, 1) # B
    Mid2 = (1, 2) # B
    Mid3 = (2, 2) # W
    Mid4 = (1, 1) # W
elif BOARD_SHAPE == 6:
    Mid1 = (2, 3) # B
    Mid2 = (3, 2) # B
    Mid3 = (2, 2) # W
    Mid4 = (3, 3) # W

def print_board(game_state):
    for row in game_state.board:
        print(' '.join('B' if cell == Player.BLACK else 'W' if cell == Player.WHITE else '.' for cell in row))
    print("\n")

class GameState:
    Rows = BOARD_SHAPE
    Cols = BOARD_SHAPE

    @property
    def grid_shape(self):
        return self.Rows

    @property
    def free_positions_count(self):
        return sum(cell == Player.NONE for row in self.board for cell in row)

    def __init__(self, board=None, current_player=Player.BLACK):
        self.board = []
        self.current_player = None
        self.piece_count = {Player.BLACK: 0, Player.WHITE: 0}
        if board is None:
            self.init_normal_game()
        else:
            self.init_from_board(board, current_player)
        self.turn_count = 0
        self.double_skip_turns = 0
        self.all_positions = [Position(r, c) for r in range(self.Rows) for c in range(self.Cols)]
        self.all_positions.append(SkipPosition())
        self.illegal_move = False
        self.winner = Player.NONE
        self.game_over = False
        self.legal_moves = {}
        self.legal_moves_list = []
        self.update_legal_moves()

    def init_normal_game(self):
        self.board = [[Player.NONE for _ in range(self.Cols)] for _ in range(self.Rows)]
        self.board[Mid1[0]][Mid1[1]] = Player.BLACK
        self.board[Mid2[0]][Mid2[1]] = Player.BLACK
        self.board[Mid3[0]][Mid3[1]] = Player.WHITE
        self.board[Mid4[0]][Mid4[1]] = Player.WHITE
        self.piece_count = {Player.BLACK: 2, Player.WHITE: 2}
        self.current_player = Player.BLACK

    def init_from_board(self, board, current_player):
        self.board = board
        self.current_player = current_player
        self.piece_count = defaultdict(int)
        for row in board:
            for cell in row:
                if cell != Player.NONE:
                    self.piece_count[cell] += 1


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
        clone.legal_moves_list = self.legal_moves_list[:]
        clone.illegal_move = self.illegal_move
        clone.all_positions = self.all_positions[:]
        return clone

    def make_move(self, pos):
        moving_player = self.current_player

        # Check the legality of the move
        if not self.is_move_legal(pos):
            self.game_over = True
            self.winner = opponent(self.current_player)
            self.illegal_move = True
            return MoveInfo(moving_player, pos, [])

        # Check and handle skip move action
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

    # def occupied_positions(self):
    #     occupied = []
    #     for r in range(self.Rows):
    #         for c in range(self.Cols):
    #             if self.board[r][c] != Player.NONE:
    #                 occupied.append(Position(r, c))
    #     return occupied

    def occupied_positions(self):
        return [Position(r, c) for r in range(self.Rows) for c in range(self.Cols) if self.board[r][c] != Player.NONE]


    def is_inside_board(self, r, c):
        return 0 <= r < self.Rows and 0 <= c < self.Cols

    # def get_flips(self, pos, player):
    #     flips = []
    #     for direction in Position.Directions:
    #         flips_in_dir = []
    #         current = Position(pos.row + direction[0], pos.col + direction[1])
    #         while self.is_inside_board(current.row, current.col) and self.board[current.row][current.col] == opponent(player):
    #             flips_in_dir.append(current)
    #             current = Position(current.row + direction[0], current.col + direction[1])
    #         if self.is_inside_board(current.row, current.col) and self.board[current.row][current.col] == player:
    #             flips.extend(flips_in_dir)
    #     return flips
    def get_flips(self, pos, player):
        flips = []
        opponent_player = opponent(player)
        directions = Position.Directions
        for direction in directions:
            flips_in_dir = []
            current_row, current_col = pos.row + direction[0], pos.col + direction[1]
            while self.is_inside_board(current_row, current_col) and self.board[current_row][
                current_col] == opponent_player:
                flips_in_dir.append(Position(current_row, current_col))
                current_row += direction[0]
                current_col += direction[1]
            if self.is_inside_board(current_row, current_col) and self.board[current_row][current_col] == player:
                flips.extend(flips_in_dir)
        return flips

    # def update_legal_moves(self):
    #     self.legal_moves.clear()
    #     for r in range(self.Rows):
    #         for c in range(self.Cols):
    #             pos = Position(r, c)
    #             if self.board[r][c] == Player.NONE:
    #                 flips = self.get_flips(pos, self.current_player)
    #                 if flips:
    #                     self.legal_moves[pos] = flips
    #     if not self.legal_moves:
    #         self.legal_moves = {SkipPosition(): []}
    #     self.legal_moves_list = list(self.legal_moves.keys())
    #     return self.legal_moves

    def update_legal_moves(self):
        self.legal_moves.clear()
        current_player = self.current_player
        opponent_player = opponent(current_player)
        directions = Position.Directions

        for r in range(self.Rows):
            for c in range(self.Cols):
                if self.board[r][c] == Player.NONE:
                    pos = Position(r, c)
                    flips = []
                    for direction in directions:
                        flips_in_dir = []
                        current_row, current_col = r + direction[0], c + direction[1]
                        while self.is_inside_board(current_row, current_col) and self.board[current_row][
                            current_col] == opponent_player:
                            flips_in_dir.append(Position(current_row, current_col))
                            current_row += direction[0]
                            current_col += direction[1]
                        if self.is_inside_board(current_row, current_col) and self.board[current_row][
                            current_col] == current_player:
                            flips.extend(flips_in_dir)
                    if flips:
                        self.legal_moves[pos] = flips

        if not self.legal_moves:
            self.legal_moves = {SkipPosition(): []}
        self.legal_moves_list = list(self.legal_moves.keys())
        return self.legal_moves

    def is_move_legal(self, action):
        return action in self.legal_moves_list

    @property
    def board_flatten(self):
        return np.array([cell for row in self.board for cell in row])


