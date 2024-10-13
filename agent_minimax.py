import random
import copy

from player import Player

class AgentMinimax(Player):

    def __init__(self, color, name):
        super().__init__(color, name, is_human=False)
        self.strategy = "minimax"

    def get_available_moves(self, board):
        if self == board.player1:
            return board.available_moves_player1()
        else:
            return board.available_moves_player2()

    def find_move(self, board):
        best_score = -float('inf')
        best_move = None, None
        for r,c in self.get_available_moves(board):
            board_copy = copy.deepcopy(board)
            board_copy.make_move(r, c, self)
            score = self.minimax(board_copy, max_level = 3)
            if score > best_score:
                best_score = score
                best_move = r,c
        if best_move == (None, None):
            print("Can't find suitable move")
        return best_move

    def make_move(self, board):
        row, col = self.find_move(board)
        if row is None or col is None:
            return False
        return board.make_move(row, col, self)

    def utility(self, board):
        w = board.winner()
        if w is None:
            return 0
        if w == self:
            return 1
        else:
            return -1

    def terminal_test(self, board):
        board.can_continue()

    def minimax(self, board, is_maximizing = True, max_level = 5, a = -float('inf'), b = float('inf'), level = 0):
        # R player is MAX player and B player is MIN player
        if self.terminal_test(board):
            return self.utility(board)
        if level > max_level:
            # return random.choice([-1,0,1])
            return 0

        if is_maximizing:
            player = self
        else:
            if self == board.player1:
                player = board.player2
            else:
                player = board.player1

        board_copy = copy.deepcopy(board)

        if is_maximizing:
            # Maximazing player
            v = -float('inf')
            for move in board_copy.available_moves(player):
                # board_copy = copy.deepcopy(board)
                board_copy.make_move_player1(move[0], move[1])
                v = max(v, self.minimax(board_copy, not is_maximizing, max_level, a, b, level + 1))
                a = max(a, v)
                if b <= a:
                    break
            return v
        else:
            # Minimizing player
            v = float('inf')
            for move in board_copy.available_moves(player):
                # board_copy = copy.deepcopy(board)
                board_copy.make_move_player2(move[0], move[1])
                v = min(v, self.minimax(board_copy, not is_maximizing, max_level, a, b, level + 1))
                b = min(b, v)
                if b <= a:
                    break
            return v


