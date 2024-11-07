from abc import ABC, abstractmethod

from player import Player

class AgentMinimax(Player):

    def __init__(self, name, code_value = 1):
        super().__init__(name, code_value, is_human=False)
        self.strategy = "minimax"

    @property
    @abstractmethod
    def max_depth(self):
        pass

    @abstractmethod
    def heuristic_fn(self, board, is_maximizing, player_code_value, opponent_code_value):
        pass

    @abstractmethod
    def order_moves(self, moves):
        pass

    def find_move(self, board):
        best_score = -float('inf')
        best_move = None, None
        available_moves = self.order_moves(board.available_moves(self.code_value))
        for r,c in available_moves:
            board_clone = board.clone()
            board_clone.make_move(r, c, self.code_value)
            score = self.minimax(board_clone, is_maximizing = False)
            if score > best_score:
                best_score = score
                best_move = r,c
        if best_move == (None, None):
            print("Can't find suitable move")
        return best_move

    def utility(self, board):
        w = board.winner()
        if w is None:
            return 0
        if w == self:
            return 1
        else:
            return -1

    def terminal_test(self, board):
        a = board.can_continue()
        return not a

    def minimax(self, board, is_maximizing = True, a = -float('inf'), b = float('inf'), level = 0):
        # R player is MAX player and B player is MIN player
        if self.terminal_test(board):
            return self.utility(board)

        player = self.code_value
        opponent = board.get_opponent(self.code_value)
        if level > self.max_depth:
            return self.heuristic_fn(board, is_maximizing, player, opponent)

        board_clone = board.clone()

        if is_maximizing:
            # Maximizing player
            v = -float('inf')
            for move in board_clone.available_moves(player):
                board_clone.make_move(move[0], move[1], player)
                v = max(v, self.minimax(board_clone, not is_maximizing, a, b, level + 1))
                a = max(a, v)
                if b <= a:
                    break
            return v
        else:
            # Minimizing player
            v = float('inf')
            for move in board_clone.available_moves(opponent):
                board_clone.make_move(move[0], move[1], opponent)
                v = min(v, self.minimax(board_clone, not is_maximizing, a, b, level + 1))
                b = min(b, v)
                if b <= a:
                    break
            return v


