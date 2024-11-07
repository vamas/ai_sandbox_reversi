import random

from player import Player

class AgentRandom(Player):

    def __init__(self, name, code_value = 1):
        super().__init__(name, code_value, is_human=False)
        self.strategy = "random"

    def find_move(self, board):
        # Choose a random valid move
        moves = board.available_moves(self.code_value)
        if moves:
            return random.choice(board.available_moves(self.code_value))
        else:
            return None, None

    # def make_move(self, board):
    #     row, col = self.find_move(board)
    #     if row is None or col is None:
    #         return False
    #     return board.make_move(row, col, self)

