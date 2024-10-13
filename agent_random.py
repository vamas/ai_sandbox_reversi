import random

from player import Player

class AgentRandom(Player):

    def __init__(self, color, name):
        super().__init__(color, name, is_human=False)
        self.strategy = "random"

    def find_move(self, board):
        # Choose a random valid move
        moves = board.available_moves(self)
        if moves:
            return random.choice(board.available_moves(self))
        else:
            return None, None

    def make_move(self, board):
        row, col = self.find_move(board)
        if row is None or col is None:
            return False
        return board.make_move(row, col, self)

