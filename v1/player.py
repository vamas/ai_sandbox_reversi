from enum import Enum

class Player(Enum):
    NONE = 0
    BLACK = 1
    WHITE = 2

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


def opponent(player):
    if player == Player.BLACK:
        return Player.WHITE
    if player == Player.WHITE:
        return Player.BLACK
    return Player.NONE