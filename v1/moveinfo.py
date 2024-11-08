class MoveInfo:
    def __init__(self, player, position, flips=None):
        self.player = player
        self.position = position
        self.flips = flips if flips is not None else []
        # self.is_skip = is_skip