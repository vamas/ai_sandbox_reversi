class Player:
    def __init__(self, name = "Player", code_value = 1, is_human = True):
        self.name = name
        self.is_human = is_human
        self.code_value = code_value

    #
    # def get_color(self):
    #     """Return the player's color (1 for Black, -1 for White)."""
    #     return self.color
