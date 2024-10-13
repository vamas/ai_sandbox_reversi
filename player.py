class Player:
    def __init__(self, color, name = "Player", is_human = True):
        self.color = color  # 1 for Black, -1 for White
        self.name = name
        self.color_name = "Black" if self.color == 1 else "White"
        self.is_human = is_human

    def get_color(self):
        """Return the player's color (1 for Black, -1 for White)."""
        return self.color
