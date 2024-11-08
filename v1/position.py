class Position:
    Directions = [
        (-1, -1),  # Top-left
        (-1, 0),   # Top
        (-1, 1),   # Top-right
        (0, -1),   # Left
        (0, 1),    # Right
        (1, -1),   # Bottom-left
        (1, 0),    # Bottom
        (1, 1)     # Bottom-right
    ]

    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.row == other.row and self.col == other.col
        return False

    def __hash__(self):
        return 8 * self.row + self.col