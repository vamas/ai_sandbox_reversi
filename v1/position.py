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

    @staticmethod
    def from_hash(hash_value):
        row = hash_value // 8
        col = hash_value % 8
        return Position(row, col)

    def equal(self, other):
        if isinstance(other, Position):
            return self.row == other.row and self.col == other.col
        return False

class SkipPosition(Position):
    def __init__(self):
        super().__init__(-1, -1)

    def __eq__(self, other):
        return isinstance(other, SkipPosition)

    def __hash__(self):
        return hash("SKIP")

    def equal(self, other):
        return isinstance(other, SkipPosition)