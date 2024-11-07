import copy

class Board:
    def __init__(self, player1_code_value = 1, player2_code_value = -1, size=8, initial_board=None):
        self.size = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.player1_code_value = player1_code_value
        self.player2_code_value = player2_code_value
        # Initial configuration
        if initial_board is None:
            self.grid[3][3] = player2_code_value  # White
            self.grid[3][4] = player1_code_value   # Black
            self.grid[4][3] = player1_code_value   # Black
            self.grid[4][4] = player2_code_value  # White
        else:
            self.grid = initial_board
        self.directions = [(-1, 0),  # up
                      (1, 0),  # down
                      (0, -1),  # left
                      (0, 1),  # right
                      (-1, -1),  # up-left
                      (-1, 1),  # up-right
                      (1, -1),  # down-left
                      (1, 1)] # down-right

    def clone(self):
        return copy.deepcopy(self)

    def get_opponent(self, player_code_value):
        if player_code_value == self.player1_code_value:
            return self.player2_code_value
        else:
            return self.player1_code_value

    def grid_shape(self):
        return self.size

    def get_flipping_pieces(self, row, col, player_code_value):
        pieces_to_flip = []
        opponent_code_value = player_code_value * -1
        for dr, dc in self.directions:
            r, c = row + dr, col + dc
            if self._is_on_board(r, c) and self.grid[r][c] == opponent_code_value:
                # Move in given direction until we find a piece of the same color
                open_pieces = []
                while self._is_on_board(r, c) and self.grid[r][c] == opponent_code_value:
                    open_pieces.append((r, c))
                    r += dr
                    c += dc
                if self._is_on_board(r, c) and self.grid[r][c] == player_code_value:
                    pieces_to_flip.extend(open_pieces)
        return list(set(pieces_to_flip))

    def is_valid_move(self, row, col, player_code_value):
        """Check if placing a piece at (row, col) is a valid move for the player."""
        if not row in range(self.size) and not col in range(self.size):
            return False
        if self.grid[row][col] != 0:
            return False

        opponent_code_value = player_code_value * -1

        for dr, dc in self.directions:
            r, c = row + dr, col + dc
            if self._is_on_board(r, c) and self.grid[r][c] == opponent_code_value:
                # Move in given direction until we find a piece of the same color
                while self._is_on_board(r, c) and self.grid[r][c] == opponent_code_value:
                    r += dr
                    c += dc
                if self._is_on_board(r, c) and self.grid[r][c] == player_code_value:
                    return True
        return False

    def _is_on_board(self, row, col):
        """Check if the position (row, col) is on the board."""
        return 0 <= row < self.size and 0 <= col < self.size

    def make_move(self, row, col, player_code_value):
        """Place a piece on the board if the move is valid, and flip opponent's pieces."""
        if not self.is_valid_move(row, col, player_code_value):
            return False
        # Place the piece and flip opponent's pieces
        self.grid[row][col] = player_code_value
        # Implement flipping logic here
        for r, c in self.get_flipping_pieces(row, col, player_code_value):
            self.grid[r][c] = player_code_value
        return True

    def available_moves(self, player_code_value):
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.is_valid_move(row, col, player_code_value):
                    moves.append((row, col))
        return moves

    def check_available_moves(self, player_code_value):
        return self.available_moves(player_code_value) != []

    def get_pieces_count(self):
        """Return the count of pieces for each player."""
        return { self.player1_code_value:  sum(cell == 1 for row in self.grid for cell in row),
                 self.player2_code_value: sum(cell == -1 for row in self.grid for cell in row) }

    def winner(self):
        """Check the board to see if there is a winner or if the game is a draw."""
        # Implement winner checking logic
        # Check if there are no moves for player1
        # Check if there are no moves for player2
        # Check if the board is full
        if self.can_continue():
            return None
        if (not self.check_available_moves(self.player1_code_value)
                and not self.check_available_moves(self.player2_code_value)):
            pieces_count = self.get_pieces_count()
            if pieces_count[self.player1_code_value] > pieces_count[self.player2_code_value]:
                return self.player1_code_value
            elif pieces_count[self.player2_code_value] > pieces_count[self.player1_code_value]:
                return self.player2_code_value
        else:
            return None

    def can_continue(self):
        return  (self.available_moves(self.player1_code_value) != []
                or self.available_moves(self.player2_code_value) != [])

    def is_full(self):
        """Check if the board is full."""
        return all(cell != 0 for row in self.grid for cell in row)

    def display(self):
        """Print the board state for debugging."""
        for row in self.grid:
            print(' '.join(['B' if cell == self.player1_code_value else 'W' if cell == self.player2_code_value else '.' for cell in row]))
