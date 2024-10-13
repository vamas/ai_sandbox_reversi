class Board:
    def __init__(self, player1, player2, size=8, initial_board=None):
        self.size = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.player1 = player1
        self.player2 = player2
        # Initial configuration
        if initial_board is None:
            self.grid[3][3] = self.player2  # White
            self.grid[3][4] = self.player1   # Black
            self.grid[4][3] = self.player1   # Black
            self.grid[4][4] = self.player2  # White
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


    def grid_shape(self):
        return self.size

    def get_flipping_pieces(self, row, col, player):
        pieces_to_flip = []
        if player == self.player1:
            opponent = self.player2
        else:
            opponent = self.player1
        for dr, dc in self.directions:
            r, c = row + dr, col + dc
            if self._is_on_board(r, c) and self.grid[r][c] == opponent:
                # Move in given direction until we find a piece of the same color
                open_pieces = []
                while self._is_on_board(r, c) and self.grid[r][c] == opponent:
                    open_pieces.append((r, c))
                    r += dr
                    c += dc
                if self._is_on_board(r, c) and self.grid[r][c] == player:
                    pieces_to_flip.extend(open_pieces)
        return list(set(pieces_to_flip))

    def is_valid_move(self, row, col, player):
        """Check if placing a piece at (row, col) is a valid move for the player."""
        if not row in range(self.size) and not col in range(self.size):
            return False
        if self.grid[row][col] != 0:
            return False

        if player == self.player1:
            opponent = self.player2
        else:
            opponent = self.player1

        for dr, dc in self.directions:
            r, c = row + dr, col + dc
            if self._is_on_board(r, c) and self.grid[r][c] == opponent:
                # Move in given direction until we find a piece of the same color
                while self._is_on_board(r, c) and self.grid[r][c] == opponent:
                    r += dr
                    c += dc
                if self._is_on_board(r, c) and self.grid[r][c] == player:
                    return True
        return False

    def _is_on_board(self, row, col):
        """Check if the position (row, col) is on the board."""
        return 0 <= row < self.size and 0 <= col < self.size

    def undo_move(self, row, col):
        self.grid[row][col] = 0

    def make_move(self, row, col, player):
        """Place a piece on the board if the move is valid, and flip opponent's pieces."""
        if not self.is_valid_move(row, col, player):
            return False
        # Place the piece and flip opponent's pieces
        self.grid[row][col] = player
        # Implement flipping logic here
        for r, c in self.get_flipping_pieces(row, col, player):
            self.grid[r][c] = player
        return True

    def make_move_player1(self, row, col):
        player = self.player1
        self.make_move(row,col,player)

    def make_move_player2(self, row, col):
        player = self.player2
        self.make_move(row, col, player)

    def available_moves_player1(self):
        player = self.player1
        return self.available_moves(player)

    def available_moves_player2(self):
        player = self.player2
        return self.available_moves(player)

    def available_moves(self, player):
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.is_valid_move(row, col, player):
                    moves.append((row, col))
        return moves

    def check_available_moves(self, player):
        return self.available_moves(player) != []

    def get_pieces_count(self):
        """Return the count of pieces for each player."""
        player1_count = sum(cell == self.player1 for row in self.grid for cell in row)
        player2_count = sum(cell == self.player2 for row in self.grid for cell in row)
        return { self.player1: player1_count, self.player2: player2_count }

    def winner(self):
        """Check the board to see if there is a winner or if the game is a draw."""
        # Implement winner checking logic
        # Check if there are no moves for player1
        # Check if there are no moves for player2
        # Check if the board is full
        if self.can_continue():
            return None
        player1_moves = self.check_available_moves(self.player1)
        player2_moves = self.check_available_moves(self.player2)
        if not player1_moves and not player2_moves:
            pieces_count = self.get_pieces_count()
            if pieces_count[self.player1] > pieces_count[self.player2]:
                return self.player1
            elif pieces_count[self.player2] > pieces_count[self.player1]:
                return self.player2
        else:
            return None

    def can_continue(self):
        return (not self.is_full()) or self.available_moves_player1() != [] or self.available_moves_player2() != []

    def is_full(self):
        """Check if the board is full."""
        return all(cell != 0 for row in self.grid for cell in row)

    def display(self):
        """Print the board state for debugging."""
        for row in self.grid:
            print(' '.join(['B' if cell == 1 else 'W' if cell == -1 else '.' for cell in row]))
