from board import Board
from player import Player

class Game:
    def __init__(self, player1, player2, initial_board=None):
        self.board = Board(player1, player2)
        self.current_player = player1  # Black goes first
        self.opponent = player2  # White is the opponent

    def get_board(self):
        """Return the game board."""
        return self.board

    def switch_turn(self):
        """Switch the current player."""
        self.current_player, self.opponent = self.opponent, self.current_player

    def play_turn(self, row, col):
        """Handle the current player's move."""
        if self.board.make_move(row, col, self.current_player.get_color()):
            self.switch_turn()  # Switch turns if the move was successful
            return True
        return False

    def is_game_over(self):
        """Check if the game is over."""
        return not self.board.can_continue()

    def get_winner(self):
        """Return the winner if the game is over, or None otherwise."""
        return self.board.winner()

