from board import Board
from player import Player

class Game:
    def __init__(self, player1, player2, initial_board=None):
        self.board = Board(player1.code_value, player2.code_value, initial_board=initial_board)
        self.player1 = player1
        self.player2 = player2
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
        if self.board.make_move(row, col, self.current_player.code_value):
            self.switch_turn()  # Switch turns if the move was successful
            return True
        return False

    def skip_turn(self):
        self.switch_turn()

    def play_agent_turn(self):
        """Handle the current player's move."""
        row, col = self.current_player.find_move(self.board)
        if row is None or col is None:
            # return False
            self.skip_turn()
        if self.board.make_move(row, col, self.current_player.code_value):
            self.switch_turn()

    def is_game_over(self):
        """Check if the game is over."""
        return not self.board.can_continue()

    def get_winner_code_value(self):
        """Return the winner if the game is over, or None otherwise."""
        return self.board.winner()

    def get_winner_name(self):
        """Return the winner if the game is over, or None otherwise."""
        winner_code = self.get_winner_code_value()
        if winner_code is None:
            return "Draw"
        if winner_code == self.player1.code_value:
            return self.player1.name
        else:
            return self.player2.name