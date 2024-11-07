import time

class Console:
    def __init__(self, game):
        self.game = game
        # Initialize Pygame and set up the window, colors, and fonts

    def get_user_move(self, player):
        while True:
            try:
                skip_turn = input("Skip turn? (y/n): ")
                if skip_turn == "y":
                    return None
                row = int(input("Enter the row (0 - 5): "))
                col = int(input("Enter the column (0 - 6): "))
                if (row, col) not in self.game.board.available_moves(player):
                    print("Invalid move. Try again.")
                else:
                    return row, col
            except ValueError:
                print("Invalid input. Please enter numbers.")

    def run(self):
        time_taken = {self.game.player1.name: 0, self.game.player2.name: 0}
        while not self.game.is_game_over():
            if self.game.current_player.is_human:
                move = self.get_user_move(self.game.current_player)
                if not move is None:
                    self.game.play_turn(move[0], move[1])
                self.game.switch_turn()
            else:
                start_time = time.time()
                current_player = self.game.current_player.name
                self.game.play_agent_turn()
                end_time = time.time()
                execution_time = end_time - start_time
                time_taken[current_player] += execution_time

        # print("Winner is: ", self.game.get_winner())
        return self.game.get_winner_name(), time_taken