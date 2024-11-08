from v1.agent import AgentType
from v1.gamestate import GameState
from v1.player import Player
from v1.position import Position

class GameManager:

    def __init__(self, agent_black, agent_white):
        self.game_state = GameState()
        self.agents = {Player.BLACK: agent_black, Player.WHITE: agent_white}
        # board = [[Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK,
        #   Player.BLACK],
        #  [Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK,
        #   Player.BLACK],
        #  [Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK,
        #   Player.BLACK],
        #  [Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK,
        #   Player.BLACK],
        #  [Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.NONE, Player.NONE],
        #  [Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK,
        #   Player.NONE],
        #  [Player.BLACK, Player.BLACK, Player.BLACK, Player.BLACK, Player.NONE, Player.NONE, Player.NONE, Player.NONE],
        #  [Player.WHITE, Player.NONE, Player.BLACK, Player.BLACK, Player.NONE, Player.NONE, Player.NONE, Player.NONE]]
        # self.game_state = GameState(board=board, current_player=Player.BLACK)
        # self.agents = {Player.BLACK: agent_black, Player.WHITE: agent_white}


    def get_user_move(self):
        while True:
            try:
                skip_turn = input("Skip turn? (y/n): ")
                if skip_turn == "y":
                    return None
                row = int(input("Enter the row (0 - 5): "))
                col = int(input("Enter the column (0 - 6): "))
                if (row, col) not in self.game_state.legal_moves.keys():
                    print("Invalid move. Try again.")
                else:
                    return Position(row, col)
            except ValueError:
                print("Invalid input. Please enter numbers.")

    def run(self):
        move_info = None
        while not self.game_state.game_over:
            # print(f"Player: {self.game_state.current_player}")
            # self.print_board()
            # print(self.game_state.board)
            if self.agents[self.game_state.current_player].agent_type == AgentType.PLAYER:
                move = self.get_user_move()
                move_info = self.game_state.make_move(move)
            else:
                move = self.agents[self.game_state.current_player].get_best_move(self.game_state)
                move_info = self.game_state.make_move(move)

        # print("Winner is: ", self.game.get_winner())
        return self.game_state.winner

    def print_board(self):
        for row in self.game_state.board:
            print(' '.join('B' if cell == Player.BLACK else 'W' if cell == Player.WHITE else '.' for cell in row))
        print("\n")