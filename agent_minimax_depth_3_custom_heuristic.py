from agent_minimax import AgentMinimax

class AgentMinimaxDepth3CustomHeuristic(AgentMinimax):

    def __init__(self, name, code_value = 1):
        super().__init__(name, code_value)
        self.strategy = "minimax-random"

    @property
    def max_depth(self):
        return 3

    def heuristic_fn(self, board, is_maximizing, player_code_value, opponent_code_value):
        pieces_count = board.get_pieces_count()
        player_count = pieces_count[player_code_value]
        opponent_count = pieces_count[opponent_code_value]

        # Difference in piece count
        piece_score = player_count - opponent_count

        # Additional factors like corner control can be added here
        # Example: if a corner is occupied by the current player, give a bonus score
        corner_score = 0
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for row, col in corners:
            if board.grid[row][col] == player_code_value:
                corner_score += 25  # A large bonus for controlling a corner

        # Outer edges occupied by player give bonus
        edges_score = 0
        edges = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
                 (7,0), (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7),
                 (0,0), (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0),
                 (0,7), (1,7), (2,7), (3,7), (4,7), (5,7), (6,7), (7,7)]
        for row, col in edges:
            if board.grid[row][col] == player_code_value:
                edges_score += 5  # A large bonus for controlling a corner

        return piece_score + corner_score

    def order_moves(self, moves):
        return moves