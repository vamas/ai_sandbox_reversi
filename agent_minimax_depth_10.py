from agent_minimax import AgentMinimax

class AgentMinimaxDepth10(AgentMinimax):

    def __init__(self, name, code_value = 1):
        super().__init__(name, code_value)
        self.strategy = "minimax-random"

    @property
    def max_depth(self):
        return 10

    def heuristic_fn(self, board, is_maximizing, player_code_value, opponent_code_value):
        return 0

    def order_moves(self, moves):
        return moves