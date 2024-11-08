import random
from v1.agent import Agent, AgentType
from v1.player import opponent


class MinimaxAgent(Agent):
    def __init__(self, player, max_depth):
        self.max_depth = max_depth
        super().__init__(player, AgentType.COMPUTER)

    def get_best_move(self, game_state):
        best_score = -float('inf')
        best_move = None
        available_moves = game_state.legal_moves.keys()
        if len(available_moves) == 1:
            return list(available_moves)[0]
        for pos in available_moves:
            game_state_clone = game_state.clone()
            game_state_clone.make_move(pos)
            score = self.minimax(game_state_clone, is_maximizing=False, level=self.max_depth)
            if score > best_score:
                best_score = score
                best_move = pos
        return best_move

    def minimax(self, game_state, is_maximizing = True, a = -float('inf'), b = float('inf'), level = 0):
        # Minimax with alpha-beta pruning
        if self.terminal_test(game_state):
            return self.utility(game_state)

        if level == 0:
            return self.heuristic_fn(game_state)

        available_moves = game_state.legal_moves.keys()

        if is_maximizing:
            # Maximizing player
            v = -float('inf')
            for move in available_moves:
                game_state_clone = game_state.clone()
                game_state_clone.make_move(move)
                v = max(v, self.minimax(game_state_clone, not is_maximizing, a, b, level - 1))
                a = max(a, v)
                if b <= a:
                    break
            return v
        else:
            # Minimizing player
            v = float('inf')
            for move in available_moves:
                game_state_clone = game_state.clone()
                game_state_clone.make_move(move)
                v = min(v, self.minimax(game_state_clone, not is_maximizing, a, b, level - 1))
                b = min(b, v)
                if b <= a:
                    break
            return v

    def terminal_test(self, game_state):
        return game_state.game_over

    def utility(self, game_state):
        w = game_state.winner
        if w is None:
            return 0
        if w == game_state.current_player:
            return 1
        else:
            return -1

    def heuristic_fn(self, game_state):
        pieces_count = game_state.piece_count
        player_count = pieces_count[game_state.current_player]
        opponent_count = pieces_count[opponent(game_state.current_player)]

        # Difference in piece count
        piece_score = player_count - opponent_count

        # Additional factors like corner control can be added here
        # Example: if a corner is occupied by the current player, give a bonus score
        corner_score = 0
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for row, col in corners:
            if game_state.board[row][col] == game_state.current_player:
                corner_score += 25  # A large bonus for controlling a corner

        # Outer edges occupied by player give bonus
        edges_score = 0
        edges = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
                 (7,0), (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7),
                 (0,0), (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0),
                 (0,7), (1,7), (2,7), (3,7), (4,7), (5,7), (6,7), (7,7)]
        for row, col in edges:
            if game_state.board[row][col] == game_state.current_player:
                edges_score += 5  # A large bonus for controlling a corner

        return piece_score + corner_score + edges_score