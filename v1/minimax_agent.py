import random
from v1.agent import Agent, AgentType
from v1.player import opponent
from v1.position import SkipPosition

heuristic_weights_8 = [
    [100,-25,10,5,5,10,-25,100],
    [-25,-25,2,2,2,2,-25,-25],
    [10,2,5,1,1,5,2,10],
    [5,2,1,2,2,1,2,5],
    [5,2,1,2,2,1,2,5],
    [10,2,5,1,1,5,2,10],
    [-25,-25,2,2,2,2,-25,-25],
    [100,-25,10,5,5,10,-25,100]
]

heuristic_weights_4 = [
    [100,-25,-25,100],
    [-25,2,2,-25],
    [-25,2,2,-25],
    [100,-25,-25,100]
]

def heuristic_fn(game_state):
    board_shape = game_state.grid_shape
    heuristic_weights = heuristic_weights_8 if board_shape == 8 else heuristic_weights_4
    player_count = 0
    opponent_count = 0
    for row in range(board_shape):
        for col in range(board_shape):
            if game_state.board[row][col] == game_state.current_player:
                player_count += heuristic_weights[row][col]
            elif game_state.board[row][col] == opponent(game_state.current_player):
                opponent_count += heuristic_weights[row][col]
    return player_count - opponent_count

def terminal_test(game_state):
    return game_state.game_over

def utility(game_state):
    w = game_state.winner
    if w is None:
        return 0
    if w == game_state.current_player:
        return 1
    else:
        return -1


class MinimaxAgent(Agent):
    def __init__(self, player, max_depth, name="MinimaxAgent"):
        self.max_depth = max_depth
        super().__init__(player, AgentType.COMPUTER, name)

    def get_best_move(self, game_state):
        best_score = -float('inf')
        best_move = SkipPosition()
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
        if terminal_test(game_state):
            return utility(game_state)

        if level == 0:
            return heuristic_fn(game_state)

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

