import random
from v1.agent import Agent, AgentType
from v1.position import SkipPosition

def game_state_hash(game_state):
    return game_state.hash()

class DoubleQTableAgent(Agent):
    def __init__(self, player, qtable):
        super().__init__(player, AgentType.COMPUTER)
        self.qtable = qtable

    def get_best_move(self, game_state):
        # TODO: Implement the get_best_move method for the QTableAgent choosing the best move based on the Q-table.
        if not game_state.legal_moves.keys():
            return SkipPosition()
        if len(game_state.legal_moves.keys()) == 1:
            return SkipPosition()
        # if game_state.hash() not in self.qtable.qtable:
        #     legal_moves = list(game_state.legal_moves.keys())
        #     random_index = random.randint(0, len(legal_moves) - 1)
        #     return legal_moves[random_index]
        # return game_state.lookup_legal_action(
        #     (self.qtable.get_best_action(game_state.hash()))[0])

        if game_state_hash(game_state) not in self.qtable.all_states():
            legal_moves = list(game_state.legal_moves.keys())
            random_index = random.randint(0, len(legal_moves) - 1)
            return legal_moves[random_index]

        return self.qtable.get_best_action(game_state_hash(game_state))


