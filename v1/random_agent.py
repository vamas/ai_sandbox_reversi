import random
from v1.agent import Agent, AgentType

class RandomAgent(Agent):
    def __init__(self, player):
        super().__init__(player, AgentType.COMPUTER)

    def get_best_move(self, game_state):
        if not game_state.legal_moves.keys():
            return None
        legal_moves = list(game_state.legal_moves.keys())
        random_index = random.randint(0, len(legal_moves) - 1)
        return legal_moves[random_index]