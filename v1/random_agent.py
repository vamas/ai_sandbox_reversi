import random

from v1.position import SkipPosition
from v1.agent import Agent, AgentType

class RandomAgent(Agent):
    def __init__(self, player, name="RandomAgent"):
        super().__init__(player, AgentType.COMPUTER, name)

    def get_best_move(self, game_state):
        return random.choice(game_state.legal_moves_list)