from v1.agent import Agent, AgentType

class ManualAgent(Agent):
    def __init__(self, player):
        super().__init__(player, AgentType.PLAYER)

    def get_best_move(self, game_state):
        return None