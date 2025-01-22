from v1.agent import Agent, AgentType

class ManualAgent(Agent):
    def __init__(self, player, name="ManualAgent"):
        super().__init__(player, AgentType.PLAYER, name)

    def get_best_move(self, game_state):
        return None