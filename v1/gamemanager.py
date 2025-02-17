from v1.agent import AgentType
from v1.gamestate import GameState
from v1.player import Player
from v1.position import Position

class GameManager:

    def __init__(self, agent_black, agent_white):
        self.game_state = GameState()
        self.agents = {Player.BLACK: agent_black, Player.WHITE: agent_white}

    def run(self):
        move_info = None
        while not self.game_state.game_over:
            if self.agents[self.game_state.current_player].agent_type == AgentType.PLAYER:
                move = self.get_user_move()
                move_info = self.game_state.make_move(move)
            else:
                move = self.agents[self.game_state.current_player].get_best_move(self.game_state)
                move_info = self.game_state.make_move(move)
        return self.game_state.winner

