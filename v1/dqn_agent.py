import random

import torch

from v1.agent import Agent, AgentType
from v1.dqn_train import position_encode, game_state_encode
from v1.position import SkipPosition
from v1.qtable_train import game_state_hash

class DQNAgent(Agent):
    def __init__(self, player, model):
        super().__init__(player, AgentType.COMPUTER)
        self.model = model

    def get_best_move(self, game_state):
        legal_moves = list(game_state.legal_moves.keys())
        if legal_moves is None:
            print("No legal moves available")
            return SkipPosition()
        if isinstance(legal_moves[0], SkipPosition):
            print("No legal moves available")
            return SkipPosition()
        legal_moves_idx = [position_encode(move) for move in legal_moves]
        q_values = self.model(torch.tensor(game_state_encode(game_state), dtype=torch.float32).unsqueeze(0))
        pos = q_values[0][:(game_state.Cols * game_state.Rows)][legal_moves_idx].argmax(axis=0).item()
        return legal_moves[pos]

