import random

import torch

from v1.agent import Agent, AgentType
from v1.dqn_helpers import (BOARD_SHAPE, game_state_one_hot_encode, position_one_hot_encode,
                            action_decode, one_hot_encoding_to_idx, action_encode, legal_moves_mask)
from v1.position import SkipPosition

class DQNAgent(Agent):
    def __init__(self, player, model, name="DQNAgent"):
        super().__init__(player, AgentType.COMPUTER, name)
        self.model = model

    def get_best_move(self, game_state):
        legal_moves = list(game_state.legal_moves.keys())
        if legal_moves is None:
            # print("No legal moves available")
            return SkipPosition()
        if isinstance(legal_moves[0], SkipPosition):
            # print("No legal moves available")
            return SkipPosition()

        q_values = self.model(torch.tensor(game_state_one_hot_encode(game_state, game_state.current_player), dtype=torch.float32).unsqueeze(0))
        q_values[torch.tensor(legal_moves_mask(game_state.legal_moves_list), dtype=torch.float32).unsqueeze(0) == 0] = -float("inf")
        best_move = action_decode(q_values[0].argmax(axis=0).item())
        return best_move
