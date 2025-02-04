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
        # q_values = self.model(torch.tensor(game_state_one_hot_encode(game_state), dtype=torch.float32).unsqueeze(0))
        # pos = position_one_hot_decode(
        #     q_values[0].argmax(axis=0).item())
        # if pos in legal_moves:
        #     return pos
        # return random.choice(legal_moves)

        q_values = self.model(torch.tensor(game_state_one_hot_encode(game_state), dtype=torch.float32).unsqueeze(0))
        # torch.tensor(legal_moves_mask(game_state.legal_actions), dtype=torch.float32)
        q_values[torch.tensor(legal_moves_mask(game_state.legal_moves_list), dtype=torch.float32).unsqueeze(0) == 0] = -float("inf")
        best_move = action_decode(q_values[0].argmax(axis=0).item())
        # best_move = action_decode(q_values[0].argmax(axis=0).item())

        # Select best move based on highest q-value taking into consideration only legal moves
        # legal_q_values = {move: q_values[0][one_hot_encoding_to_idx(position_one_hot_encode(move))].item() for move in legal_moves}
        # legal_q_values = {move: q_values[0][action_encode(move)].item() for move in legal_moves}
        # best_move = max(legal_q_values, key=legal_q_values.get)
        return best_move

        # q_values = self.model(
        #     torch.tensor(game_state_one_hot_encode(game_state), dtype=torch.float32).unsqueeze(0))
        # best_move = position_one_hot_decode(q_values[0].argmax(axis=0).item())
        # if best_move in legal_moves:
        #     return best_move
        # return random.choice(legal_moves)

