from enum import Enum
from abc import ABC, abstractmethod

class AgentType(Enum):
    PLAYER = 1
    COMPUTER = 2

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self): # Required to be usable as a key in a dictionary
        return hash(self.value)

class Agent(ABC):
    def __init__(self, player, agent_type):
        self.player = player
        self.agent_type = agent_type

    @abstractmethod
    def get_best_move(self, game_state):
        pass