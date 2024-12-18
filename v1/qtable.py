from abc import ABC, abstractmethod


class QTable(ABC):

    def __init__(self, default_q_value):
        self.default_q_value = default_q_value
        self._qtable_flag = False

    @property
    def qtable_flag(self):
        return self._qtable_flag

    @qtable_flag.setter
    def qtable_flag(self, flag):
        self._qtable_flag = flag

    @property
    @abstractmethod
    def qtable_to_read(self):
        pass

    @property
    @abstractmethod
    def qtable_to_write(self):
        pass

    @abstractmethod
    def get_q_value(self, state, action):
        pass

    @abstractmethod
    def update_q_value(self, state, action, value):
        pass

    @abstractmethod
    def get_best_action(self, state):
        pass

    @abstractmethod
    def all_states(self):
        pass

    @abstractmethod
    def get_max_q_value(self, state):
        pass
