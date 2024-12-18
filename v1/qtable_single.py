from v1.qtable import QTable


class SingleQTable(QTable):
    def __init__(self, default_q_value=0):
        self.qtable = {}
        super().__init__(default_q_value)

    @property
    def qtable_to_read(self):
        return self.qtable

    @property
    def qtable_to_write(self):
        return self.qtable

    def get_q_value(self, state, action):
        if state not in self.qtable_to_read:
            self.qtable[state] = {}
        if action not in self.qtable_to_read[state]:
            self.qtable_to_read[state][action] = self.default_q_value
        return self.qtable_to_read[state][action]

    def update_q_value(self, state, action, value):
        self.qtable_to_write[state][action] = value

    def get_best_action(self, state):
        if state not in self.qtable:
            return None
        return max(self.qtable[state].items(), key=lambda item: item[1])[0]

    def get_max_q_value(self, state):
        if state not in self.qtable:
            return self.default_q_value
        return max(self.qtable[state].values())

    def qtable_size(self):
        return sum(len(actions) for actions in self.qtable.values())

    def all_states(self):
        return self.qtable.keys()