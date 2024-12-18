import random

from v1.qtable import QTable


class DoubleQTable(QTable):
    def __init__(self, default_q_value=0):
        self.qtable_a = {}  # First Q-table (Q_A)
        self.qtable_b = {}  # Second Q-table (Q_B)
        super().__init__(default_q_value)

    @property
    def qtable_to_read(self):
        if self.qtable_flag:
            return self.qtable_a
        return self.qtable_b

    @property
    def qtable_to_write(self):
        if not self.qtable_flag:
            return self.qtable_a
        return self.qtable_b

    def get_q_value(self, state, action):
        if state not in self.qtable_to_read:
            self.qtable_to_read[state] = {}
        if action not in self.qtable_to_read[state]:
            self.qtable_to_read[state][action] = self.default_q_value
        return self.qtable_to_read[state][action]

    def get_q_value_a(self, state, action):
        if state not in self.qtable_a:
            self.qtable_a[state] = {}
        if action not in self.qtable_a[state]:
            self.qtable_a[state][action] = self.default_q_value
        return self.qtable_a[state][action]

    def get_q_value_b(self, state, action):
        if state not in self.qtable_b:
            self.qtable_b[state] = {}
        if action not in self.qtable_b[state]:
            self.qtable_b[state][action] = self.default_q_value
        return self.qtable_b[state][action]

    def update_q_value(self, state, action, value):
        # Randomly select one of the Q-tables for updating
        if random.random() < 0.5:
            if state not in self.qtable_a:
                self.qtable_a[state] = {}
            self.qtable_a[state][action] = value
            if state not in self.qtable_b:
                self.qtable_b[state][action] = self.default_q_value
        else:
            if state not in self.qtable_b:
                self.qtable_b[state] = {}
            self.qtable_b[state][action] = value
            if state not in self.qtable_a:
                self.qtable_a[state][action] = self.default_q_value

    def update_q_value_a(self, state, action, value):
        self.qtable_a[state][action] = value

    def update_q_value_b(self, state, action, value):
        self.qtable_b[state][action] = value

    def get_best_action(self, state):
        if state not in self.qtable_a and state not in self.qtable_b:
            return None

        # Combine Q-values from Q_A and Q_B to determine the best action
        combined_q_values = {}
        for action in set(self.qtable_a.get(state, {}).keys()).union(self.qtable_b.get(state, {}).keys()):
            combined_q_values[action] = self.get_q_value_a(state, action) + self.get_q_value_b(state, action)

        return max(combined_q_values.items(), key=lambda item: item[1])[0]

    def get_max_q_value(self, state):
        if state not in self.qtable_a and state not in self.qtable_b:
            return self.default_q_value

        # Combine Q-values from Q_A and Q_B to determine the max Q-value
        combined_q_values = []
        for action in set(self.qtable_a.get(state, {}).keys()).union(self.qtable_b.get(state, {}).keys()):
            combined_q_values.append(self.get_q_value_a(state, action) + self.get_q_value_b(state, action))

        return max(combined_q_values)

    def qtable_size(self):
        return sum(len(actions) for actions in self.qtable_a.values()) + sum(
            len(actions) for actions in self.qtable_b.values())

    def all_states(self):
        return set(self.qtable_a.keys()).union(self.qtable_b.keys())

    def update(self, state, action, reward, next_state, alpha, gamma):
        # Randomly select one of the Q-tables for updating
        if random.random() < 0.5:
            # Update Q_A
            next_best_action = self.get_best_action(next_state)
            next_q_b = self.get_q_value_b(next_state, next_best_action) if next_best_action is not None else 0
            updated_value = (1 - alpha) * self.get_q_value_a(state, action) + alpha * (reward + gamma * next_q_b)
            self.update_q_value_a(state, action, updated_value)
            if state not in self.qtable_b:
                self.qtable_b[state][action] = self.qtable_a[state][action]
        else:
            # Update Q_B
            next_best_action = self.get_best_action(next_state)
            next_q_a = self.get_q_value_a(next_state, next_best_action) if next_best_action is not None else 0
            updated_value = (1 - alpha) * self.get_q_value_b(state, action) + alpha * (reward + gamma * next_q_a)
            self.update_q_value_b(state, action, updated_value)
            if state not in self.qtable_a:
                self.qtable_a[state][action] = self.qtable_b[state][action]

