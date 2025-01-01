from collections import deque
import random
import numpy as np

# Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self._is_buffer_ready = False
        self._push_counter = 0
        self._size = size

    @property
    def is_buffer_ready(self):
        return self._is_buffer_ready

    @property
    def data_points(self):
        return self._push_counter

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self._push_counter += 1
        if self._push_counter >= self._size:
            self._is_buffer_ready = True

    def sample(self, batch_size):
        self._is_buffer_ready = False
        self._push_counter = 0
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)
