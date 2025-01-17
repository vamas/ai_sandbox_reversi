from collections import deque
import random
import numpy as np

# Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self._is_buffer_ready = False

    @property
    def is_buffer_ready(self):
        return len(self.buffer) >= self.buffer.maxlen

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def size(self):
        return len(self.buffer)

    def purge(self):
        self.buffer.clear()
        self._is_buffer_ready = False
