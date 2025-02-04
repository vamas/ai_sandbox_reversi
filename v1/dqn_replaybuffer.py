from collections import deque
import random
import numpy as np

# Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=int(size*1.1))
        self.maxlen = size
        self._is_buffer_ready = False

    @property
    def is_buffer_ready(self):
        return len(self.buffer) >= self.maxlen

    def push(self, state, action, reward, next_state, done, legal_moves):
        self.buffer.append((state, action, reward, next_state, done, legal_moves))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, legal_moves = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), np.array(legal_moves))

    def size(self):
        return len(self.buffer)

    def purge(self):
        self.buffer.clear()
        self._is_buffer_ready = False
