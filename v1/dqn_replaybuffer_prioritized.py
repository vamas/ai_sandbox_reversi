import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.2):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = uniform, 1 = full prioritization)
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0  # Circular buffer index

    @property
    def is_buffer_ready(self):
        return len(self.buffer) >= self.capacity

    def push(self, state, action, reward, next_state, done, legal_moves):
        """Stores a new transition with the highest priority"""
        max_priority = self.priorities.max() if self.buffer else 1.0  # Start with highest priority
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done, legal_moves))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done, legal_moves)

        self.priorities[self.position] = max_priority  # Assign high priority
        self.position = (self.position + 1) % self.capacity  # Circular buffer

    def sample(self, batch_size, beta=0.4):
        """Samples a batch of transitions, weighted by priority"""
        if len(self.buffer) == 0:
            return []

        # Compute probabilities
        priorities = self.priorities[:len(self.buffer)] ** self.alpha
        probs = priorities / priorities.sum()  # Normalize

        # Sample indices based on priority probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Compute importance-sampling weights (to compensate for bias)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        self.priorities[indices] = np.abs(td_errors) + 1e-5  # Small epsilon to prevent zero probability

    def purge(self):
        self.buffer.clear()
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.position = 0