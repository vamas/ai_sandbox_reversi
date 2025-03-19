from collections import deque
import random
import numpy as np

# Replay Buffer with variable sampling_threshold based on buffer size
class ReplayBuffer:
    def __init__(self, size, refresh_threshold=0.2):
        # Allocate extra space as before.
        # refresh_threshold defines how many old experiences to delete
        self.buffer = deque(maxlen=int(size * 1.1))
        self.maxlen = size
        self.refresh_threshold = refresh_threshold

    @property
    def is_buffer_ready(self):
        return len(self.buffer) >= self.maxlen

    def compute_sampling_threshold(self):
        """
        Compute sampling_threshold based on the buffer size (self.maxlen):
          - Small buffer (<= 10,000): ~15 samples per experience.
          - Medium buffer (<= 50,000): ~8 samples per experience.
          - Large buffer (> 50,000): ~4 samples per experience.
        """
        if self.maxlen <= 10000:
            return 15
        elif self.maxlen <= 50000:
            return 8
        else:
            return 4

    def push(self, state, action, reward, next_state, done, legal_moves, sampling_threshold=None):
        # If not provided, compute sampling_threshold automatically based on buffer size.
        if sampling_threshold is None:
            sampling_threshold = self.compute_sampling_threshold()
        # Store each experience as a dictionary that includes a sample counter.
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'legal_moves': legal_moves,
            'sample_count': 0,
            'sampling_threshold': sampling_threshold
        }
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Returns an array of sample batches.
        The number of batches is determined by the recommended sampling threshold
        computed from the buffer size.
        For each batch, only eligible experiences (i.e. those that haven't been sampled
        more than their threshold) are considered.
        """
        # Determine how many rounds (batches) we want to sample.
        num_rounds = self.compute_sampling_threshold()
        batches = []
        for _ in range(num_rounds):
            # Filter for experiences that haven't reached their sampling threshold.
            eligible = [exp for exp in self.buffer if exp['sample_count'] < exp['sampling_threshold']]
            if not eligible:
                break  # No more eligible experiences to sample.
            # If fewer eligible experiences than the batch size, sample all eligible.
            if len(eligible) < batch_size:
                chosen = random.sample(eligible, len(eligible))
            else:
                chosen = random.sample(eligible, batch_size)
            # Update sample count for each chosen experience.
            for exp in chosen:
                exp['sample_count'] += 1
            # Extract each component from the chosen experiences.
            states = np.array([exp['state'] for exp in chosen])
            actions = np.array([exp['action'] for exp in chosen])
            rewards = np.array([exp['reward'] for exp in chosen])
            next_states = np.array([exp['next_state'] for exp in chosen])
            dones = np.array([exp['done'] for exp in chosen])
            legal_moves = np.array([exp['legal_moves'] for exp in chosen])
            # Append the batch (as a tuple of arrays) to the batches list.
            batches.append((states, actions, rewards, next_states, dones, legal_moves))

        # Delete old experiences
        self.delete_old_experiences()

        # Return the batches as a numpy array (this will be an object array if batches vary in length).
        return batches

    def delete_old_experiences(self):
        elements_to_delete = int(len(self.buffer) * self.refresh_threshold)
        for i in range(elements_to_delete):
            self.buffer.popleft()

