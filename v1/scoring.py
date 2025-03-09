DEFAULT_SCORE = 1500

class ELOSystem:
    def __init__(self, k_factor=32):
        self.ratings = {}  # Store ELO ratings for each model version
        self.k_factor = k_factor

    def expected_score(self, rating_a, rating_b):
        """Compute expected score for player A."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, model_a, model_b, result_a):
        """Update ELO for both models."""
        ra, rb = self.ratings.get(model_a, DEFAULT_SCORE), self.ratings.get(model_b, DEFAULT_SCORE)
        expected_a = self.expected_score(ra, rb)
        expected_b = 1 - expected_a

        # Update ratings
        self.ratings[model_a] = ra + self.k_factor * (result_a - expected_a)
        self.ratings[model_b] = rb + self.k_factor * ((1 - result_a) - expected_b)
