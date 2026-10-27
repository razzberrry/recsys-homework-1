import numpy as np
from .ials import ImplicitALS


class CustomTargetALS(ImplicitALS):
    def __init__(self, factors=32, alpha=20, regularization=1.0, iterations=12,
                 seed=42, half_life_days=365):
        super().__init__(factors, alpha, regularization, iterations, seed)
        self.half_life_days = half_life_days

    def make_targets(self, frame, cutoff=None):
        if cutoff is None:
            raise ValueError("The recommendation cutoff is required")
        age = (cutoff - frame.timestamp.to_numpy()) / 86400
        if np.any(age < 0):
            raise ValueError("Training contains interactions after the cutoff")
        recency = 0.5 + 0.5 * np.exp2(-age / self.half_life_days)
        return ((frame.rating.to_numpy() - 3) / 2) * recency
