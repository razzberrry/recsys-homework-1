import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .base import Recommender, interaction_matrix


class ImplicitALS(Recommender):
    def __init__(self, factors=32, alpha=20, regularization=1.0, iterations=12, seed=42):
        self.factors = factors
        self.alpha = alpha
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed

    def make_targets(self, frame, cutoff=None):
        return (frame.rating.to_numpy() >= 4).astype(np.float64)

    def fit(self, frame, shape, cutoff=None, **kwargs):
        targets = self.make_targets(frame, cutoff)
        target_matrix = interaction_matrix(frame, shape, targets)
        target_matrix.sort_indices()
        item_targets = target_matrix.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.01, (shape[0], self.factors))
        self.item_factors = rng.normal(0, 0.01, (shape[1], self.factors))
        for _ in range(self.iterations):
            self.user_factors = self._update(target_matrix, self.item_factors)
            self.item_factors = self._update(item_targets, self.user_factors)
        return self

    def _update(self, targets, fixed):
        base = fixed.T @ fixed + self.regularization * np.eye(self.factors)
        result = np.zeros((targets.shape[0], self.factors))
        for row in range(targets.shape[0]):
            start, end = targets.indptr[row:row + 2]
            indices = targets.indices[start:end]
            if not len(indices):
                continue
            vectors = fixed[indices]
            lhs = base + self.alpha * (vectors.T @ vectors)
            rhs = vectors.T @ ((1 + self.alpha) * targets.data[start:end])
            result[row] = cho_solve(cho_factor(lhs, check_finite=False), rhs,
                                     check_finite=False)
        return result

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
