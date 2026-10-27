import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .base import Recommender, interaction_matrix


class EASE(Recommender):
    def __init__(self, regularization=100):
        self.regularization = regularization

    def fit(self, frame, shape, **kwargs):
        self.history = interaction_matrix(frame[frame.rating >= 4], shape)
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::shape[1] + 1] += self.regularization
        precision = cho_solve(cho_factor(gram, check_finite=False),
                              np.eye(shape[1]), check_finite=False)
        self.weights = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights, 0)
        return self

    def score(self, users):
        return np.asarray(self.history[users] @ self.weights)
