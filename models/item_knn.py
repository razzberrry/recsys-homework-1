import numpy as np
from scipy import sparse
from .base import Recommender, interaction_matrix


class ItemKNN(Recommender):
    def __init__(self, neighbors=50, shrinkage=10):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, frame, shape, **kwargs):
        self.history = interaction_matrix(frame[frame.rating >= 4], shape)
        counts = (self.history.T @ self.history).toarray()
        norm = np.sqrt(np.diag(counts))
        denominator = np.outer(norm, norm)
        similarity = np.divide(counts, denominator, out=np.zeros_like(counts),
                               where=denominator > 0)
        if self.shrinkage:
            similarity *= counts / (counts + self.shrinkage)
        np.fill_diagonal(similarity, 0)
        rows, cols, values = [], [], []
        for target in range(shape[1]):
            order = np.argsort(-similarity[:, target], kind="stable")[:self.neighbors]
            order = order[similarity[order, target] > 0]
            rows.extend(order)
            cols.extend([target] * len(order))
            values.extend(similarity[order, target])
        self.weights = sparse.csc_matrix((values, (rows, cols)), shape=(shape[1], shape[1]))
        return self

    def score(self, users):
        return (self.history[users] @ self.weights).toarray()
