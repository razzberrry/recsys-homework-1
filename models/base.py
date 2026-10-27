import numpy as np
from scipy import sparse


def interaction_matrix(frame, shape, values=None):
    if values is None:
        values = np.ones(len(frame), dtype=np.float64)
    return sparse.csr_matrix((values, (frame.u, frame.i)), shape=shape)


class Recommender:
    def recommend(self, users, seen, k=10):
        scores = np.asarray(self.score(users), dtype=np.float64).copy()
        if not np.isfinite(scores).all():
            raise ValueError("Model returned non-finite scores")
        for row, user in enumerate(users):
            known = seen.indices[seen.indptr[user]:seen.indptr[user + 1]]
            if scores.shape[1] - len(known) < k:
                raise ValueError("Fewer than k unseen candidates")
            scores[row, known] = -np.inf
        return np.argsort(-scores, axis=1, kind="stable")[:, :k]
