import numpy as np
import torch
from torch import nn
from .base import Recommender, interaction_matrix


class FMNetwork(nn.Module):
    def __init__(self, n_features, factors):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(n_features, 1)
        self.factors = nn.Embedding(n_features, factors)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, std=0.03)

    def forward(self, features):
        embeddings = self.factors(features)
        interactions = 0.5 * (embeddings.sum(1).square() - embeddings.square().sum(1)).sum(1)
        return self.bias + self.linear(features).sum((1, 2)) + interactions


class FactorizationMachine(Recommender):
    def __init__(self, factors=32, learning_rate=0.01, weight_decay=1e-5,
                 epochs=10, negatives=3, batch_size=2048, seed=42, use_metadata=True):
        self.factors = factors
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.negatives = negatives
        self.batch_size = batch_size
        self.seed = seed
        self.use_metadata = use_metadata

    def _feature_ids(self, users, items):
        user_ids = torch.as_tensor(users, dtype=torch.long).reshape(-1, 1)
        item_ids = self.item_features[torch.as_tensor(items, dtype=torch.long)]
        return torch.cat((user_ids, item_ids), dim=1)

    def _sample_unseen(self, users, seen, rng):
        items = rng.integers(0, self.n_items, size=len(users))
        bad = np.asarray(seen[users, items]).ravel() != 0
        while bad.any():
            items[bad] = rng.integers(0, self.n_items, size=bad.sum())
            bad = np.asarray(seen[users, items]).ravel() != 0
        return items

    def fit(self, frame, shape, metadata=None, **kwargs):
        if metadata is None and self.use_metadata:
            raise ValueError("Metadata is required")
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.n_items = shape[1]
        features = [np.arange(shape[1]) + shape[0]]
        offset = sum(shape)
        if self.use_metadata:
            for field in ["brand", "category"]:
                values = metadata[field].fillna("unknown").astype(str)
                vocabulary = {value: j for j, value in enumerate(sorted(values.unique()))}
                features.append(values.map(vocabulary).to_numpy() + offset)
                offset += len(vocabulary)
        self.item_features = torch.as_tensor(np.column_stack(features), dtype=torch.long)
        self.network = FMNetwork(offset, self.factors)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        seen = interaction_matrix(frame, shape)
        if np.any(np.diff(seen.indptr) == shape[1]):
            raise ValueError("Cannot draw negatives for a user who saw the whole catalog")
        base_users = frame.u.to_numpy()
        base_items = frame.i.to_numpy()
        base_labels = (frame.rating.to_numpy() >= 4).astype(np.float32)
        positive_users = base_users[base_labels == 1]
        for _ in range(self.epochs):
            sampled_users = np.repeat(positive_users, self.negatives)
            sampled_items = self._sample_unseen(sampled_users, seen, rng)
            users = np.concatenate((base_users, sampled_users))
            items = np.concatenate((base_items, sampled_items))
            labels = torch.from_numpy(np.concatenate((base_labels, np.zeros(len(sampled_users), np.float32))))
            order = rng.permutation(len(users))
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                features = self._feature_ids(users[batch], items[batch])
                loss = criterion(self.network(features), labels[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self._prepare_scoring()
        return self

    @torch.no_grad()
    def _prepare_scoring(self):
        vectors = self.network.factors(self.item_features)
        self.item_vectors = vectors.sum(1).numpy()
        self.item_bias = (self.network.linear(self.item_features).sum((1, 2))
                          + 0.5 * (vectors.sum(1).square() - vectors.square().sum(1)).sum(1)).numpy()

    @torch.no_grad()
    def score(self, users):
        users = torch.as_tensor(users, dtype=torch.long)
        user_vectors = self.network.factors(users).numpy()
        user_bias = self.network.linear(users).numpy()
        return (user_vectors @ self.item_vectors.T + user_bias
                + self.item_bias[None, :] + self.network.bias.item())
