import numpy as np
import pandas as pd
import torch
from scipy import sparse

from models import ItemKNN, ImplicitALS, EASE, FactorizationMachine, CustomTargetALS
from models.base import interaction_matrix
from models.fm import FMNetwork


def example():
    return pd.DataFrame({"u": [0, 0, 1, 1, 2, 2], "i": [0, 1, 1, 2, 0, 3],
                         "rating": [5, 2, 4, 5, 4, 1], "timestamp": [10, 11, 12, 13, 14, 15]})


def test_als_update_matches_dense_weighted_least_squares():
    model = ImplicitALS(factors=3, alpha=7, regularization=0.8)
    fixed = np.random.default_rng(9).normal(size=(4, 3))
    target = sparse.csr_matrix(([1., 0., -0.5], ([0, 0, 0], [0, 1, 3])), shape=(1, 4))
    confidence = np.diag([8., 8., 1., 8.])
    dense_target = np.array([1., 0., 0., -0.5])
    expected = np.linalg.solve(fixed.T @ confidence @ fixed + 0.8 * np.eye(3),
                               fixed.T @ confidence @ dense_target)
    np.testing.assert_allclose(model._update(target, fixed)[0], expected, atol=1e-12)


def test_ease_satisfies_constrained_normal_equations():
    frame = example()
    model = EASE(regularization=3).fit(frame, (3, 4))
    x = interaction_matrix(frame[frame.rating >= 4], (3, 4)).toarray()
    gradient = (x.T @ x + 3 * np.eye(4)) @ model.weights - x.T @ x
    np.fill_diagonal(gradient, 0)
    np.testing.assert_allclose(gradient, 0, atol=1e-12)
    np.testing.assert_allclose(np.diag(model.weights), 0)


def test_item_knn_cosine_and_column_neighborhoods():
    model = ItemKNN(neighbors=1, shrinkage=0).fit(example(), (3, 4))
    weights = model.weights.toarray()
    assert (np.count_nonzero(weights, axis=0) <= 1).all()
    assert weights[1, 2] == weights[2, 1] == 1
    np.testing.assert_array_equal(np.diag(weights), 0)


def test_fm_formula_matches_explicit_pair_sum():
    torch.manual_seed(12)
    net = FMNetwork(10, 3)
    ids = torch.tensor([[0, 3, 6, 8], [1, 4, 6, 9]])
    expected = net.bias + net.linear(ids).sum((1, 2))
    for a in range(ids.shape[1]):
        for b in range(a + 1, ids.shape[1]):
            expected = expected + (net.factors(ids[:, a]) * net.factors(ids[:, b])).sum(1)
    torch.testing.assert_close(net(ids), expected)


def test_fm_optimized_scoring_and_sampling():
    torch.set_num_threads(1)
    frame = example()
    meta = pd.DataFrame({"brand": ["a", "b", "a", "c"], "category": ["x", "y", "y", "z"]})
    model = FactorizationMachine(factors=3, epochs=1, batch_size=4).fit(frame, (3, 4), metadata=meta)
    u, i = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
    with torch.no_grad():
        direct = model.network(model._feature_ids(u.ravel(), i.ravel())).numpy().reshape(3, 4)
    np.testing.assert_allclose(model.score(np.arange(3)), direct, atol=1e-7)
    users = np.repeat(np.arange(3), 100)
    seen = interaction_matrix(frame, (3, 4))
    sampled = model._sample_unseen(users, seen, np.random.default_rng(1))
    assert not np.asarray(seen[users, sampled]).any()


def test_custom_target_is_not_just_confidence_reweighting():
    frame = pd.DataFrame({"rating": [1, 3, 4, 5], "timestamp": [0, 0, 0, 86400 * 10]})
    model = CustomTargetALS(half_life_days=10)
    actual = model.make_targets(frame, cutoff=86400 * 10)
    np.testing.assert_allclose(actual, [-0.75, 0, 0.375, 1])
    assert model.alpha == ImplicitALS().alpha


def test_recommendations_exclude_all_observed_ratings_and_are_deterministic():
    frame = example()
    seen = interaction_matrix(frame, (3, 4))
    for cls in [ItemKNN, EASE, ImplicitALS, CustomTargetALS]:
        model = cls().fit(frame, (3, 4), cutoff=20)
        recs = model.recommend(np.arange(3), seen, k=2)
        for user, items in enumerate(recs):
            assert not set(items) & set(frame[frame.u == user].i)
        np.testing.assert_array_equal(recs, model.recommend(np.arange(3), seen, k=2))
