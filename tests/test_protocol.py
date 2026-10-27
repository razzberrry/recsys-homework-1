import numpy as np
import pandas as pd

from data import clean_ratings, select_sample
from evaluation import ranking_metrics, paired_bootstrap


def test_metrics_on_known_ranking():
    result = ranking_metrics(np.arange(10), {0, 2, 15})
    assert np.isclose(result["Recall@10"], 2 / 3)
    assert np.isclose(result["NDCG@10"], (1 + 1 / np.log2(4)) /
                      (1 + 1 / np.log2(3) + 1 / np.log2(4)))
    assert result["HitRate@10"] == 1


def test_future_cannot_change_training_selection_or_index_mapping():
    past = pd.DataFrame({"user": ["u1", "u1", "u2", "u2"], "item": ["a", "b", "b", "c"],
                         "rating": [4] * 4, "timestamp": [1, 2, 3, 4]})
    future = pd.DataFrame({"user": ["new", "u1"], "item": ["new", "new"],
                           "rating": [5, 5], "timestamp": [11, 12]})
    first, users, items = select_sample(past, train_end=10, max_items=2, min_user_events=1)
    second, users2, items2 = select_sample(pd.concat([past, future]), train_end=10,
                                         max_items=2, min_user_events=1)
    assert users == users2 and items == items2
    pd.testing.assert_frame_equal(first, second)


def test_deduplication_keeps_first_observation():
    raw = pd.DataFrame({"user": ["a", "a", "b", "b"], "item": ["i"] * 4,
                        "rating": [2, 5, 5, 1], "timestamp": [1, 3, 2, 2]})
    clean, _ = clean_ratings(raw)
    assert clean.set_index("user").rating.to_dict() == {"a": 2, "b": 1}


def test_bootstrap_identical_models_has_zero_interval():
    values = pd.DataFrame({"NDCG@10": [0, 0.2, 0.5]})
    result = paired_bootstrap(values, values, repeats=20)
    assert all(value == 0 for value in result.values())
