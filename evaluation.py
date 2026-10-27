import numpy as np
import pandas as pd


def ranking_metrics(recommendations, relevant, k=10):
    hits = np.isin(recommendations[:k], list(relevant)).astype(float)
    discount = 1 / np.log2(np.arange(2, k + 2))
    ideal = discount[:min(k, len(relevant))].sum()
    if not len(relevant):
        raise ValueError("Metrics require at least one relevant item")
    return {"NDCG@10": float((hits * discount).sum() / ideal),
            "Recall@10": float(hits.sum() / len(relevant)),
            "HitRate@10": float(hits.any())}


def evaluate(model, future, seen, k=10, batch_size=128):
    relevant = future[future.rating >= 4].groupby("u").i.apply(set).to_dict()
    users = np.array(sorted(relevant), dtype=np.int64)
    if not len(users):
        raise ValueError("No eligible evaluation users")
    rows, all_recs = [], []
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        recommendations = model.recommend(batch, seen, k)
        for user, recs in zip(batch, recommendations):
            known = set(seen.indices[seen.indptr[user]:seen.indptr[user + 1]])
            assert not (known & relevant[user])
            assert len(set(recs)) == k and not (known & set(recs))
            rows.append({"u": user, **ranking_metrics(recs, relevant[user], k)})
        all_recs.append(recommendations)
    per_user = pd.DataFrame(rows).set_index("u")
    summary = per_user.mean().to_dict()
    summary["CatalogCoverage@10"] = len(np.unique(np.vstack(all_recs))) / seen.shape[1]
    summary["n_users"] = len(users)
    return summary, per_user


def paired_bootstrap(left, right, metric="NDCG@10", seed=42, repeats=2000):
    if not left.index.equals(right.index):
        raise ValueError("Paired comparison needs identical users")
    difference = (left[metric] - right[metric]).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.empty(repeats)
    for b in range(repeats):
        means[b] = difference[rng.integers(0, len(difference), len(difference))].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean_difference": float(difference.mean()), "ci95_low": float(low), "ci95_high": float(high)}
