import json
import time
from pathlib import Path

import pandas as pd

from data import TRAIN_END, VALID_END
from evaluation import evaluate
from models import ItemKNN, ImplicitALS, EASE, FactorizationMachine, CustomTargetALS
from models.base import interaction_matrix

SEED = 42
MODEL_CLASSES = {"ItemKNN": ItemKNN, "iALS": ImplicitALS, "EASE": EASE,
                 "FM": FactorizationMachine, "CustomALS": CustomTargetALS}
SEARCH = {
    "ItemKNN": [dict(neighbors=k, shrinkage=s) for k, s in [(30, 0), (50, 10), (100, 30)]],
    "iALS": [dict(factors=f, alpha=a, regularization=r, iterations=12, seed=SEED)
              for f, a, r in [(32, 10, 1), (32, 40, 5), (64, 20, 5)]],
    "EASE": [dict(regularization=r) for r in [30, 100, 300]],
    "FM": [dict(factors=f, learning_rate=0.01, weight_decay=w, epochs=e,
                 negatives=3, batch_size=2048, seed=SEED, use_metadata=True)
           for f, w, e in [(16, 1e-5, 8), (32, 1e-5, 12), (32, 1e-4, 8)]],
}


def tune(name, configs, dataset):
    seen = interaction_matrix(dataset["train"], dataset["shape"])
    rows = []
    for config_id, config in enumerate(configs):
        start = time.perf_counter()
        model = MODEL_CLASSES[name](**config).fit(dataset["train"], dataset["shape"],
                                                 metadata=dataset["metadata"], cutoff=TRAIN_END)
        fitted = time.perf_counter()
        scores, _ = evaluate(model, dataset["valid"], seen)
        row = {"model": name, "config_id": config_id, "params": json.dumps(config, sort_keys=True),
               "fit_seconds": fitted - start, "evaluation_seconds": time.perf_counter() - fitted, **scores}
        rows.append(row)
        print(f"{name} {config_id + 1}/{len(configs)}: NDCG@10={scores['NDCG@10']:.5f}, "
              f"Recall@10={scores['Recall@10']:.5f}, fit={row['fit_seconds']:.1f}s", flush=True)
    table = pd.DataFrame(rows)
    best = table["NDCG@10"].idxmax()
    return table, configs[best].copy()


def final_comparison(dataset, chosen, directory="results"):
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    development = pd.concat([dataset["train"], dataset["valid"]], ignore_index=True)
    seen = interaction_matrix(development, dataset["shape"])
    rows, per_users = [], {}
    for name, config in chosen.items():
        start = time.perf_counter()
        model = MODEL_CLASSES[name](**config).fit(development, dataset["shape"],
                                                 metadata=dataset["metadata"], cutoff=VALID_END)
        fitted = time.perf_counter()
        scores, per_user = evaluate(model, dataset["test"], seen)
        rows.append({"model": name, **scores, "fit_seconds": fitted - start,
                     "evaluation_seconds": time.perf_counter() - fitted,
                     "params": json.dumps(config, sort_keys=True)})
        per_users[name] = per_user
        per_user.to_csv(directory / f"per_user_{name}.csv")
        print(f"Test {name}: NDCG@10={scores['NDCG@10']:.5f}, Recall@10={scores['Recall@10']:.5f}", flush=True)
    return pd.DataFrame(rows).set_index("model"), per_users


def metadata_ablation(dataset, fm_config):
    config = {**fm_config, "use_metadata": False}
    development = pd.concat([dataset["train"], dataset["valid"]], ignore_index=True)
    model = FactorizationMachine(**config).fit(development, dataset["shape"], cutoff=VALID_END)
    scores, per_user = evaluate(model, dataset["test"], interaction_matrix(development, dataset["shape"]))
    return scores, per_user
