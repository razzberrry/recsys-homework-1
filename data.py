import gzip
import hashlib
import html
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TRAIN_END = int(pd.Timestamp("2017-01-01", tz="UTC").timestamp())
VALID_END = int(pd.Timestamp("2017-09-01", tz="UTC").timestamp())


def download_data(directory=None):
    directory = Path(directory or os.environ.get("RECSYS_DATA_DIR", ROOT / ".data"))
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    for filename, entry in manifest.items():
        path = directory / filename
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            print("Downloading", filename, flush=True)
            urllib.request.urlretrieve(entry["url"], temporary)
            temporary.replace(path)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}. Remove the file and download again.")
    return directory


def clean_ratings(raw):
    stats = {
        "raw_rows": len(raw), "raw_users": raw.user.nunique(), "raw_items": raw.item.nunique(),
        "missing_by_column": raw.isna().sum().to_dict(),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_user_item_rows": int(raw.duplicated(["user", "item"]).sum()),
    }
    valid = raw.dropna(subset=["user", "item", "rating", "timestamp"]).copy()
    valid = valid[valid.rating.between(1, 5) & (valid.timestamp > 0)]
    stats["invalid_or_missing_rows"] = len(raw) - len(valid)
    valid = (valid.sort_values(["timestamp", "rating", "user", "item"], kind="stable")
             .drop_duplicates(["user", "item"], keep="first").reset_index(drop=True))
    valid["timestamp"] = valid.timestamp.astype(np.int64)
    stats["clean_rows"] = len(valid)
    return valid, stats


def select_sample(clean, train_end=TRAIN_END, max_items=3000, min_user_events=3):
    past = clean[clean.timestamp < train_end]
    ranking = (past.groupby("item").size().rename("n").reset_index()
               .sort_values(["n", "item"], ascending=[False, True], kind="stable"))
    selected_items = ranking.head(max_items).item
    past = past[past.item.isin(selected_items)]
    counts = past.groupby("user").size()
    users = sorted(counts[counts >= min_user_events].index)
    past = past[past.user.isin(users)]
    items = sorted(past.item.unique())
    sample = clean[clean.user.isin(users) & clean.item.isin(items)].copy()
    sample["u"] = sample.user.map({value: i for i, value in enumerate(users)}).astype(np.int64)
    sample["i"] = sample.item.map({value: i for i, value in enumerate(items)}).astype(np.int64)
    return sample.reset_index(drop=True), users, items


def clean_text(value):
    if not isinstance(value, str):
        return "unknown"
    value = re.sub(r"<[^>]*>", " ", html.unescape(value))
    value = re.sub(r"\s+", " ", value).strip()
    return value if value else "unknown"


def load_metadata(path, items):
    wanted = set(items)
    records = {}
    rows = duplicates = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            obj = json.loads(line)
            rows += 1
            asin = obj.get("asin")
            if asin not in wanted:
                continue
            if asin in records:
                duplicates += 1
                continue
            categories = obj.get("category", [])
            category = categories[-1] if isinstance(categories, list) and categories else "unknown"
            records[asin] = {"title": clean_text(obj.get("title")),
                             "brand": clean_text(obj.get("brand")),
                             "category": clean_text(category)}
    metadata = pd.DataFrame([records.get(i, {"title": "unknown", "brand": "unknown",
                                              "category": "unknown"}) for i in items])
    stats = {"metadata_source_rows": rows, "metadata_repeated_selected_items": duplicates,
             "items_without_metadata": len(wanted - records.keys()),
             "unknown_by_field": metadata.eq("unknown").sum().to_dict(),
             "brands": metadata.brand.nunique(), "categories": metadata.category.nunique()}
    return metadata, stats


def prepare_data(directory):
    raw = pd.read_csv(Path(directory) / "Musical_Instruments.csv",
                      names=["item", "user", "rating", "timestamp"])
    clean, stats = clean_ratings(raw)
    sample, users, items = select_sample(clean)
    train = sample[sample.timestamp < TRAIN_END].copy()
    valid = sample[(sample.timestamp >= TRAIN_END) & (sample.timestamp < VALID_END)].copy()
    test = sample[sample.timestamp >= VALID_END].copy()
    metadata, meta_stats = load_metadata(Path(directory) / "meta_Musical_Instruments.json.gz", items)
    stats.update(meta_stats)
    stats.update({"sample_rows": len(sample), "users": len(users), "items": len(items),
                  "train_rows": len(train), "validation_rows": len(valid), "test_rows": len(test),
                  "sparsity": 1 - len(sample) / (len(users) * len(items)),
                  "positive_share": float((sample.rating >= 4).mean())})
    original_train = clean[clean.timestamp < TRAIN_END]
    exclusion_rows = []
    for label, start, stop in [("validation", TRAIN_END, VALID_END), ("test", VALID_END, np.inf)]:
        period = clean[(clean.timestamp >= start) & (clean.timestamp < stop)]
        warm = period[period.user.isin(original_train.user) & period.item.isin(original_train.item)]
        selected = period[period.user.isin(users) & period.item.isin(items)]
        exclusion_rows.append({"split": label, "all_clean_events": len(period),
                               "cold_user_or_item_vs_train": len(period) - len(warm),
                               "warm_but_outside_sample": len(warm) - len(selected),
                               "selected_events": len(selected),
                               "selected_users": selected.user.nunique(),
                               "users_with_positive": selected[selected.rating >= 4].user.nunique()})
    assert 100_000 <= len(sample) <= 2_000_000
    assert min(len(users), len(items)) >= 1000
    assert train.timestamp.max() < valid.timestamp.min() <= valid.timestamp.max() < test.timestamp.min()
    assert not sample.duplicated(["u", "i"]).any()
    return {
        "sample": sample,
        "train": train,
        "valid": valid,
        "test": test,
        "metadata": metadata,
        "shape": (len(users), len(items)),
        "stats": stats,
        "exclusions": pd.DataFrame(exclusion_rows).set_index("split"),
    }
