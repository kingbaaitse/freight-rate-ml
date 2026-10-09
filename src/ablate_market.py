"""Ablation: does the 'second-order' market signal exist at all?

Compares, under the OFFICIAL time-based backtest (train Jan-Aug -> test Sep-Oct),
  A) strong-features-only model (no market_index / no date / no seasonal terms)
  B) same + market/day features

Run:  python src/ablate_market.py
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURES, LOG_TARGET, TARGET, apply_stats, fit_stats

BASE = ["distance", "log_distance", "inv_distance", "equipment", "enc_lane",
        "weight_clean", "weight_was_missing", "weight_was_negative", "weight_is_capped",
        "quote_signal"]
MARKET = ["market_index", "mi_daily_mean", "mi_row_dev", "mi_was_missing",
          "dow", "month", "doy", "is_weekend", "days_since_start",
          "pickup", "delivery", "lane", "enc_pickup", "enc_delivery"]

PARAMS = dict(objective="regression", metric="rmse", verbosity=-1, deterministic=True,
              seed=7, num_threads=4, learning_rate=0.05, num_leaves=63,
              min_child_samples=60, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1)


def mape(y, p):
    return float(np.mean(np.abs(y - p) / y))


def run(feats: list[str]) -> float:
    tr = pd.read_csv("data/train_test.csv", parse_dates=["date"])
    dev = tr[tr["date"] < "2025-09-01"].reset_index(drop=True)
    tst = tr[tr["date"] >= "2025-09-01"].reset_index(drop=True)
    stats = fit_stats(dev)
    dtr, dte = apply_stats(dev, stats, is_train=True), apply_stats(tst, stats)
    dtr[LOG_TARGET] = np.log(dtr[TARGET].astype(float))
    cats = [c for c in feats if dtr[c].dtype.name == "category"]
    m = lgb.train(PARAMS, lgb.Dataset(dtr[feats], label=dtr[LOG_TARGET], categorical_feature=cats),
                  num_boost_round=1500,
                  valid_sets=[lgb.Dataset(dte[feats], label=np.log(dte[TARGET].astype(float)), categorical_feature=cats)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = np.exp(m.predict(dte[feats], num_iteration=m.best_iteration))
    return mape(dte[TARGET].to_numpy(float), pred)


if __name__ == "__main__":
    a = run(BASE)
    b = run(sorted(set(BASE + MARKET)))
    print(f"A) no-market/day features : MAPE = {a:.4%}")
    print(f"B) with market/day        : MAPE = {b:.4%}")
    print(f"delta (B-A, negative wins): {(b - a):+.4%}")
