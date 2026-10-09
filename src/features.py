"""Shared feature engineering for the freight-rate challenge.

Every transform is fit on TRAIN ONLY (see `fit_stats`) and applied to
validation / december batches via `apply_stats` -- no leakage.
One deliberate transductive step: the per-date market mean is computed
within the batch being featurized (train batch, validation batch). The
scoring batch always contains ~150-200 rows per date, so this mirrors
production use of a date-level market regime feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "posted_rate"
LOG_TARGET = "log_rate"
EQUIP_ORDER = ["Dry Van", "Flatbed", "Reefer"]

CAT_COLS = ["equipment", "pickup", "delivery", "lane", "dow"]


def _clean_weight(frame: pd.DataFrame, equip_median: pd.Series | None) -> pd.DataFrame:
    frame = frame.copy()
    w = frame["weight"].astype(float)
    frame["weight_was_negative"] = (w < 0).astype(np.int8)  # sign-entry errors
    w = w.abs()  # abs(neg) matches pos distribution (mean 31724 vs 31415)
    frame["weight_was_missing"] = w.isna().astype(np.int8)
    if equip_median is not None:
        w = w.fillna(frame["equipment"].map(equip_median))
    frame["weight_clean"] = w.astype(float)
    frame["weight_is_capped"] = (frame["weight_clean"] == 47500.0).astype(np.int8)
    return frame


def _date_parts(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["dow"] = frame["date"].dt.dayofweek.astype(str)
    frame["doy"] = frame["date"].dt.dayofyear.astype(float)
    frame["month"] = frame["date"].dt.month.astype(str)
    frame["is_weekend"] = (frame["date"].dt.dayofweek >= 5).astype(np.int8)
    frame["days_since_start"] = (frame["date"] - pd.Timestamp("2025-01-01")).dt.days.astype(float)
    return frame


def fit_stats(train: pd.DataFrame) -> dict:
    """Learn everything the model needs from the train frame only."""
    train = _date_parts(_clean_weight(train, None))
    # equipment weight medians use cleaned (abs) weights
    equip_wmedian = train.groupby("equipment")["weight_clean"].median()

    train["log_rate"] = np.log(train[TARGET].astype(float))
    train["rpm"] = train[TARGET].astype(float) / train["distance"].astype(float)
    train["log_rpm"] = np.log(train["rpm"])
    global_log_rpm = float(train["log_rpm"].mean())
    equip_log_rpm = train.groupby("equipment")["log_rpm"].mean().to_dict()

    def smooth(series_map: pd.Series, m: float, prior: pd.Series | float) -> dict:
        stats = train.groupby(series_map)["log_rpm"].agg(["sum", "count"])
        prior_vals = prior if isinstance(prior, float) else stats.index.map(prior).fillna(global_log_rpm)
        return (((stats["sum"] + m * prior_vals) / (stats["count"] + m))).to_dict()

    train["lane"] = train["pickup"] + ">" + train["delivery"]
    equip_prior = train["equipment"].map(equip_log_rpm)
    stats = {
        "equip_wmedian": equip_wmedian,
        "global_log_rpm": global_log_rpm,
        "equip_log_rpm": equip_log_rpm,
        # shrinkage strengths scale with observation counts
        "enc_pickup": smooth(train["pickup"], 100.0, global_log_rpm),
        "enc_delivery": smooth(train["delivery"], 100.0, global_log_rpm),
        "enc_lane": smooth(train["lane"], 50.0, equip_prior),
        "dow_mi": train.groupby(train["date"].dt.dayofweek)["market_index"].mean().to_dict(),
        "global_mi": float(train["market_index"].median()),
    }
    return stats


def apply_stats(frame: pd.DataFrame, stats: dict, is_train: bool = False) -> pd.DataFrame:
    frame = _date_parts(_clean_weight(frame, stats["equip_wmedian"]))
    frame["lane"] = frame["pickup"] + ">" + frame["delivery"]

    # --- market_index: daily batch mean (regime) + row deviation (idiosyncratic)
    mi = frame["market_index"].astype(float)
    frame["mi_was_missing"] = mi.isna().astype(np.int8)
    day_med = frame.groupby(frame["date"].dt.date)["market_index"].transform("median")
    mi = mi.fillna(day_med).fillna(stats["global_mi"])
    frame["market_index"] = mi
    frame["mi_daily_mean"] = frame.groupby(frame["date"].dt.date)["market_index"].transform("mean")
    frame["mi_row_dev"] = frame["market_index"] - frame["mi_daily_mean"]

    # --- core physics: rate ~= distance x rpm(distance, equipment)
    d = frame["distance"].astype(float)
    frame["log_distance"] = np.log(d)
    frame["inv_distance"] = 1.0 / d

    # --- graceful fallbacks for unseen cities / lanes (12% of val lanes are new)
    frame["enc_pickup"] = frame["pickup"].map(stats["enc_pickup"]).fillna(stats["global_log_rpm"])
    frame["enc_delivery"] = frame["delivery"].map(stats["enc_delivery"]).fillna(stats["global_log_rpm"])
    lane_prior = frame["equipment"].map(stats["equip_log_rpm"]).fillna(stats["global_log_rpm"])
    frame["enc_lane"] = frame["lane"].map(stats["enc_lane"]).fillna(lane_prior)

    frame["equipment"] = pd.Categorical(frame["equipment"], categories=EQUIP_ORDER)
    frame["pickup"] = frame["pickup"].astype("category")
    frame["delivery"] = frame["delivery"].astype("category")
    frame["lane"] = frame["lane"].astype("category")
    frame["dow"] = frame["dow"].astype("category")
    frame["month"] = frame["month"].astype("category")
    return frame


FEATURES = [
    "distance", "inv_distance",
    "equipment",
    "enc_lane", "enc_pickup", "enc_delivery",
    "weight_clean",
    "quote_signal",
]
