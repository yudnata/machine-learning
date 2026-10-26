from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.preprocessing import StandardScaler

from .common import ROOT, file_hash, read_config, require, signature, write_json


def clean_events(raw, cfg):
    data = raw.copy()
    for col in ["latitude", "longitude", "depth", "magnitude", "geometry_z"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    data["time"] = pd.to_datetime(data["time"], errors="coerce", utc=True, format="mixed")
    box = cfg["bbox"]
    valid = (data["time"].notna() & data["latitude"].between(box["min_lat"], box["max_lat"])
             & data["longitude"].between(box["min_lon"], box["max_lon"])
             & data["time"].ge(pd.Timestamp(cfg["train_start"], tz="UTC"))
             & data["time"].lt(pd.Timestamp(cfg["test_end_exclusive"], tz="UTC")))
    valid_type = data["event_type"].isin(cfg["accepted_event_types"])
    rejected_type = data.loc[valid & ~valid_type].copy()
    rejected_type["reason"] = "event_type_not_accepted"
    rejected = data.loc[~valid].copy()
    rejected["reason"] = "invalid_time_coordinate_or_outside_scope"
    data = data.loc[valid & valid_type].copy()
    data["lastupdate"] = pd.to_datetime(data["lastupdate"], errors="coerce", utc=True, format="mixed")
    fallback = data[["time", "latitude", "longitude", "magnitude"]].astype(str).agg("|".join, axis=1)
    id_values = data["eventid"].fillna("").astype(str).str.strip()
    data["dedup_key"] = np.where(id_values.ne(""), "id:" + id_values, "fallback:" + fallback)
    data = data.sort_values("lastupdate", na_position="first")
    duplicates = data.loc[data.duplicated("dedup_key", keep="last")].copy()
    duplicates["reason"] = "duplicate_older_revision"
    data = data.drop_duplicates("dedup_key", keep="last")
    bad_depth = ~data["depth"].between(0, cfg["max_depth_km"])
    bad_magnitude = ~data["magnitude"].between(-2, 10)
    data["depth_invalid"] = bad_depth
    data["magnitude_invalid"] = bad_magnitude
    data.loc[bad_depth, "depth"] = np.nan
    data.loc[bad_magnitude, "magnitude"] = np.nan
    data["year"] = data["time"].dt.year
    data["period"] = data["time"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    data["event_key"] = data["dedup_key"]
    rejected_parts = [frame for frame in [rejected, rejected_type, duplicates] if not frame.empty]
    rejected_all = pd.concat(rejected_parts, ignore_index=True) if rejected_parts else rejected.iloc[:0].copy()
    return data.sort_values("time").reset_index(drop=True), rejected_all


def grid_panel(events, cfg, grid_km=None):
    grid_km = grid_km or cfg["grid_km"]
    size = grid_km * 1000
    data = events.loc[events["magnitude"].ge(cfg["min_magnitude"])].copy()
    if data.empty:
        raise ValueError("Tidak ada event di atas ambang magnitudo.")
    forward = Transformer.from_crs("EPSG:4326", cfg["projection"], always_xy=True)
    inverse = Transformer.from_crs(cfg["projection"], "EPSG:4326", always_xy=True)
    x, y = forward.transform(data["longitude"].to_numpy(), data["latitude"].to_numpy())
    data["grid_x"] = np.floor(np.asarray(x) / size).astype(int)
    data["grid_y"] = np.floor(np.asarray(y) / size).astype(int)
    data["grid_id"] = data["grid_x"].astype(str) + "_" + data["grid_y"].astype(str)
    data["shallow"] = np.where(data["depth"].notna(), data["depth"].lt(cfg["shallow_depth_km"]).astype(float), np.nan)
    train_end = pd.Timestamp(cfg["train_end_exclusive"], tz="UTC")
    historic_ids = set(data.loc[data["time"].lt(train_end), "grid_id"])
    if not historic_ids:
        raise ValueError("Tidak ada grid training.")
    all_grids = data[["grid_id", "grid_x", "grid_y"]].drop_duplicates("grid_id")
    aggregate = data.groupby(["grid_id", "period"], observed=True).agg(
        earthquake_count=("event_key", "size"), avg_magnitude=("magnitude", "mean"),
        max_magnitude=("magnitude", "max"), avg_depth=("depth", "mean"),
        depth_count=("depth", "count"), shallow_count=("shallow", "sum")).reset_index()
    start = pd.Period(cfg["train_start"], freq="Q")
    end_ts = pd.Timestamp(cfg["test_end_exclusive"], tz="UTC")
    last = (end_ts - pd.Timedelta(microseconds=1)).tz_localize(None).to_period("Q")
    period_rows = []
    for p in pd.period_range(start, last, freq="Q"):
        a = p.start_time.tz_localize("UTC")
        b = (p + 1).start_time.tz_localize("UTC")
        observed_end = min(b, end_ts)
        complete = observed_end == b
        period_rows.append({"period": str(p), "year": p.year, "quarter": p.quarter,
                            "period_complete": complete, "observed_days": (observed_end-a).total_seconds()/86400,
                            "observed_months": 3.0 if complete else (observed_end-a).total_seconds()/86400/30.4375,
                            "split": "train" if a < train_end else ("test" if complete else "test_partial")})
    panel = all_grids.merge(pd.DataFrame(period_rows), how="cross")
    panel = panel.loc[~panel["split"].eq("train") | panel["grid_id"].isin(historic_ids)].copy()
    panel = panel.merge(aggregate, on=["grid_id", "period"], how="left", validate="one_to_one")
    for col in ["earthquake_count", "depth_count", "shallow_count"]:
        panel[col] = panel[col].fillna(0).astype(int)
    panel["frequency_per_month"] = panel["earthquake_count"] / panel["observed_months"]
    panel["log_frequency"] = np.log1p(panel["frequency_per_month"])
    panel["shallow_fraction"] = panel["shallow_count"] / panel["depth_count"].replace(0, np.nan)
    panel["depth_coverage"] = panel["depth_count"] / panel["earthquake_count"].replace(0, np.nan)
    panel["low_support"] = panel["earthquake_count"].between(1, cfg["low_support_count"]-1)
    panel["new_grid_2026"] = ~panel["grid_id"].isin(historic_ids)
    eligible = panel[cfg["features"]].notna().all(axis=1) & panel["depth_coverage"].ge(0.8)
    panel["state"] = np.select([panel["earthquake_count"].eq(0), eligible], ["NO_EVENTS", "ELIGIBLE"], default="INCOMPLETE_FEATURES")
    lon, lat = inverse.transform((panel["grid_x"]+.5)*size, (panel["grid_y"]+.5)*size)
    panel["center_longitude"], panel["center_latitude"] = lon, lat
    panel["grid_km"] = grid_km
    panel["row_id"] = panel["grid_id"] + "|" + panel["period"]
    panel = panel.sort_values(["period", "grid_id"]).reset_index(drop=True)
    if panel["row_id"].duplicated().any():
        raise ValueError("row_id grid-triwulan duplikat.")
    return panel, data


def validate_collection(root, cfg):
    manifest = json.loads(require(root / "data/raw/collection_manifest.json", "01 collection").read_text())
    if not manifest.get("all_complete") or manifest["raw_sha256"] != file_hash(root / "data/raw/events.csv"):
        raise ValueError("Manifest collection tidak lengkap atau data mentah telah berubah.")
    cursor = pd.Timestamp(cfg["train_start"], tz="UTC")
    for interval in manifest["intervals"]:
        if pd.Timestamp(interval["start"]) != cursor or not interval["complete"]:
            raise ValueError("Ada celah unduhan. Nol kejadian tidak boleh diasumsikan.")
        cursor = pd.Timestamp(interval["end_exclusive"])
    if cursor != pd.Timestamp(cfg["test_end_exclusive"], tz="UTC"):
        raise ValueError("Cakupan unduhan berbeda dari konfigurasi.")
    return manifest


def preprocess(root=ROOT):
    root = Path(root)
    cfg = read_config(root)
    validate_collection(root, cfg)
    raw = pd.read_csv(root / "data/raw/events.csv")
    clean, rejected = clean_events(raw, cfg)
    panel, included = grid_panel(clean, cfg)
    training = panel.loc[panel["split"].eq("train") & panel["state"].eq("ELIGIBLE")]
    if len(training) < 10:
        raise ValueError("Kurang dari 10 observasi training yang layak.")
    scaler = StandardScaler().fit(training[cfg["features"]])
    eligible = panel.loc[panel["state"].eq("ELIGIBLE")]
    scaled = pd.DataFrame(scaler.transform(eligible[cfg["features"]]), columns=cfg["features"])
    scaled.insert(0, "row_id", eligible["row_id"].to_numpy())
    output = root / "data/processed"
    clean.to_csv(output / "events_clean.csv", index=False)
    rejected.to_csv(output / "events_rejected.csv", index=False)
    included.to_csv(output / "events_with_grid.csv", index=False)
    panel.to_csv(output / "panel.csv", index=False)
    scaled.to_csv(output / "features_scaled.csv", index=False)
    joblib.dump({"scaler": scaler, "features": cfg["features"], "fit_split": "train", "fit_rows": len(training)}, output / "transform.joblib")
    meta = {"config_sha256": signature(cfg), "raw_sha256": file_hash(root / "data/raw/events.csv"),
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "n_raw": len(raw), "n_clean": len(clean),
            "n_rejected": len(rejected), "n_invalid_depth": int(clean["depth_invalid"].sum()),
            "n_train_eligible": len(training), "n_train_periods": int(panel.loc[panel["split"].eq("train"), "period"].nunique()),
            "min_magnitude_provisional": cfg["min_magnitude"], "depth_coverage_min": 0.8,
            "zero_event_policy": "NO_EVENTS terpisah, statistik magnitudo/kedalaman tetap NaN",
            "grid_universe": "train: grid pernah aktif 2021-2025; test: gabungan historis dan grid baru 2026"}
    write_json(output / "metadata.json", meta)
    return panel, clean, meta
