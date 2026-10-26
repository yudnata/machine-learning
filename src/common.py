from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_config(root=ROOT):
    root = Path(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    for name in ["data/raw/monthly", "data/processed", "models/kmeans", "models/dbscan",
                 "reports/figures", "reports/evaluation", "reports/temporal",
                 "reports/stability", "reports/testing_2026", "maps", "reports/validation"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def require(path, stage):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Belum ada {path.name}. Jalankan notebook {stage} dahulu.")
    return path


def load_prepared(root=ROOT):
    import joblib
    root = Path(root)
    config = read_config(root)
    meta = json.loads(require(root / "data/processed/metadata.json", "02 preprocessing").read_text())
    if meta["config_sha256"] != signature(config):
        raise ValueError("Konfigurasi berubah. Jalankan ulang 02 preprocessing dan seluruh tahap setelahnya.")
    raw = root / "data/raw/events.csv"
    if meta["raw_sha256"] != file_hash(raw):
        raise ValueError("Data mentah berubah. Jalankan ulang 02 preprocessing.")
    panel = pd.read_csv(root / "data/processed/panel.csv", keep_default_na=False, na_values=[""])
    scaled = pd.read_csv(root / "data/processed/features_scaled.csv")
    bundle = joblib.load(root / "data/processed/transform.joblib")
    train = panel.loc[panel["split"].eq("train") & panel["state"].eq("ELIGIBLE")].copy()
    train = train.sort_values("row_id").reset_index(drop=True)
    X = scaled.set_index("row_id").loc[train["row_id"], config["features"]].to_numpy()
    if not np.isfinite(X).all():
        raise ValueError("Fitur training tidak finite.")
    return config, panel, train, X, bundle, meta


def save_model_metadata(root, method, meta, extra):
    write_json(Path(root) / f"models/{method}/metadata.json",
               {"prepared_sha256": file_hash(Path(root) / "data/processed/metadata.json"), **extra})


def load_model(root, method):
    import joblib
    root = Path(root)
    meta = json.loads(require(root / f"models/{method}/metadata.json", f"model {method}").read_text())
    if meta["prepared_sha256"] != file_hash(root / "data/processed/metadata.json"):
        raise ValueError(f"Model {method} memakai preprocessing lama. Jalankan ulang notebook model.")
    return joblib.load(root / f"models/{method}/model.joblib"), meta
