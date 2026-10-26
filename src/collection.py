from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .common import ROOT, file_hash, read_config, write_json


def parse_feature(feature):
    properties = feature.get("properties") or {}
    coordinates = (feature.get("geometry") or {}).get("coordinates") or []
    return {
        "eventid": feature.get("id") or properties.get("unid"),
        "time": properties.get("time"),
        "longitude": coordinates[0] if len(coordinates) > 0 else None,
        "latitude": coordinates[1] if len(coordinates) > 1 else None,
        "depth": properties.get("depth"),
        "depth_source": "properties.depth",
        "geometry_z": coordinates[2] if len(coordinates) > 2 else None,
        "magnitude": properties.get("mag"),
        "magnitude_type": properties.get("magtype"),
        "event_type": properties.get("evtype"),
        "region": properties.get("flynn_region") or properties.get("region"),
        "lastupdate": properties.get("lastupdate"),
        "source_catalog": properties.get("source_catalog"),
        "author": properties.get("auth"),
    }


def collect(root=ROOT):
    root = Path(root)
    cfg = read_config(root)
    start = pd.Timestamp(cfg["train_start"], tz="UTC")
    end = pd.Timestamp(cfg["test_end_exclusive"], tz="UTC")
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"], respect_retry_after_header=True)))
    box = cfg["bbox"]
    base = {"format": "json", "minlatitude": box["min_lat"], "maxlatitude": box["max_lat"],
            "minlongitude": box["min_lon"], "maxlongitude": box["max_lon"],
            "limit": cfg["api_limit"], "orderby": "time-asc"}

    def fetch_interval(a, b):
        params = {**base, "starttime": a.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                  "endtime": (b - pd.Timedelta(microseconds=1)).strftime("%Y-%m-%dT%H:%M:%S.%f")}
        response = session.get(cfg["api_url"], params=params, timeout=(15, 120))
        time.sleep(cfg["request_pause_seconds"])
        if response.status_code == 204:
            return [], [{"url": response.url, "status": 204, "count": 0}]
        response.raise_for_status()
        payload = response.json()
        if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
            raise ValueError("Respons EMSC bukan FeatureCollection; unduhan tidak dinyatakan lengkap.")
        features = payload["features"]
        receipt = {"url": response.url, "status": response.status_code, "count": len(features)}
        if len(features) >= cfg["api_limit"]:
            if b - a <= pd.Timedelta(seconds=1):
                raise RuntimeError("Batas API tercapai bahkan pada interval 1 detik.")
            midpoint = a + (b - a) / 2
            left, lr = fetch_interval(a, midpoint)
            right, rr = fetch_interval(midpoint, b)
            return left + right, [receipt] + lr + rr
        return features, [receipt]

    rows, manifests = [], []
    for a in pd.date_range(start, end, freq="MS", inclusive="left"):
        b = min(a + pd.offsets.MonthBegin(1), end)
        cache = root / f"data/raw/monthly/{a:%Y-%m}.json"
        key = {"start": a.isoformat(), "end_exclusive": b.isoformat(), "base": base, "api": cfg["api_url"]}
        saved = json.loads(cache.read_text()) if cache.exists() else None
        if not saved or saved.get("request") != key or not saved.get("complete"):
            features, receipts = fetch_interval(a, b)
            saved = {"request": key, "complete": True, "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                     "requests": receipts, "features": features}
            write_json(cache, saved)
        records = [parse_feature(f) for f in saved["features"]]
        rows.extend(records)
        manifests.append({"start": a.isoformat(), "end_exclusive": b.isoformat(), "complete": True,
                          "events_returned": len(records), "cache_sha256": file_hash(cache),
                          "retrieved_at_utc": saved["retrieved_at_utc"]})
        print(f"{a:%Y-%m}: {len(records):,} event (cache tersedia)", flush=True)
    if not rows:
        raise ValueError("Katalog kosong. Periksa permintaan; jangan lanjutkan model.")
    events = pd.DataFrame(rows)
    events.to_csv(root / "data/raw/events.csv", index=False)
    write_json(root / "data/raw/collection_manifest.json", {"source": cfg["api_url"], "intervals": manifests,
               "start": start.isoformat(), "end_exclusive": end.isoformat(), "all_complete": True,
               "license": "EMSC / SeismicPortal, CC BY 4.0", "raw_sha256": file_hash(root / "data/raw/events.csv")})
    return events, pd.DataFrame(manifests)
