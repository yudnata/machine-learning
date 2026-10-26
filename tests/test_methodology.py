from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import DBSCAN

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.analysis import extend_dbscan, temporal_tables
from src.collection import parse_feature
from src.common import file_hash, read_config, write_json
from src.preprocessing import clean_events, grid_panel, preprocess, validate_collection


@pytest.fixture
def cfg():
    return copy.deepcopy(read_config())


def synthetic_events():
    rows=[]
    for p in pd.period_range("2021Q1","2026Q4",freq="Q"):
        for location,(lat,lon) in enumerate([(-5,110),(-2,120),(2,130)]):
            if p==pd.Period("2022Q2") and location==0:
                continue
            rows.append({"eventid":f"{p}_{location}","time":(p.start_time+pd.Timedelta(days=2)).isoformat(),
                         "lastupdate":"2026-10-25T00:00:00Z","latitude":lat,"longitude":lon,
                         "depth":20+location*40,"geometry_z":-20-location*40,
                         "magnitude":3.2+location*.4,"region":"fixture","magnitude_type":"m",
                         "depth_source":"properties.depth","source_catalog":"fixture","author":"fixture","event_type":"ke"})
    return pd.DataFrame(rows)


def prepare_root(root, cfg, raw):
    root.mkdir(exist_ok=True)
    write_json(root/"config.json",cfg)
    read_config(root)
    raw.to_csv(root/"data/raw/events.csv",index=False)
    start=pd.Timestamp(cfg["train_start"],tz="UTC").isoformat()
    end=pd.Timestamp(cfg["test_end_exclusive"],tz="UTC").isoformat()
    write_json(root/"data/raw/collection_manifest.json",{
        "all_complete":True,"raw_sha256":file_hash(root/"data/raw/events.csv"),
        "intervals":[{"start":start,"end_exclusive":end,"complete":True}]})


def test_depth_uses_property_not_negative_geometry():
    feature={"id":"e","properties":{"depth":25,"mag":4},"geometry":{"coordinates":[118,-2,-25]}}
    parsed=parse_feature(feature)
    assert parsed["depth"]==25
    assert parsed["geometry_z"]==-25
    del feature["properties"]["depth"]
    assert parse_feature(feature)["depth"] is None


def test_cleaning_mixed_timestamps_missing_id_and_revision(cfg):
    raw=synthetic_events().iloc[:4].copy()
    raw.loc[0,"time"]="2021-01-03T00:00:00Z"
    raw.loc[1,"time"]="2021-01-03T01:00:00.500Z"
    raw.loc[0:1,"eventid"]=None
    duplicate=raw.iloc[[2]].copy()
    duplicate["lastupdate"]="2026-10-25T01:00:00.900Z"
    duplicate["depth"]=55
    raw=pd.concat([raw,duplicate],ignore_index=True)
    clean,rejected=clean_events(raw,cfg)
    assert len(clean)==4
    assert len(rejected)==1
    assert clean["eventid"].isna().sum()==2
    assert clean.loc[clean["eventid"].eq(duplicate.iloc[0]["eventid"]),"depth"].item()==55


def test_zero_events_and_invalid_depth_remain_distinct(cfg):
    raw=synthetic_events()
    raw.loc[0,"depth"]=-20
    clean,_=clean_events(raw,cfg)
    panel,_=grid_panel(clean,cfg)
    historic=panel.loc[panel["split"].eq("train")]
    assert historic.groupby("grid_id")["period"].nunique().eq(20).all()
    empty=historic.loc[historic["state"].eq("NO_EVENTS")]
    assert len(empty)==1
    assert empty["earthquake_count"].eq(0).all()
    assert empty[["avg_magnitude","max_magnitude","avg_depth"]].isna().all().all()
    incomplete=historic.loc[historic["state"].eq("INCOMPLETE_FEATURES")]
    assert len(incomplete)==1
    assert incomplete["earthquake_count"].eq(1).all()
    assert panel.loc[panel["split"].eq("test_partial"),"period"].eq("2026Q4").all()


def test_test_data_cannot_change_training_grid_or_scaler(tmp_path,cfg):
    raw=synthetic_events()
    prepare_root(tmp_path,cfg,raw)
    first,_,_=preprocess(tmp_path)
    scaler1=joblib.load(tmp_path/"data/processed/transform.joblib")["scaler"]
    test=raw["time"].str.startswith("2026")
    raw.loc[test,"magnitude"]=9
    raw.loc[test,"depth"]=700
    extra=raw.loc[test].iloc[[0]].copy()
    extra["eventid"]="new2026"
    extra["longitude"]=100
    raw=pd.concat([raw,extra],ignore_index=True)
    prepare_root(tmp_path,cfg,raw)
    second,_,_=preprocess(tmp_path)
    scaler2=joblib.load(tmp_path/"data/processed/transform.joblib")["scaler"]
    np.testing.assert_allclose(scaler1.mean_,scaler2.mean_)
    np.testing.assert_allclose(scaler1.scale_,scaler2.scale_)
    pd.testing.assert_frame_equal(first.loc[first["split"].eq("train")].reset_index(drop=True),second.loc[second["split"].eq("train")].reset_index(drop=True))
    assert second.loc[second["new_grid_2026"],"split"].ne("train").all()


def test_collection_gap_is_not_zero_events(tmp_path,cfg):
    prepare_root(tmp_path,cfg,synthetic_events())
    path=tmp_path/"data/raw/collection_manifest.json"
    manifest=json.loads(path.read_text())
    manifest["intervals"][0]["start"]="2021-02-01T00:00:00+00:00"
    write_json(path,manifest)
    with pytest.raises(ValueError,match="celah"):
        validate_collection(tmp_path,cfg)


def test_temporal_does_not_bridge_empty_noise_or_missing_quarter():
    labels=pd.DataFrame({"grid_id":["A"]*7,"period":["2021Q1","2021Q2","2021Q3","2021Q4","2022Q1","2022Q2","2022Q4"],
                         "profile":["KMEANS_P0","NO_EVENTS","KMEANS_P1","KMEANS_P1","KMEANS_P0","NOISE","KMEANS_P1"]})
    _,summary,transitions,_,spells=temporal_tables(labels)
    assert summary.iloc[0]["n_comparable_pairs"]==2
    assert summary.iloc[0]["n_profile_changes"]==1
    assert summary.iloc[0]["stability_rate"]==.5
    assert spells["duration_quarters"].max()==2
    assert transitions.to_numpy().sum()==2


def test_dbscan_extension_rejects_ambiguous_and_outside_radius():
    model=DBSCAN(eps=1,min_samples=2).fit(np.array([[-.7],[-.6],[.6],[.7]]))
    assert len(np.unique(model.labels_))==2
    labels,matches,distance=extend_dbscan(model,np.array([[0.],[10.],[-1.4]]))
    assert labels.tolist()==[-2,-1,0]
    assert matches.tolist()==[2,0,1]
    assert np.isfinite(distance).all()
