import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import requests

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.collection import collect
from src.common import read_config, write_json


def configure(root):
    config=read_config()
    config.update(train_start="2021-01-01",test_end_exclusive="2021-02-01",api_limit=2,request_pause_seconds=0)
    write_json(root/"config.json",config)


def test_saturated_response_splits_intervals_and_cache_avoids_network(tmp_path,monkeypatch):
    configure(tmp_path)
    calls=[]

    class Response:
        status_code=200
        url="https://example.test/recorded-query"
        def __init__(self,features): self.features=features
        def raise_for_status(self): return None
        def json(self): return {"type":"FeatureCollection","features":self.features}

    class Session:
        def mount(self,*args,**kwargs): pass
        def get(self,url,params,timeout):
            calls.append(params)
            start=pd.Timestamp(params["starttime"],tz="UTC")
            end=pd.Timestamp(params["endtime"],tz="UTC")
            if end-start>pd.Timedelta(days=20):
                features=[{"id":"truncated"}]*2
            else:
                midpoint=start+(end-start)/2
                features=[{"id":start.isoformat(),"geometry":{"coordinates":[118,-2,-25]},
                           "properties":{"time":midpoint.isoformat(),"depth":25,"mag":4,"evtype":"ke"}}]
            return Response(features)

    monkeypatch.setattr("src.collection.requests.Session",Session)
    events,coverage=collect(tmp_path)
    assert len(calls)==3
    assert len(events)==2
    assert events["eventid"].ne("truncated").all()
    assert coverage["complete"].all()
    assert "T" in calls[0]["endtime"]
    before=len(calls)
    cached,_=collect(tmp_path)
    assert len(calls)==before
    assert len(cached)==2


def test_http_failure_never_writes_success_manifest(tmp_path,monkeypatch):
    configure(tmp_path)

    class Session:
        def mount(self,*args,**kwargs): pass
        def get(self,*args,**kwargs): raise requests.HTTPError("upstream failed")

    monkeypatch.setattr("src.collection.requests.Session",Session)
    with pytest.raises(requests.HTTPError):
        collect(tmp_path)
    assert not (tmp_path/"data/raw/collection_manifest.json").exists()
