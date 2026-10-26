from __future__ import annotations

import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
packages = {
    "numpy":"numpy", "pandas":"pandas", "scipy":"scipy", "scikit-learn":"sklearn",
    "matplotlib":"matplotlib", "seaborn":"seaborn", "requests":"requests", "pyproj":"pyproj",
    "folium":"folium", "branca":"branca", "joblib":"joblib", "threadpoolctl":"threadpoolctl",
    "ipykernel":"ipykernel", "ipython":"IPython", "nbformat":"nbformat", "nbclient":"nbclient",
    "jupyterlab":"jupyterlab", "pytest":"pytest",
    "plotly":"plotly", "geopandas":"geopandas", "shapely":"shapely",
}


def verify():
    versions={}
    for package,module in packages.items():
        importlib.import_module(module)
        versions[package]=importlib.metadata.version(package)
    from pyproj import Transformer
    x,y=Transformer.from_crs("EPSG:4326","EPSG:6933",always_xy=True).transform(118,-2.5)
    assert abs(x)>0 and abs(y)>0
    report={"executable":sys.executable,"python_version":sys.version,"imports":"PASS","projection":"PASS","packages":versions}
    target=ROOT / "reports/validation"
    target.mkdir(parents=True,exist_ok=True)
    (target / "environment.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (ROOT / "requirements-lock.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True),encoding="utf-8")
    print(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    verify()
