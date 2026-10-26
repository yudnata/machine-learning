"""Siapkan garis daratan dan Leaflet lokal; tidak mengunduh tile peta."""
import base64
import hashlib
import json
from pathlib import Path

import requests
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/maps"


def prepare():
    ASSETS.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "SeismicResearchMapAssets/1.0 (local research notebook)"
    sources = []

    def fetch(url):
        response = session.get(url, timeout=(15, 120))
        response.raise_for_status()
        sources.append({"url": url, "sha256": hashlib.sha256(response.content).hexdigest()})
        return response.content

    base = "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/"
    js = fetch(base + "leaflet.js").decode("utf-8")
    css = fetch(base + "leaflet.css").decode("utf-8")
    for name in ["layers.png", "layers-2x.png", "marker-icon.png"]:
        encoded = base64.b64encode(fetch(base + "images/" + name)).decode("ascii")
        css = css.replace("images/" + name, "data:image/png;base64," + encoded)
    # Source map is a development aid and would otherwise create an external request.
    js = "\n".join(line for line in js.splitlines() if "sourceMappingURL=" not in line)
    (ASSETS / "leaflet.js").write_text(js, encoding="utf-8")
    (ASSETS / "leaflet.css").write_text(css, encoding="utf-8")
    (ASSETS / "LEAFLET_LICENSE.txt").write_bytes(fetch("https://cdn.jsdelivr.net/npm/leaflet@1.9.4/LICENSE"))
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"
    collection = json.loads(fetch(url))
    region = box(85, -20, 155, 15)
    pieces = []
    for feature in collection["features"]:
        geometry = shape(feature["geometry"])
        if geometry.intersects(region):
            clipped = geometry.intersection(region).simplify(.008, preserve_topology=True)
            if not clipped.is_empty:
                pieces.append(clipped)
    land = unary_union(pieces)
    result = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"source": "Natural Earth 1:50m land"}, "geometry": mapping(land)}]}
    (ASSETS / "land_southeast_asia.geojson").write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    (ASSETS / "sources.json").write_text(json.dumps({"sources": sources, "land_license": "public domain",
        "land_terms": "https://www.naturalearthdata.com/about/terms-of-use/", "leaflet_version": "1.9.4",
        "clip_bounds_wgs84": [85, -20, 155, 15], "simplify_tolerance_degrees": .008,
        "purpose": "context daratan; tidak dipakai untuk clipping atau fitur clustering"}, indent=2), encoding="utf-8")
    print("Aset peta lokal siap:", ASSETS)


if __name__ == "__main__":
    prepare()
