import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, read_config
from src.mapping import create_base_map


def test_basemap_is_embedded_and_has_no_tile_or_cdn_requests():
    rendered = create_base_map(ROOT, read_config()).get_root().render()
    assert not re.search(r'<script\b[^>]*\bsrc\s*=', rendered, re.I)
    assert not re.search(r'<link\b[^>]*\brel=["\']stylesheet', rendered, re.I)
    assert 'tile.openstreetmap.org' not in rendered
    assert 'L.tileLayer(' not in rendered
    assert 'sourceMappingURL=' not in rendered
    assert 'data:image/png;base64,' in rendered
    assert 'land_context' in rendered
    assert '.fitBounds(' in rendered
    assert '"scrollWheelZoom": false' in rendered


def test_land_context_is_valid_and_covers_indonesia():
    from shapely.geometry import shape
    geojson = json.loads((ROOT / 'assets/maps/land_southeast_asia.geojson').read_text())
    land = shape(geojson['features'][0]['geometry'])
    assert land.is_valid
    assert not land.is_empty
    from shapely.geometry import Point
    assert land.covers(Point(110.4, -7.8))
    assert not land.covers(Point(110, -15))
