from __future__ import annotations

import json
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from branca.element import MacroElement, Template
from pyproj import Transformer

from .common import ROOT, load_model, load_prepared, require

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#882255"]
SPECIAL = {"NO_EVENTS": "#ededed", "INCOMPLETE_FEATURES": "#999999", "NOISE": "#333333",
           "NO_CORE_MATCH": "#333333", "AMBIGUOUS_CORE_MATCH": "#b58900"}


def create_base_map(root, cfg):
    assets = Path(root) / "assets/maps"
    for name in ["leaflet.js", "leaflet.css", "land_southeast_asia.geojson"]:
        if not (assets / name).is_file():
            raise FileNotFoundError(f"Aset peta hilang: {name}. Jalankan scripts/prepare_map_assets.py.")
    map_obj = folium.Map(location=[-2.5, 118], zoom_start=5, tiles=None,
                         min_zoom=4, max_zoom=12, control_scale=True,
                         scroll_wheel_zoom=False, prefer_canvas=True)
    map_obj.default_js = []
    map_obj.default_css = []
    header = map_obj.get_root().header
    header.add_child(folium.Element("<style>" + (assets / "leaflet.css").read_text(encoding="utf-8") + "</style>"), name="local_leaflet_css")
    header.add_child(folium.Element("<script>" + (assets / "leaflet.js").read_text(encoding="utf-8") + "</script>"), name="local_leaflet_js")
    header.add_child(folium.Element("<style>.leaflet-container{background:#e5f0f5;font-family:Arial,sans-serif}</style>"), name="map_context_style")
    folium.map.CustomPane("land_context", z_index=200, pointer_events=False).add_to(map_obj)
    land = json.loads((assets / "land_southeast_asia.geojson").read_text(encoding="utf-8"))
    folium.GeoJson(land, name="Daratan Natural Earth", control=False, pane="land_context", interactive=False,
                   style_function=lambda feature: {"color":"#687e86", "weight":.8, "fillColor":"#fbfaf5", "fillOpacity":1}).add_to(map_obj)
    box = cfg["bbox"]
    map_obj.fit_bounds([[box["min_lat"], box["min_lon"]], [box["max_lat"], box["max_lon"]]], padding=(30, 30))
    attribution = MacroElement()
    attribution._template = Template('''{% macro script(this, kwargs) %}
    {{this.map_name}}.attributionControl.addAttribution('Daratan: <a href="https://www.naturalearthdata.com/">Natural Earth</a> (public domain); katalog: EMSC');
    {% endmacro %}''')
    attribution.map_name = map_obj.get_name()
    map_obj.add_child(attribution)
    return map_obj


def polygon_coordinates(grid_x, grid_y, grid_km, inverse):
    size = grid_km*1000
    x, y = grid_x*size, grid_y*size
    corners = [(x,y),(x+size,y),(x+size,y+size),(x,y+size),(x,y)]
    # Sisi proyeksi didensifikasi sebelum dibawa kembali ke WGS84.
    points = []
    for a, b in zip(corners, corners[1:]):
        for fraction in np.linspace(0, 1, 5, endpoint=False):
            lon, lat = inverse.transform(a[0]+fraction*(b[0]-a[0]), a[1]+fraction*(b[1]-a[1]))
            points.append([round(lon, 5), round(lat, 5)])
    points.append(points[0])
    return [points]


def profile_color(profile):
    if profile in SPECIAL:
        return SPECIAL[profile]
    return COLORS[int(profile.rsplit("_P",1)[1]) % len(COLORS)]


def readable_period(period):
    year, quarter = str(period).split("Q")
    months = {"1": "Januari-Maret", "2": "April-Juni", "3": "Juli-September", "4": "Oktober-Desember"}
    return f"{months[quarter]} {year}" + (" (belum lengkap)" if period == "2026Q4" else "")


def profile_descriptions(root, method, cfg):
    profiles = pd.read_csv(Path(root) / f"models/{method}/profiles.csv")
    clusters = profiles.loc[profiles["profile"].str.contains("_P", regex=False)].copy()
    maximum_count = clusters["earthquake_count"].max()
    descriptions = {}
    for row in clusters.itertuples():
        number = int(row.profile.rsplit("_P", 1)[1])
        group = f"Kelompok {chr(65 + number)}"
        magnitude = f"{row.avg_magnitude:.1f}".replace(".", ",")
        count = f"{row.earthquake_count:.1f}".replace(".", ",")
        depth = f"{row.avg_depth:.0f}"
        if row.avg_depth < cfg["shallow_depth_km"]:
            characteristic = f"Cenderung dangkal, magnitudo sekitar {magnitude}"
        else:
            characteristic = f"Kedalaman sekitar {depth} km, magnitudo {magnitude}"
        if row.earthquake_count == maximum_count:
            characteristic = "Kejadian lebih sering; " + ("cenderung dangkal" if row.avg_depth < cfg["shallow_depth_km"] else f"kedalaman sekitar {depth} km")
        descriptions[row.profile] = {
            "name": group + ": " + characteristic,
            "detail": f"Rata-rata {count} kejadian / 3 bulan; magnitudo {magnitude}; kedalaman {depth} km.",
            "source": "Ringkasan statistik cluster historis 2021-2025, bukan kategori yang menentukan anggota cluster.",
            "color": profile_color(row.profile),
        }
    special_names = {
        "NO_EVENTS": f"Tidak ada gempa tercatat dengan magnitudo minimal {cfg['min_magnitude']:g}",
        "INCOMPLETE_FEATURES": "Data belum cukup untuk menentukan kelompok",
        "NOISE": "Tidak masuk kelompok historis DBSCAN",
        "NO_CORE_MATCH": "Belum cocok dengan kelompok historis",
        "AMBIGUOUS_CORE_MATCH": "Cocok dengan lebih dari satu kelompok",
    }
    for code, name in special_names.items():
        descriptions[code] = {"name": name, "detail": "Status data atau pencocokan model; bukan tingkat bahaya.",
                              "source": "Status pipeline", "color": profile_color(code)}
    return descriptions


def profile_interpretation_table(root, method, cfg):
    profiles = pd.read_csv(Path(root) / f"models/{method}/profiles.csv")
    descriptions = profile_descriptions(root, method, cfg)
    profiles["nama_kelompok"] = profiles["profile"].map(lambda code: descriptions[code]["name"])
    table = profiles[["profile", "nama_kelompok", "earthquake_count", "avg_magnitude", "avg_depth", "n_grid_quarters"]].copy()
    return table.rename(columns={
        "profile": "Kode algoritma", "nama_kelompok": "Nama untuk menjelaskan hasil",
        "earthquake_count": "Gempa per kotak per triwulan (rata-rata)",
        "avg_magnitude": "Magnitudo (rata-rata)", "avg_depth": "Kedalaman km (rata-rata)",
        "n_grid_quarters": "Jumlah kotak-periode pembentuk kelompok",
    }).round(2)


def attach_grid_toggle(map_obj, layers, default_key):
    control = MacroElement()
    control._template = Template('''
    {% macro script(this, kwargs) %}
    const gridMap = {{this.map_name}};
    const gridLayers = { {% for key, layer in this.layers %}{{key|tojson}}: {{layer}}{% if not loop.last %},{% endif %}{% endfor %} };
    gridMap._seismicGridState = {visible:true, currentLayer:gridLayers[{{this.default_key|tojson}}]};
    const gridToggleControl = L.control({position:'bottomleft'});
    gridToggleControl.onAdd = function() {
      const container = L.DomUtil.create('div');
      const button = document.createElement('button');
      button.type='button';button.textContent='Sembunyikan grid';
      button.setAttribute('aria-label','Tampilkan atau sembunyikan kotak grid');
      button.setAttribute('aria-pressed','true');
      button.style.cssText='padding:8px 12px;min-height:40px;background:white;color:#222;border:1px solid #777;cursor:pointer;font:14px Arial,sans-serif;';
      button.onclick=function() {
        const state=gridMap._seismicGridState;state.visible=!state.visible;
        Object.values(gridLayers).forEach(function(layer){if(gridMap.hasLayer(layer)){gridMap.removeLayer(layer);}});
        if(state.visible){gridMap.addLayer(state.currentLayer);}else{gridMap.closePopup();}
        button.textContent=state.visible?'Sembunyikan grid':'Tampilkan grid';
        button.setAttribute('aria-pressed',String(state.visible));
      };
      container.appendChild(button);
      L.DomEvent.disableClickPropagation(container);L.DomEvent.disableScrollPropagation(container);
      return container;
    };gridToggleControl.addTo(gridMap);
    {% endmacro %}''')
    control.map_name = map_obj.get_name()
    control.layers = [(key, layer.get_name()) for key, layer in layers.items()]
    control.default_key = default_key
    map_obj.add_child(control)


def attach_selector(map_obj, layers, default_period, title, legend, grid_km=50):
    control = MacroElement()
    control._template = Template('''
    {% macro script(this, kwargs) %}
    const mapRef = {{this.map_name}};
    const periodLayers = { {% for period, layer in this.layers %}{{period|tojson}}: {{layer}}{% if not loop.last %},{% endif %}{% endfor %} };
    const selector = L.control({position:'topright'});
    selector.onAdd = function() {
      const container = L.DomUtil.create('div');
      container.style.cssText='background:white;color:#222;padding:14px;width:min(350px,calc(100vw - 85px));box-sizing:border-box;max-height:85vh;overflow:auto;border:1px solid #777;line-height:1.45;font:14px Arial,sans-serif;';
      const heading = document.createElement('strong'); heading.textContent={{this.title|tojson}};container.appendChild(heading);
      const intro=document.createElement('p');intro.textContent='Satu kotak sekitar {{this.grid_km}} x {{this.grid_km}} km. Warna menunjukkan kelompok dengan ciri gempa yang serupa.';container.appendChild(intro);
      const origin=document.createElement('p');origin.textContent='Algoritma membentuk kelompok. Nama di bawah adalah penjelasan kita berdasarkan rata-rata data 2021-2025.';container.appendChild(origin);
      const label=document.createElement('label');label.textContent='Pilih periode ';label.style.display='block';
      const select=document.createElement('select');select.setAttribute('aria-label','Tahun dan triwulan');select.style.cssText='padding:7px;min-width:190px;margin:8px 0';
      const periodLabels={{this.period_labels|tojson}};
      Object.keys(periodLayers).forEach(function(p){const opt=document.createElement('option');opt.value=p;opt.textContent=periodLabels[p];opt.selected=p==={{this.default_period|tojson}};select.appendChild(opt);});
      select.onchange=function(){Object.values(periodLayers).forEach(function(layer){if(mapRef.hasLayer(layer)){mapRef.removeLayer(layer);}});const state=mapRef._seismicGridState;if(state){state.currentLayer=periodLayers[select.value];if(state.visible){mapRef.addLayer(state.currentLayer);}}else{mapRef.addLayer(periodLayers[select.value]);}};
      label.appendChild(select);container.appendChild(label);
      const hint=document.createElement('p');hint.textContent='Klik kotak untuk melihat data periode itu. Angka pada legenda adalah rata-rata historis kelompok; nilai setiap kotak dapat berbeda.';container.appendChild(hint);
      {% for item in this.legend %}
      {const row=document.createElement('div');row.style.cssText='margin:10px 0';const swatch=document.createElement('span');swatch.style.cssText='display:inline-block;width:14px;height:14px;margin-right:6px;background:{{item.color}};border:1px solid #555';row.appendChild(swatch);const name=document.createElement('strong');name.textContent={{item.name|tojson}};row.appendChild(name);const detail=document.createElement('div');detail.style.cssText='margin-left:22px;font-size:12px;color:#444';detail.textContent={{item.detail|tojson}};row.appendChild(detail);container.appendChild(row);}
      {% endfor %}
      L.DomEvent.disableClickPropagation(container);L.DomEvent.disableScrollPropagation(container);return container;
    };selector.addTo(mapRef);
    {% endmacro %}''')
    control.map_name = map_obj.get_name()
    control.layers = [(p, layer.get_name()) for p,layer in layers.items()]
    control.default_period = default_period
    control.period_labels = {p: readable_period(p) for p in layers}
    control.title = title
    control.legend = legend
    control.grid_km = grid_km
    map_obj.add_child(control)


def create_maps(root=ROOT):
    root = Path(root)
    cfg, _, _, _, _, _ = load_prepared(root)
    require(root / "reports/testing_2026/labels_all_2026.csv", "09 testing 2026")
    test = pd.read_csv(root / "reports/testing_2026/labels_all_2026.csv")
    inverse = Transformer.from_crs(cfg["projection"], "EPSG:4326", always_xy=True)
    paths = []
    for method in ["kmeans", "dbscan"]:
        load_model(root, method)
        descriptions = profile_descriptions(root, method, cfg)
        explanation_path = root / "reports/evaluation" / f"{method}_profile_explanations.json"
        explanation_path.write_text(json.dumps(descriptions, indent=2, ensure_ascii=False), encoding="utf-8")
        labels = pd.read_csv(root / f"models/{method}/training_labels.csv")
        test_labels = test.copy()
        test_labels["profile"] = test_labels[f"{method}_profile"]
        combined = pd.concat([labels, test_labels], ignore_index=True)
        grids = combined[["grid_id", "grid_x", "grid_y"]].drop_duplicates("grid_id")
        geometry = {r.grid_id: polygon_coordinates(r.grid_x, r.grid_y, cfg["grid_km"], inverse) for r in grids.itertuples()}
        map_obj = create_base_map(root, cfg)
        periods = sorted(combined["period"].unique())
        default = sorted(combined.loc[combined["split"].eq("test"), "period"].unique())[-1] if combined["split"].eq("test").any() else "2025Q4"
        layers = {}
        for period in periods:
            subset = combined.loc[combined["period"].eq(period)]
            features = []
            for row in subset.itertuples():
                explanation = descriptions[row.profile]
                out_of_range = getattr(row, "outside_training_distance_q99", False)
                assignment_note = "Penetapan kelompok berdasarkan ciri aktivitas."
                if method == "kmeans" and pd.notna(out_of_range) and bool(out_of_range):
                    assignment_note = "Ciri data jauh dari pola historis. K-Means tetap memberi kelompok terdekat; periksa datanya."
                properties = {"grid_id":row.grid_id, "period":readable_period(row.period), "split":"Pembentukan kelompok (2021-2025)" if row.split == "train" else "Pengujian data baru (2026); periode belum lengkap" if row.split == "test_partial" else "Pengujian data baru (2026)",
                              "profile":row.profile, "group_name":explanation["name"], "historical_summary":explanation["detail"],
                              "event_count":int(row.earthquake_count), "avg_magnitude":round(row.avg_magnitude,2) if pd.notna(row.avg_magnitude) else "Tidak tersedia",
                              "avg_depth_km":round(row.avg_depth,2) if pd.notna(row.avg_depth) else "Tidak tersedia",
                              "support_note":"Hanya 1-4 kejadian; ringkasan periode ini mudah berubah." if row.low_support else "Tidak ada kejadian di atas ambang magnitudo pada periode ini." if row.earthquake_count == 0 else f"Berdasarkan {int(row.earthquake_count)} kejadian tercatat.",
                              "assignment_note":assignment_note, "color":profile_color(row.profile)}
                features.append({"type":"Feature", "id":row.grid_id, "geometry":{"type":"Polygon","coordinates":geometry[row.grid_id]}, "properties":properties})
            group = folium.FeatureGroup(name=period, show=period==default, control=False)
            folium.GeoJson({"type":"FeatureCollection","features":features},
                style_function=lambda f:{"fillColor":f["properties"]["color"],"color":"#555555","weight":.35,"fillOpacity":.65},
                highlight_function=lambda f:{"weight":2,"fillOpacity":.85},
                tooltip=folium.GeoJsonTooltip(fields=["group_name","period","event_count"], aliases=["Kelompok aktivitas","Periode dipilih","Gempa tercatat pada kotak ini"]),
                popup=folium.GeoJsonPopup(fields=["grid_id","period","split","group_name","event_count","avg_magnitude","avg_depth_km","support_note","historical_summary","assignment_note","profile"],
                       aliases=["Kode kotak","Periode dipilih","Pemakaian data","Kelompok hasil algoritma","Jumlah gempa pada periode ini","Rata-rata magnitudo periode ini","Rata-rata kedalaman periode ini (km)","Dukungan data","Pembanding: rata-rata historis kelompok","Catatan penetapan kelompok","Kode untuk audit"],
                       max_width=420, style="font:13px Arial,sans-serif;white-space:normal;max-width:380px;")).add_to(group)
            group.add_to(map_obj)
            layers[period] = group
        legend = [descriptions[p] for p in sorted(combined["profile"].unique(), key=lambda p:(0 if "_P" in p else 1, p))]
        attach_grid_toggle(map_obj, layers, default)
        attach_selector(map_obj, layers, default, f"Kelompok aktivitas gempa · {'K-Means' if method == 'kmeans' else 'DBSCAN'}", legend, cfg["grid_km"])
        path = root / f"maps/{method}_profiles_by_quarter.html"
        map_obj.save(str(path))
        paths.append(path)
        require(root / f"reports/temporal/{method}/grid_stability.csv", "07 transisi")
        summary = pd.read_csv(root / f"reports/temporal/{method}/grid_stability.csv")
        summary = summary.merge(grids, on="grid_id", validate="one_to_one")
        changes_map = create_base_map(root, cfg)
        features = []
        for row in summary.itertuples():
            color = "#999999" if row.n_comparable_pairs == 0 else ("#0072B2" if row.n_profile_changes==0 else "#E69F00" if row.n_profile_changes<=2 else "#D55E00")
            props={"grid_id":row.grid_id,"n_profile_changes":row.n_profile_changes,"n_comparable_pairs":row.n_comparable_pairs,
                   "stability_rate":f"{row.stability_rate:.0%} dari pasangan yang bisa dibandingkan" if pd.notna(row.stability_rate) else "Belum dapat dihitung",
                   "max_profile_run_quarters":row.max_profile_run_quarters,"color":color}
            features.append({"type":"Feature","geometry":{"type":"Polygon","coordinates":geometry[row.grid_id]},"properties":props})
        changes_layer = folium.GeoJson({"type":"FeatureCollection","features":features},
            style_function=lambda f:{"fillColor":f["properties"]["color"],"color":"#555","weight":.4,"fillOpacity":.7},
            tooltip=folium.GeoJsonTooltip(fields=["grid_id","n_profile_changes","n_comparable_pairs","stability_rate","max_profile_run_quarters"],
                aliases=["Kode kotak","Berapa kali berganti kelompok","Pasangan periode yang dapat dibandingkan","Proporsi tetap di kelompok sama","Paling lama bertahan (triwulan)"])).add_to(changes_map)
        attach_grid_toggle(changes_map, {"changes": changes_layer}, "changes")
        title = '<div style="position:fixed;top:12px;left:60px;z-index:999;background:white;color:#222;padding:12px;border:1px solid #555;max-width:440px;line-height:1.5;font:14px Arial,sans-serif">'
        title += f'<strong>Berapa kali wilayah berganti kelompok? · {method.upper()}</strong><br>Perbandingan antartriwulan selama 2021-2025.<br><span style="color:#0072B2">■</span> Biru: 0 pergantian; <span style="color:#E69F00">■</span> kuning: 1-2; <span style="color:#D55E00">■</span> jingga tua: lebih dari 2.<br>Abu-abu: belum cukup periode untuk dibandingkan.<br>Arah pergantian dapat naik atau turun pada masing-masing fitur. Ini bukan jumlah gempa atau tingkat bahaya.<br>Arahkan kursor ke kotak untuk melihat jumlah pasangan yang dibandingkan.</div>'
        changes_map.get_root().html.add_child(folium.Element(title))
        change_path=root / f"maps/{method}_profile_changes_2021_2025.html"
        changes_map.save(str(change_path))
        paths.append(change_path)
        geojson={"type":"FeatureCollection","features":[{"type":"Feature","id":key,"properties":{"grid_id":key},"geometry":{"type":"Polygon","coordinates":value}} for key,value in geometry.items()]}
        (root / f"maps/{method}_grid_geometry.geojson").write_text(json.dumps(geojson),encoding="utf-8")
    return paths
