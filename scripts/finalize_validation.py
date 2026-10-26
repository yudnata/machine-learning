from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.common import load_prepared, load_model, write_json


def finalize():
    cfg,panel,train,X,bundle,meta=load_prepared(ROOT)
    km,km_meta=load_model(ROOT,"kmeans")
    db,db_meta=load_model(ROOT,"dbscan")
    execution=json.loads((ROOT/"reports/validation/notebook_execution.json").read_text())
    assert execution["status"]=="PASS"
    assert len(execution["notebooks"])==11
    assert all(n["code_cells"]==n["executed_cells"] for n in execution["notebooks"])
    assert panel.loc[panel["split"].eq("train")].groupby("grid_id")["period"].nunique().eq(20).all()
    np.testing.assert_allclose(bundle["scaler"].mean_,train[cfg["features"]].mean().to_numpy())
    empty=panel.loc[panel["state"].eq("NO_EVENTS")]
    assert empty[["avg_magnitude","max_magnitude","avg_depth"]].isna().all().all()
    scaled=pd.read_csv(ROOT/"data/processed/features_scaled.csv")
    assert len(scaled)==panel["state"].eq("ELIGIBLE").sum()
    assert np.isfinite(scaled[cfg["features"]]).all().all()
    test=pd.read_csv(ROOT/"reports/testing_2026/labels_all_2026.csv")
    assert test["split"].ne("train").all()
    assert set(test.loc[test["split"].eq("test"),"period"])=={"2026Q1","2026Q2","2026Q3"}
    assert set(test.loc[test["split"].eq("test_partial"),"period"])=={"2026Q4"}
    notebook_checks=[]
    for path in sorted((ROOT/"notebooks").rglob("*.ipynb")):
        nb=nbformat.read(path,as_version=4)
        nbformat.validate(nb)
        assert nb.metadata.kernelspec.name=="seismic-python312"
        for cell in nb.cells:
            if cell.cell_type=="code":
                compile(cell.source,str(path),"exec")
                assert cell.execution_count is not None
                assert not any(o.output_type=="error" for o in cell.outputs)
        notebook_checks.append({"file":str(path.relative_to(ROOT)),"cells":len(nb.cells),"status":"PASS"})
    maps=[]
    for method in ["kmeans","dbscan"]:
        path=ROOT/f"maps/{method}_profiles_by_quarter.html"
        text=path.read_text(encoding="utf-8")
        assert "select.onchange" in text
        assert '"2026Q3"' in text and '"2021Q1"' in text
        assert "Polygon" in text and "properties" in text
        geojson=json.loads((ROOT/f"maps/{method}_grid_geometry.geojson").read_text())
        assert all(f["geometry"]["type"]=="Polygon" for f in geojson["features"])
        maps.append({"method":method,"geometry_count":len(geojson["features"]),"html_bytes":path.stat().st_size})
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/"src").glob("*.py"))}
    write_json(ROOT/"reports/validation/data_integrity.json",{
        "status":"PASS","train_only_scaler":"PASS","20_historical_quarters":"PASS","zero_statistics_missing":"PASS",
        "test_complete_and_partial_separate":"PASS","notebooks":notebook_checks,"maps_static_checks":maps,
        "source_sha256":hashes,"note":"Pemeriksaan peta di sini bersifat struktural; hasil pemeriksaan browser dicatat terpisah jika tersedia."})
    comparison=pd.read_csv(ROOT/"reports/evaluation/model_comparison.csv")
    holdout=pd.read_csv(ROOT/"reports/testing_2026/holdout_metrics.csv")
    sensitivity=pd.read_csv(ROOT/"reports/stability/grid_and_magnitude_ari.csv")
    baseline=sensitivity.loc[(sensitivity["perturbation"].eq("grid_km") & sensitivity["value"].eq(cfg["grid_km"]))
                             | (sensitivity["perturbation"].eq("min_magnitude") & sensitivity["value"].eq(cfg["min_magnitude"]))]
    np.testing.assert_allclose(baseline["ari_same_events_including_noise"],1.0,atol=1e-10)
    completeness=pd.read_csv(ROOT/"reports/evaluation/completeness_diagnostic.csv")
    support=pd.read_csv(ROOT/"reports/stability/minimum_event_support.csv")
    low_support_fraction=float(train["low_support"].mean())
    n_dbscan_clusters=len(set(db.labels_)-{-1})
    content=f'''# Ringkasan hasil eksekusi

Snapshot ini dihasilkan dari data EMSC nyata yang tersimpan di data/raw/monthly. Semua 11 notebook berhasil dijalankan pada kernel Python 3.12.11 (Seismic Research). Notebook asli tidak ditulis oleh pipeline ini.

- Baris mentah API: {meta['n_raw']:,}.
- Baris bersih sesuai cakupan/tipe event: {meta['n_clean']:,}; ditolak: {meta['n_rejected']:,}.
- Observasi grid-triwulan training layak: {meta['n_train_eligible']:,}.
- Cakupan historis: {meta['n_train_periods']} triwulan, 2021Q1-2025Q4.
- Proporsi baris training dengan 1-4 event: {low_support_fraction:.2%}; ini batas penting interpretasi profil.
- K-Means terpilih: k={km_meta['selected_k']}.
- DBSCAN terpilih: eps={db_meta['eps']:.6f}, min_samples={db_meta['min_samples']}, {n_dbscan_clusters} cluster, noise={(db.labels_==-1).mean():.2%}.
- Status DBSCAN hanya diagnostik: {db_meta['diagnostic_only']}.
- Testing lengkap: 2026Q1-Q3; 2026Q4 dipisahkan sebagai parsial.

## Evaluasi historis

```text
{comparison.to_string(index=False)}
```

Metrik perbandingan harus dibaca bersama scope. Tabel parameter search memakai rata-rata tiga sampel Silhouette; tabel ringkasan di atas memakai seed 42. K-Means/DBSCAN tidak memiliki accuracy tanpa label acuan.

## Testing temporal

```text
{holdout.to_string(index=False)}
```

DBSCAN testing adalah ekstensi core-radius. Noise_fraction di tabel testing berarti seluruh titik yang tidak ditugaskan (termasuk ambigu), bukan hasil refit DBSCAN; ambiguous_fraction dilaporkan terpisah. K-Means selalu memberi label, sehingga assignment 100% bukan bukti semua pengamatan cocok dengan profil historis.

## Sensitivitas resolusi dan magnitudo

```text
{sensitivity.to_string(index=False)}
```

ARI dihitung pada event ID bersama dan bersifat event-weighted. Perbedaan grid, threshold magnitudo, dan cakupan overlap harus dibaca sebelum menyatakan pola stabil. ARI bukan validasi terhadap kondisi tektonik sebenarnya.

## Sensitivitas dukungan event

```text
{support.to_string(index=False)}
```

Mayoritas unit mempunyai dukungan event rendah. Pada satu event, mean dan maksimum M sama serta proporsi dangkal hanya 0/1; struktur cluster dapat dipengaruhi diskretisasi itu. Jangan menyebut semua transisi sebagai perubahan regional yang kuat tanpa pemeriksaan dukungan, kestabilan dan kualitas katalog.

## Diagnostik completeness katalog

```text
{completeness.to_string(index=False)}
```

Ambang M>=3.0 belum membuktikan katalog lengkap; jenis magnitudo belum dihomogenisasi. Lihat docs/metodologi.md untuk batas seluruh keputusan. Temuan ini masih analisis eksploratif aktivitas dalam katalog, bukan peta risiko atau bukti final perubahan seismik fisik.

## Bukti verifikasi

reports/validation/environment.json, notebook_execution.json, data_integrity.json dan tests.xml mencatat interpreter, eksekusi, integritas, serta pengujian. Peta HTML tersedia di maps; riwayat per grid dan matriks transisi ada di reports/temporal.
'''
    (ROOT/"reports/ringkasan_hasil.md").write_text(content,encoding="utf-8")
    print(content[:1800])


if __name__=="__main__":
    finalize()
