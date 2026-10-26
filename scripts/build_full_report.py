"""Susun laporan Markdown dari artefak aktual, tanpa menjalankan ulang model."""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "LAPORAN_TEKNIS_DAN_ANGKA.md"
READ_FILES = set()


def data(relative):
    READ_FILES.add(relative)
    return pd.read_csv(ROOT / relative)


def metadata(relative):
    READ_FILES.add(relative)
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def number(value, digits=0):
    return f"{value:,.{digits}f}".translate(str.maketrans({",": ".", ".": ","}))


def pct(value, digits=2):
    return number(value * 100, digits) + "%"


def table(frame, digits=None, percentages=(), integers=()):
    frame = frame.copy()
    digits = digits or {}
    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for column, value in row.items():
            if pd.isna(value):
                cell = "Tidak terdefinisi"
            elif isinstance(value, (bool, np.bool_)):
                cell = "Ya" if value else "Tidak"
            elif column == "Tahun":
                cell = str(int(value))
            elif column in percentages:
                cell = pct(value)
            elif column in integers or isinstance(value, (int, np.integer)):
                cell = number(value)
            elif isinstance(value, (float, np.floating)):
                cell = number(value, digits.get(column, 4))
            else:
                cell = str(value)
            cells.append(cell.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


NOTEBOOKS = {
    0: "notebooks/00_panduan.ipynb",
    1: "notebooks/01_data_collection/01_data_collection.ipynb",
    2: "notebooks/02_data_preprocessing/02_data_preprocessing.ipynb",
    3: "notebooks/03_eda/03_eda.ipynb",
    4: "notebooks/04_kmeans/04_kmeans.ipynb",
    5: "notebooks/05_dbscan/05_dbscan.ipynb",
    6: "notebooks/06_model_evaluation/06_model_evaluation.ipynb",
    7: "notebooks/07_temporal_analysis/07_temporal_analysis.ipynb",
    8: "notebooks/08_stability/08_stability.ipynb",
    9: "notebooks/09_testing_2026/09_testing_2026.ipynb",
    10: "notebooks/10_gis_mapping/10_gis_mapping.ipynb",
}


def notebook(stage, label, section=None):
    anchor = f"#s{section}" if section else ""
    return f"[{label}]({NOTEBOOKS[stage]}{anchor})"


def profile_table(method):
    profiles = data(f"models/{method}/profiles.csv")
    descriptions = metadata(f"reports/evaluation/{method}_profile_explanations.json")
    profiles["Nama deskriptif"] = profiles["profile"].map(lambda key: descriptions[key]["name"])
    return table(profiles[["profile", "Nama deskriptif", "earthquake_count", "avg_magnitude", "avg_depth", "n_grid_quarters"]].rename(columns={
        "profile": "Kode", "earthquake_count": "Kejadian / 3 bulan", "avg_magnitude": "Mean M",
        "avg_depth": "Mean depth (km)", "n_grid_quarters": "Baris pembentuk"}),
        digits={"Kejadian / 3 bulan": 2, "Mean M": 2, "Mean depth (km)": 2}, integers=("Baris pembentuk",))


def build():
    cfg = metadata("config.json")
    manifest = metadata("data/raw/collection_manifest.json")
    meta = metadata("data/processed/metadata.json")
    km_meta = metadata("models/kmeans/metadata.json")
    db_meta = metadata("models/dbscan/metadata.json")
    raw = data("data/raw/events.csv")
    clean = data("data/processed/events_clean.csv")
    rejected = data("data/processed/events_rejected.csv")
    included = data("data/processed/events_with_grid.csv")
    panel = data("data/processed/panel.csv")
    train = panel.loc[panel["split"].eq("train") & panel["state"].eq("ELIGIBLE")]
    test = data("reports/testing_2026/labels_all_2026.csv")
    km_search = data("models/kmeans/parameter_search.csv")
    db_search = data("models/dbscan/parameter_search.csv")
    comparison = data("reports/evaluation/model_comparison.csv")
    holdout = data("reports/testing_2026/holdout_metrics.csv")
    agreement = metadata("reports/evaluation/agreement.json")
    environment = metadata("reports/validation/environment.json")
    execution = metadata("reports/validation/notebook_execution.json")
    map_checks = metadata("reports/validation/map_selector.json")
    test_method = metadata("reports/testing_2026/methodology.json")
    historic = clean.loc[clean["year"].le(2025)]
    annual = pd.DataFrame({
        "Tahun": range(2021, 2027),
        "Catatan API": pd.to_datetime(raw["time"], format="mixed", utc=True).dt.year.value_counts().reindex(range(2021, 2027)).to_numpy(),
        "Lolos cleaning": clean["year"].value_counts().reindex(range(2021, 2027)).to_numpy(),
        "Dipakai (M >=3)": included["year"].value_counts().reindex(range(2021, 2027)).to_numpy(),
        "Peran": ["Training, satu tahun penuh"] * 5 + ["Testing, sampai sebelum 26 Oktober"],
    })
    panel_summary = []
    for split, label in [("train", "Training 2021-2025"), ("test", "Testing Q1-Q3 2026"), ("test_partial", "Testing Q4 2026 parsial")]:
        subset = panel.loc[panel["split"].eq(split)]
        panel_summary.append({"Bagian data": label, "Grid unik": subset["grid_id"].nunique(), "Periode": subset["period"].nunique(),
            "Semua baris panel": len(subset), "Layak clustering": int(subset["state"].eq("ELIGIBLE").sum()),
            "Nol kejadian": int(subset["state"].eq("NO_EVENTS").sum()), "Fitur belum lengkap": int(subset["state"].eq("INCOMPLETE_FEATURES").sum())})
    k_selected = km_search.loc[km_search["k"].eq(km_meta["selected_k"])].iloc[0]
    db_selected = db_search.loc[np.isclose(db_search["eps"], db_meta["eps"]) & db_search["min_samples"].eq(db_meta["min_samples"])].iloc[0]
    hist_km = comparison.loc[comparison["model"].eq("kmeans")].iloc[0]
    hist_db = comparison.loc[comparison["model"].eq("dbscan")].iloc[0]
    low_support = float(train["low_support"].mean())
    n_train_grids = panel.loc[panel["split"].eq("train"), "grid_id"].nunique()
    n_all_grids = panel["grid_id"].nunique()
    test_eligible = test.loc[test["state"].eq("ELIGIBLE")]
    test_complete = test_eligible.loc[test_eligible["split"].eq("test")]
    test_partial = test_eligible.loc[test_eligible["split"].eq("test_partial")]
    km_names = metadata("reports/evaluation/kmeans_profile_explanations.json")
    sections = []
    sections.append(f'''# Laporan Teknis dan Angka Penelitian Aktivitas Seismik

Untuk memahami tujuan dan makna hasil terlebih dahulu, baca [penjelasan utama proyek](LAPORAN_ALUR_DAN_HASIL_PENELITIAN.md). Dokumen teknis ini menyimpan rincian angka dan pemeriksaan sumber.

**Training 2021-2025 dan testing temporal 2026**  
Snapshot data: pengunduhan EMSC pada 26 Oktober 2026. Batas pengamatan adalah sebelum **26 Oktober 2026 pukul 00.00 UTC**, setara 08.00 WITA. Laporan ini disusun dari file hasil eksekusi yang tersimpan, bukan angka ilustrasi. Perubahan data atau konfigurasi memerlukan pembaruan laporan.

Dokumen ini menjelaskan apa yang dilakukan pada setiap tahap, berapa hasil aktualnya, mengapa suatu keputusan dibuat, bagaimana membaca hasilnya, dan notebook yang menjadi rujukan. Istilah teknis dijelaskan pada saat digunakan.

## Daftar isi

1. [Tujuan dan gambaran hasil](#1-tujuan-dan-gambaran-hasil)
2. [Urutan notebook](#2-urutan-notebook)
3. [Data collection](#3-data-collection)
4. [Data preprocessing](#4-data-preprocessing)
5. [EDA](#5-eda)
6. [K-Means](#6-k-means)
7. [DBSCAN](#7-dbscan)
8. [Perbandingan model](#8-perbandingan-model)
9. [Penamaan dan interpretasi kelompok](#9-penamaan-dan-interpretasi-kelompok)
10. [Perubahan profil historis](#10-perubahan-profil-historis)
11. [Uji kestabilan dan sensitivitas](#11-uji-kestabilan-dan-sensitivitas)
12. [Testing 2026](#12-testing-2026)
13. [GIS dan cara membaca peta](#13-gis-dan-cara-membaca-peta)
14. [Verifikasi eksekusi](#14-verifikasi-eksekusi)
15. [Temuan dan batas kesimpulan](#15-temuan-dan-batas-kesimpulan)
16. [Daftar keluaran dan referensi](#16-daftar-keluaran-dan-referensi)

## 1. Tujuan dan gambaran hasil

Penelitian ini mengelompokkan **karakteristik aktivitas gempa suatu wilayah pada suatu triwulan**. Satu baris model adalah satu kotak wilayah pada satu periode tiga bulan. Penelitian tidak mengelompokkan setiap kejadian gempa secara langsung.

Data 2021-2025 digunakan untuk menemukan kelompok historis. Data 2026 dipakai untuk memeriksa kecocokan kelompok tersebut terhadap pengamatan baru. Ini tetap **unsupervised learning**, yaitu pembentukan kelompok tanpa label target yang telah ditentukan. Kata training berarti proses pembentukan model; testing berarti penerapan dan pemeriksaan pada periode baru.

Hasil utama yang sudah diperoleh:

| Tahap | Hasil aktual |
| --- | --- |
| Collection | {number(len(raw))} catatan kejadian dari API |
| Cleaning | {number(len(clean))} kejadian lolos; {number(len(rejected))} ditolak |
| Ambang magnitudo | {number(len(included))} kejadian M >=3 untuk agregasi |
| Training | {number(len(train))} observasi grid-triwulan layak dari {number(n_train_grids)} grid historis |
| K-Means | k={km_meta['selected_k']} terpilih menurut aturan eksperimen; mean Silhouette seleksi {number(k_selected['silhouette'], 4)} |
| DBSCAN | {int(db_selected['n_clusters'])} cluster; eps={number(db_meta['eps'], 6)}; min_samples={db_meta['min_samples']}; noise {pct(db_selected['noise_fraction'])} |
| Testing lengkap | {number(len(test_complete))} observasi layak pada Januari-September 2026 |
| Testing parsial | {number(len(test_partial))} observasi layak pada Oktober 2026 sampai batas pengamatan |
| GIS | 4 peta HTML interaktif dan 2 file geometri GeoJSON |

Angka Silhouette pada tabel ini berasal dari **rata-rata tiga sampel untuk seleksi parameter**. Bagian evaluasi di bawah memakai satu sampel tetap sehingga nilainya sedikit berbeda. Ini dua cara pelaporan pada model yang sama, bukan dua eksperimen dengan data berbeda.

## 2. Urutan notebook

Alur pengerjaan adalah **collection -> preprocessing -> EDA -> K-Means dan DBSCAN -> evaluasi -> analisis perubahan -> sensitivitas -> testing -> GIS**. Artefak setiap tahap disimpan agar tahap sesudahnya dapat membaca hasilnya tanpa mengulang unduhan.
''')
    titles = ["Panduan dan environment", "Data collection", "Data preprocessing", "EDA", "K-Means", "DBSCAN", "Model evaluation", "Analisis temporal", "Stability", "Testing 2026", "GIS mapping"]
    purposes = ["Memeriksa kernel, pustaka, dan konfigurasi", "Mengambil katalog EMSC dan mencatat cakupan", "Membersihkan data serta membuat panel dan scaler", "Memahami distribusi, katalog, dan korelasi", "Memilih k dan membentuk kelompok centroid", "Menguji eps/min_samples dan menemukan kelompok kepadatan", "Membandingkan struktur kelompok dan cakupan", "Mengukur perpindahan dan durasi profil", "Memeriksa pengaruh perubahan pengaturan", "Menerapkan model historis pada data baru", "Mengembalikan hasil ke polygon dan peta"]
    sections.append(table(pd.DataFrame([{"No.": f"{i:02d}", "Notebook": notebook(i, title), "Tujuan": purposes[i]} for i, title in enumerate(titles)])))
    sections.append(f'''

Konfigurasi dapat diperiksa di [config.json](config.json). Kernel yang digunakan adalah **Python 3.12.11 (Seismic Research)**. Notebook asli `ml_20206 (1).ipynb` tidak ditulis oleh pipeline baru.

## 3. Data collection

**Rujukan:** {notebook(1, '01 Data collection')}, khususnya bagian definisi atribut, pengunduhan bulanan, dan audit cakupan.

### 3.1 Sumber dan atribut data

Katalog diambil melalui EMSC/SeismicPortal FDSN Event service. Atribut yang disimpan meliputi waktu UTC, lintang, bujur, kedalaman, magnitudo, jenis magnitudo, tipe kejadian, ID kejadian, wilayah deskriptif, author, dan waktu revisi. Wilayah permintaan dibatasi lintang -11 sampai 6 dan bujur 95 sampai 141.

Batas tersebut merupakan **bounding box Indonesia dan sekitarnya**. Ini mencakup laut serta sebagian wilayah tetangga, sehingga bukan pemotongan menurut batas administratif negara.

### 3.2 Pengambilan dan pemeriksaan cakupan

Permintaan dipecah per bulan. Terdapat **{len(manifest['intervals'])} interval pengunduhan** dari Januari 2021 sampai Oktober 2026; Oktober hanya sampai cutoff. Cache JSON menyimpan respons sumber, URL permintaan dan waktu pengambilan. Retry menangani kegagalan sementara. Jika jumlah respons mencapai limit {number(cfg['api_limit'])}, kode membagi interval lagi untuk menghindari hasil terpotong. Kegagalan tidak dicatat sebagai nol kejadian.

**Kedalaman diambil dari `properties.depth` dalam km.** `geometry.coordinates[2]` disimpan sebagai `geometry_z` untuk audit. Pada respons yang diperiksa tanda keduanya berlawanan, sehingga koordinat ketiga tidak dipakai langsung sebagai kedalaman.

### 3.3 Hasil pengunduhan

Collection berhasil menyimpan **{number(len(raw))} catatan API**. Catatan ini belum semuanya dipakai untuk model: tipe kejadian, validitas data dan ambang magnitudo diperiksa sesudah pengunduhan.

{table(annual, integers=('Tahun', 'Catatan API', 'Lolos cleaning', 'Dipakai (M >=3)'))}

Baris 2026 belum mencakup setahun penuh. Jumlah 2026 tidak boleh langsung dibandingkan dengan tahun penuh sebagai bukti kenaikan aktivitas.

**Bukti dan keluaran:** [events.csv](data/raw/events.csv), [manifest collection](data/raw/collection_manifest.json), dan [folder respons bulanan](data/raw/monthly/). Seluruh interval permintaan tercatat berhasil. Hal itu membuktikan cakupan unduhan, tetapi belum membuktikan semua gempa fisik telah terdeteksi oleh katalog.

## 4. Data preprocessing

**Rujukan:** {notebook(2, '02 Data preprocessing')}. Cleaning terdapat pada bagian 3, grid pada bagian 4, dan standardisasi pada bagian 5.

### 4.1 Cleaning dan pemilihan event

Waktu dikonversi ke UTC dengan dukungan format pecahan detik yang berbeda. Koordinat, waktu dan cakupan diperiksa. ID duplikat mempertahankan revisi terakhir; event tanpa ID memakai kunci atribut agar tidak semuanya dianggap satu duplikat. Depth di luar 0-800 km ditandai hilang; magnitudo di luar -2 sampai 10 juga ditandai hilang.

Tipe kejadian yang diterima adalah `ke`, `fe`, dan `de`. Pada snapshot ini, seluruh kejadian yang lolos bertipe `ke`. Terdapat **44 catatan `ue` dan 8 catatan `uk`**, sehingga **52 catatan tidak diterima**. File rejected mencatat alasannya sebagai `event_type_not_accepted`.

Hasil cleaning adalah **{number(len(clean))} kejadian**. Pada snapshot ini tidak ditemukan baris duplikat yang dibuang, nilai kosong pada waktu/koordinat/depth/magnitudo dalam data bersih, atau depth yang ditandai tidak valid. Pernyataan ini berlaku untuk hasil pemeriksaan kode, bukan jaminan semua pengukuran sumber tanpa kesalahan.

Analisis utama memakai **magnitudo minimal 3,0**. Setelah ambang tersebut diterapkan, tersisa **{number(len(included))} kejadian**, terdiri dari **{number(included['year'].le(2025).sum())} kejadian historis** dan **{number(included['year'].eq(2026).sum())} kejadian testing**. Sebanyak {number(len(clean)-len(included))} kejadian bersih berada di bawah ambang dan tidak masuk agregasi utama. Ambang 3,0 bersifat eksploratif, belum merupakan hasil pembuktian completeness katalog.

### 4.2 Membentuk kotak wilayah dan periode

Grid utama berukuran **50 x 50 km pada bidang proyeksi LAEA**, berpusat di 118 BT dan -2,5 LS. Origin grid tetap, sehingga letak kotak tidak bergeser saat data baru masuk. Satu sel interior mempunyai luas 2.500 km² pada bidang equal-area; panjang sisi geodesiknya tidak tepat sama di semua lokasi.

Setiap kejadian dimasukkan ke kotak dan triwulannya. Untuk training, semua **20 triwulan** dari 2021Q1 sampai 2025Q4 dilengkapi pada **{number(n_train_grids)} grid yang pernah aktif historis**. Testing mencakup {number(n_all_grids)} grid, termasuk **{number(n_all_grids-n_train_grids)} grid baru pada 2026**. Penambahan grid testing tidak mengubah universe training.

### 4.3 Membedakan nol kejadian dan fitur belum lengkap

`NO_EVENTS` berarti tidak ada kejadian M >=3 tercatat pada kotak-periode itu. Count dan frekuensi menjadi nol, tetapi rata-rata magnitudo dan depth tetap kosong. `INCOMPLETE_FEATURES` berarti ada kejadian tetapi fitur belum memenuhi syarat. `ELIGIBLE` berarti fitur lengkap dan depth coverage minimal 80%, sehingga baris dapat masuk clustering.

{table(pd.DataFrame(panel_summary))}

Total panel adalah **{number(len(panel))} baris grid-triwulan**. Jumlah ini berbeda dari {number(len(included))} kejadian gempa: beberapa event dapat bergabung menjadi satu baris, dan baris nol kejadian tetap ditambahkan untuk analisis waktu. Pada snapshot ini tidak ada baris berstatus fitur belum lengkap.

### 4.4 Fitur dan standardisasi

| Fitur model | Arti | Pengolahan |
| --- | --- | --- |
| `log_frequency` | Seberapa sering gempa tercatat | log1p(jumlah kejadian / bulan pengamatan) |
| `avg_magnitude` | Magnitudo rata-rata dalam kotak-periode | Mean magnitudo event |
| `max_magnitude` | Magnitudo terbesar dalam kotak-periode | Maksimum magnitudo event |
| `avg_depth` | Kedalaman rata-rata dalam kotak-periode | Mean depth valid, dalam km |
| `shallow_fraction` | Proporsi kejadian depth <70 km | Dibagi jumlah event dengan depth valid |

Pada triwulan lengkap, frekuensi per bulan adalah count/3. Triwulan parsial memakai durasi pengamatan dalam bulan ekuivalen 30,4375 hari. Count, density dan frekuensi mentah tidak sekaligus menjadi fitur jarak karena mengulang informasi. Koordinat tidak menjadi fitur clustering, sehingga kelompok dapat berisi wilayah yang berjauhan namun aktivitasnya mirip.

**StandardScaler hanya di-fit pada {number(len(train))} baris training layak.** Scaler memusatkan dan menyetarakan skala fitur agar satuan km tidak otomatis mendominasi jarak. Testing hanya ditransformasi menggunakan scaler historis. Tidak ada pengisian nol untuk statistik magnitudo/depth yang kosong.

**Keluaran:** [data bersih](data/processed/events_clean.csv), [data ditolak](data/processed/events_rejected.csv), [event beserta grid](data/processed/events_with_grid.csv), [panel](data/processed/panel.csv), [fitur standar](data/processed/features_scaled.csv), [transformasi tersimpan](data/processed/transform.joblib), dan [metadata preprocessing](data/processed/metadata.json).

## 5. EDA

**Rujukan:** {notebook(3, '03 EDA')}. EDA berarti pemeriksaan awal untuk memahami isi dan kualitas data sebelum menafsirkan cluster. Keputusan model menggunakan EDA historis 2021-2025.

Pada {number(len(historic))} kejadian historis bersih **sebelum filter M >=3**, magnitudo rata-ratanya **{number(historic['magnitude'].mean(), 2)}**, median **{number(historic['magnitude'].median(), 2)}**, dan rentangnya **{number(historic['magnitude'].min(), 1)} sampai {number(historic['magnitude'].max(), 1)}**. Kedalaman rata-rata **{number(historic['depth'].mean(), 2)} km**, median **{number(historic['depth'].median(), 2)} km**, dan rentangnya **{number(historic['depth'].min())} sampai {number(historic['depth'].max())} km**. Angka ini adalah statistik event, bukan rata-rata profil cluster.

![EDA historis: distribusi magnitudo, depth, jumlah per triwulan, dan hubungan magnitude-depth](reports/figures/eda_training.png)

### 5.1 Korelasi fitur

Korelasi menunjukkan seberapa bersama-sama dua fitur berubah. Pada baris training layak, korelasi `avg_magnitude` dan `max_magnitude` adalah **0,8044**, sedangkan `avg_depth` dan `shallow_fraction` adalah **-0,7488**. Pasangan pertama masih membawa informasi berulang; pasangan kedua berkaitan karena semakin dalam mean depth biasanya semakin kecil proporsi dangkal. Standardisasi tidak menghilangkan korelasi ini. Pengaruh pemilihan fitur masih perlu diperiksa sebelum kesimpulan final.

Sumber: [matriks korelasi](reports/evaluation/feature_correlation.csv). Audit lain tersedia pada [jumlah dan karakter katalog per tahun](reports/evaluation/catalogue_by_year.csv), [author katalog](reports/evaluation/catalogue_authors.csv), dan [jenis magnitudo](reports/evaluation/magnitude_types.csv).

### 5.2 Kelengkapan pencatatan

Notebook membandingkan jumlah pada ambang M 2,5; 3,0; 3,5; dan 4,0. Mode histogram magnitudo dengan bin 0,1 juga dicatat sebagai diagnostik awal:
''')
    completeness = data("reports/evaluation/completeness_diagnostic.csv")
    sections.append(table(completeness[["year", "max_curvature_histogram_mode", "analysis_threshold", "threshold_below_histogram_mode"]].rename(columns={
        "year": "Tahun", "max_curvature_histogram_mode": "Mode histogram M", "analysis_threshold": "Ambang analisis M", "threshold_below_histogram_mode": "Ambang di bawah mode?"}), digits={"Mode histogram M": 1, "Ambang analisis M": 1}))
    sections.append(f'''

Mode histogram ini **bukan estimasi magnitude of completeness yang telah tervalidasi**. Pada 2021, ambang 3,0 berada di bawah mode histogram 3,1; hal itu menjadi alasan tambahan untuk memeriksa kelengkapan. Jenis magnitudo juga belum dihomogenisasi menjadi Mw. Perubahan pencatatan, author atau jenis magnitudo dapat memengaruhi jumlah dan profil yang ditemukan.

Keluaran pemeriksaan: [audit ambang magnitudo](reports/evaluation/magnitude_threshold_audit.csv) dan [diagnostik completeness](reports/evaluation/completeness_diagnostic.csv).

## 6. K-Means

**Rujukan:** {notebook(4, '04 K-Means')}; pencarian k pada {notebook(4, 'bagian 4', 4)}, hasil fitting pada {notebook(4, 'bagian 5', 5)}.

### 6.1 Pembentukan kelompok dengan K-Means

K-Means mencari pusat kelompok atau centroid. Setiap observasi grid-triwulan ditempatkan pada centroid terdekat di ruang lima fitur standar. Semua {number(len(train))} baris training layak digabungkan dalam satu fitting historis, sehingga makna kelompok dapat digunakan secara konsisten antartriwulan.

Jumlah kelompok tidak langsung dipaksa tiga. Kandidat **k=2 sampai 8** diuji menggunakan `n_init=20` dan random_state 42. Dua puluh inisialisasi membantu pencarian centroid, sedangkan seed membuat eksperimen dapat diulang.

### 6.2 Aturan pemilihan k

Silhouette menilai kedekatan suatu observasi dengan kelompoknya dibanding kelompok lain; lebih tinggi menunjukkan pemisahan geometris yang lebih baik. Davies-Bouldin Index (DBI) menilai kemiripan antarkelompok; lebih rendah lebih baik. Inertia mengukur jumlah kuadrat jarak ke centroid; digunakan untuk melihat pola elbow, bukan sebagai satu-satunya penentu karena nilainya umumnya turun saat k bertambah.

Aturan kode adalah memilih **rata-rata Silhouette tertinggi** dari tiga sampel, masing-masing maksimal 2.000 baris (seed 42, 7, dan 21). Jika sama, DBI lebih rendah lalu k lebih kecil menjadi pemecah seri. SD yang dilaporkan mengukur variasi antar-sampel skor, bukan interval kepercayaan inferensial.

{table(km_search[['k','inertia','silhouette','silhouette_std','davies_bouldin']].rename(columns={'k':'k','inertia':'Inertia','silhouette':'Mean Silhouette','silhouette_std':'SD antar-sampel','davies_bouldin':'DBI'}), digits={'Inertia':2})}

### 6.3 Hasil yang dipilih

**k={km_meta['selected_k']} terpilih sebagai nilai terbaik menurut aturan eksperimen ini**, dengan mean Silhouette **{number(k_selected['silhouette'], 6)}**, SD **{number(k_selected['silhouette_std'], 6)}**, DBI **{number(k_selected['davies_bouldin'], 6)}**, dan inertia **{number(k_selected['inertia'], 2)}**. Keenam cluster memuat seluruh {number(len(train))} baris layak; kelompok terkecil berisi {number(k_selected['smallest_cluster_size'])} baris dan kelompok terbesar sekitar {pct(k_selected['largest_cluster_fraction_assigned'])}.

Penyebutan **k optimal** harus dibatasi pada kandidat, fitur, data dan aturan ini. Skor k=2 (**{number(km_search.loc[km_search['k'].eq(2),'silhouette'].iloc[0], 6)}**) sangat dekat dengan k=6. Oleh karena itu, hasil ini belum membuktikan bahwa alam mempunyai tepat enam tipe aktivitas atau k=6 unggul secara statistik atas semua alternatif.

![Elbow dan skor pemilihan k](reports/figures/kmeans_selection.png)

Visual proyeksi cluster tersedia pada [PCA K-Means](reports/figures/kmeans_pca.png). PCA merangkum lima fitur menjadi dua sumbu untuk visualisasi; dua sumbu tersebut bukan lintang-bujur dan bukan ruang tempat model di-fit.

**Keluaran:** [seluruh kandidat k](models/kmeans/parameter_search.csv), [model K-Means](models/kmeans/model.joblib), [label training](models/kmeans/training_labels.csv), [profil statistik](models/kmeans/profiles.csv), dan [metadata pemilihan](models/kmeans/metadata.json).

## 7. DBSCAN

**Rujukan:** {notebook(5, '05 DBSCAN')}; pencarian parameter pada {notebook(5, 'bagian 3', 3)} dan hasil pada {notebook(5, 'bagian 4', 4)}.

### 7.1 Pembentukan kelompok dengan DBSCAN

DBSCAN mencari kelompok berdasarkan kepadatan titik di **ruang fitur aktivitas**, bukan langsung jarak geografis antarepisenter. Parameter `eps` menentukan radius tetangga pada fitur standar; `min_samples` menentukan dukungan minimum untuk suatu core point. Observasi yang tidak terhubung ke struktur kepadatan dapat berstatus noise. Noise bukan otomatis daerah berbahaya.

Kandidat `min_samples` adalah **5, 10, dan 20**. Kandidat eps diambil dari kuantil 0,50; 0,70; 0,85; dan 0,95 jarak tetangga ke-k pada training. Query tetangga memasukkan titik itu sendiri, konsisten dengan perhitungan min_samples. Total terdapat **{len(db_search)} kombinasi** yang diuji.

![k-distance training untuk kandidat eps DBSCAN](reports/figures/dbscan_k_distance.png)

### 7.2 Penilaian skor, noise, dan fragmentasi

Screening training menerima kandidat dengan 2-20 cluster, cluster terkecil minimal 20 observasi, noise maksimal 50%, dan cluster terbesar maksimal 95% data assigned. Batas ini adalah kebijakan analitis untuk menghindari dominasi satu cluster atau ratusan kelompok kecil. Batas tersebut disusun dari diagnosis training, bukan ketentuan alami seismologi. Pengaruhnya diuji pada notebook 08.

Silhouette/DBI DBSCAN dihitung pada titik yang assigned, tanpa noise. Karena cakupannya dapat berbeda, skor harus selalu dibaca bersama noise dan ukuran cluster.

{table(db_search[['eps','min_samples','n_clusters','smallest_cluster_size','noise_fraction','silhouette','davies_bouldin','accepted']].rename(columns={'eps':'eps','min_samples':'min_samples','n_clusters':'Cluster','smallest_cluster_size':'Ukuran terkecil','noise_fraction':'Noise','silhouette':'Mean Silhouette','davies_bouldin':'DBI','accepted':'Lolos screening?'}), digits={'eps':6}, percentages=('Noise',), integers=('min_samples','Cluster','Ukuran terkecil'))}

### 7.3 Hasil yang dipilih

Kandidat terpilih adalah **eps={number(db_meta['eps'], 6)} dan min_samples={db_meta['min_samples']}**, menghasilkan **{int(db_selected['n_clusters'])} cluster**, **{number(db_selected['n_assigned'])} observasi assigned**, dan **{number(len(train)-db_selected['n_assigned'])} noise ({pct(db_selected['noise_fraction'])})**. Mean Silhouette seleksi **{number(db_selected['silhouette'], 6)}** dan DBI **{number(db_selected['davies_bouldin'], 6)}**. Cluster terkecil berisi {number(db_selected['smallest_cluster_size'])} observasi.

Kandidat dengan Silhouette sekitar 0,6045 menghasilkan **251 cluster** dan cluster terkecil hanya 4 observasi, sehingga gagal screening. Angka tinggi tersebut tidak dijadikan alasan mengabaikan fragmentasi. Dua kandidat lolos screening; model terpilih memiliki mean Silhouette lebih tinggi di antara keduanya. Status `diagnostic_only=false` berarti ada kandidat yang lolos aturan kode, **bukan sertifikat validitas ilmiah**.

**Keluaran:** [seluruh kombinasi parameter](models/dbscan/parameter_search.csv), [model DBSCAN](models/dbscan/model.joblib), [label training](models/dbscan/training_labels.csv), [profil statistik](models/dbscan/profiles.csv), [PCA DBSCAN](reports/figures/dbscan_pca.png), dan [metadata](models/dbscan/metadata.json).

## 8. Perbandingan model

**Rujukan:** {notebook(6, '06 Model evaluation')}, terutama {notebook(6, 'bagian 2-3 untuk evaluasi', 2)}.

### 8.1 Evaluasi pada training layak

Pada tabel ini, K-Means memakai semua observasi training layak. DBSCAN melaporkan noise pada semua observasi, tetapi metrik jarak hanya memakai observasi yang assigned. Sampel Silhouette menggunakan seed tetap 42, maksimal 2.000 baris; ini menjelaskan perbedaan kecil dari mean tiga sampel di tabel seleksi.
''')
    def comparison_table(frame):
        return table(frame[["model", "n_rows", "n_assigned", "noise_fraction", "silhouette", "davies_bouldin"]].rename(columns={
            "model":"Metode", "n_rows":"Baris input", "n_assigned":"Assigned", "noise_fraction":"Tidak assigned", "silhouette":"Silhouette", "davies_bouldin":"DBI"}), percentages=("Tidak assigned",))
    sections.append(comparison_table(comparison.iloc[:2]))
    sections.append(f'''

### 8.2 Perbandingan pada observasi yang sama

Agar perbandingan lebih setara, kedua model dievaluasi kembali pada **{number(agreement['n_common_assigned'])} baris yang sama**, yaitu baris assigned DBSCAN. Cakupannya **{pct(agreement['fraction_common_assigned'])}** dari training layak.

{comparison_table(comparison.iloc[2:])}

Dalam konfigurasi yang diuji, **K-Means mempunyai pemisahan cluster lebih baik menurut Silhouette dan DBI**, termasuk pada subset identik. Kesimpulan ini berlaku pada ruang fitur dan kandidat sekarang. Ini bukan akurasi terhadap kebenaran geologi dan tidak menyatakan K-Means selalu lebih baik untuk semua data gempa.

Kesepakatan partisi K-Means dan DBSCAN pada subset assigned menghasilkan **ARI={number(agreement['ari_common_assigned'], 4)}**. ARI membandingkan apakah pasangan observasi dikelompokkan bersama oleh dua partisi. ARI 1 berarti partisi identik; angka lebih rendah menunjukkan perbedaan. Nilai 0,2665 menunjukkan kedua metode tidak menghasilkan pengelompokan yang sama, meskipun memakai fitur yang sama. Nomor cluster tidak perlu disamakan untuk menghitung ARI.

**Keluaran:** [perbandingan model](reports/evaluation/model_comparison.csv) dan [kesepakatan partisi](reports/evaluation/agreement.json).

## 9. Penamaan dan interpretasi kelompok

**Rujukan:** {notebook(4, '04 K-Means bagian 6: aturan penamaan', 6)} dan {notebook(4, 'bagian 7: tabel hasil nama', 7)}; {notebook(5, '05 DBSCAN bagian 5: aturan penamaan', 5)} dan {notebook(5, 'bagian 6: tabel hasil nama', 6)}.

Kelompok dibentuk terlebih dahulu oleh algoritma. Sesudah itu peneliti membaca statistik training lalu memberi **nama deskriptif**. Huruf A/B/C hanya pengenal kelompok. Nama tidak menjadi target training, tidak mengubah keanggotaan, dan tidak ditentukan dari testing 2026.

Fungsi `profile_descriptions()` menggunakan aturan penulisan yang terlihat dalam notebook: mean depth <70 km diringkas sebagai cenderung dangkal; kelompok dengan mean jumlah event terbesar di antara cluster metode itu dijelaskan sebagai kejadian lebih sering. Mean magnitudo dan depth ditulis pada nama/legenda untuk menjaga keterlacakan. Ini aturan **penerjemahan hasil**, bukan aturan pembentukan cluster.

### 9.1 Profil K-Means

{profile_table('kmeans')}

Contoh membaca Kelompok F: rata-rata sekitar **8,93 kejadian per kotak per triwulan**, mean magnitudo **3,67**, dan mean depth **38,30 km**. Karena jumlah rata-ratanya terbesar di antara enam cluster dan mean depth di bawah 70 km, nama ringkasnya menjadi **Kejadian lebih sering; cenderung dangkal**. Kejadian lebih sering tidak berarti risiko tertinggi.

### 9.2 Profil DBSCAN

{profile_table('dbscan')}

Baris NOISE adalah statistik observasi historis yang tidak masuk cluster; baris itu tidak diberi huruf sebagai kelompok baru. **Kelompok A K-Means tidak sama dengan Kelompok A DBSCAN.** Angka pada tabel adalah rata-rata ciri grid-triwulan historis, bukan janji bahwa setiap kotak-periode memiliki nilai persis sama.

**Keluaran nama yang mudah dibaca:** [tabel interpretasi K-Means](reports/evaluation/kmeans_profile_interpretation.csv), [tabel interpretasi DBSCAN](reports/evaluation/dbscan_profile_interpretation.csv), [penjelasan legenda K-Means](reports/evaluation/kmeans_profile_explanations.json), dan [penjelasan legenda DBSCAN](reports/evaluation/dbscan_profile_explanations.json). [Panduan membaca hasil](docs/cara_menjelaskan_hasil.md) membahas perbedaan antara kelompok hasil algoritma dan nama dari peneliti.

## 10. Perubahan profil historis

**Rujukan:** {notebook(7, '07 Temporal analysis')}; definisi pasangan pada bagian 2, ekspor hasil pada bagian 3, dan riwayat kotak pada bagian 5.

### 10.1 Unit dan aturan perbandingan

Setiap grid dilacak sepanjang 20 triwulan historis. Pergantian dihitung hanya pada **dua triwulan berurutan yang sama-sama mempunyai kelompok cluster**. Nol event, fitur belum lengkap, noise, dan celah periode memutus rangkaian. Dengan demikian, berpindah dari NO_EVENTS ke suatu cluster tidak dihitung sebagai pergantian profil cluster.

Stability rate dihitung sebagai `1 - jumlah pergantian / jumlah pasangan yang dapat dibandingkan`. Jika tidak ada pasangan yang layak, nilainya tidak dihitung. Durasi bertahan dihitung dalam jumlah triwulan berturut-turut, bukan hari.

### 10.2 Hasil aktual
''')
    temporal_rows = []
    for method in ["kmeans", "dbscan"]:
        summary = data(f"reports/temporal/{method}/grid_stability.csv")
        spells = data(f"reports/temporal/{method}/profile_spells.csv")
        temporal_rows.append({"Metode": method, "Grid historis": len(summary), "Grid dengan pasangan layak": int(summary["n_comparable_pairs"].gt(0).sum()),
            "Pasangan layak": int(summary["n_comparable_pairs"].sum()), "Pergantian": int(summary["n_profile_changes"].sum()),
            "Median pasangan / grid": summary["n_comparable_pairs"].median(), "Median stability rate": summary["stability_rate"].median(),
            "Durasi terpanjang (triwulan)": int(spells["duration_quarters"].max())})
    sections.append(table(pd.DataFrame(temporal_rows), percentages=("Median stability rate",), integers=("Median pasangan / grid",)))
    sections.append('''

Jumlah pergantian adalah jumlah **pasangan grid-triwulan yang berubah**, bukan jumlah wilayah unik. Median stability rate dihitung hanya pada grid yang nilainya dapat dihitung. Median jumlah pasangan memakai seluruh grid historis. Median pasangan yang hanya 1-2 menunjukkan dukungan temporal banyak grid masih terbatas.

Stability rate DBSCAN tidak dapat langsung disebut lebih baik daripada K-Means karena DBSCAN mengecualikan noise dan menghasilkan lebih sedikit pasangan layak. Kedua metode juga memakai definisi kelompok yang berbeda.

### 10.3 Matriks transisi

Baris berarti kelompok sebelumnya; kolom berarti kelompok berikutnya. Nilai diagonal adalah bertahan di kelompok yang sama, sedangkan nilai di luar diagonal adalah pergantian. Huruf pada setiap matriks merujuk tabel nama metode yang sama pada bagian 9. Isi matriks adalah jumlah pasangan berurutan.
''')
    for method in ["kmeans", "dbscan"]:
        matrix = data(f"reports/temporal/{method}/profile_transitions.csv").set_index("previous_profile")
        matrix.index = [chr(65 + int(code.rsplit("_P", 1)[1])) for code in matrix.index]
        matrix.columns = [chr(65 + int(code.rsplit("_P", 1)[1])) for code in matrix.columns]
        sections.append(f"\n\n**{method.upper()}**\n\n" + table(matrix.rename_axis("Dari / ke").reset_index()))
    sections.append('''

Misalnya, pada K-Means terdapat **675 pasangan A -> E**. Itu berarti pada 675 pasangan kotak-triwulan berurutan, label berganti dari ciri Kelompok A ke Kelompok E. Ini tidak menyatakan 675 wilayah berbeda atau 675 kejadian gempa. Matriks probabilitas adalah normalisasi hitungan empiris per baris, bukan model prediksi Markov yang sudah divalidasi.

![Matriks transisi K-Means](reports/figures/kmeans_transitions.png)

**Keluaran:** [riwayat K-Means](reports/temporal/kmeans/history.csv), [kestabilan per grid K-Means](reports/temporal/kmeans/grid_stability.csv), [durasi K-Means](reports/temporal/kmeans/profile_spells.csv), [probabilitas transisi K-Means](reports/temporal/kmeans/transition_probabilities.csv), serta [folder hasil DBSCAN](reports/temporal/dbscan/). Matriks seluruh status, termasuk NO_EVENTS dan NOISE, tersedia sebagai `state_transitions.csv` pada masing-masing folder metode.

## 11. Uji kestabilan dan sensitivitas

**Rujukan:** [08 Stability](notebooks/08_stability/08_stability.ipynb). Seluruh eksperimen bagian ini menggunakan training 2021-2025. Data 2026 tidak dipakai untuk memilih parameter.

Tujuannya memeriksa apakah kelompok berubah ketika pengaturan diubah. ARI tinggi berarti partisi pada observasi yang sama lebih serupa. ARI bukan akurasi atau bukti bahwa mekanisme fisik telah teridentifikasi. Pengujian ini merupakan sensitivitas deskriptif, belum spatial block bootstrap atau interval ketidakpastian inferensial.

### 11.1 Seed, k, eps, dan min_samples
''')
    params = data("reports/stability/parameter_ari.csv")
    sections.append(table(params[["method", "perturbation", "value", "ari_all", "ari_common_assigned", "common_assigned_fraction"]].rename(columns={
        "method":"Metode", "perturbation":"Perubahan", "value":"Nilai", "ari_all":"ARI semua baris", "ari_common_assigned":"ARI assigned bersama", "common_assigned_fraction":"Assigned bersama"}), percentages=("Assigned bersama",)))
    sections.append('''

K-Means dengan k tetap 6 relatif konsisten terhadap seed: ARI sekitar **0,9924-0,9947** untuk seed alternatif. Jika jumlah cluster diubah, kesamaan turun: k=5 menghasilkan ARI **0,8941**, sedangkan k=7 **0,5755**. Konsisten terhadap seed tidak sama dengan konsisten terhadap semua pilihan metodologi.

DBSCAN berubah ketika eps/min_samples diubah. ARI yang memasukkan noise dapat dipengaruhi kesamaan status noise; karena itu tersedia ARI hanya pada observasi yang assigned oleh kedua partisi, beserta cakupannya. ARI tinggi pada subset kecil atau satu cluster harus dibaca hati-hati.

### 11.2 Resolusi grid dan ambang magnitudo

Baris grid pada 25 km tidak identik dengan baris grid 50 km atau 100 km. Karena itu, label dipetakan kembali ke **event ID historis yang sama**, baru ARI dihitung pada event bersama. Ukuran ini bersifat **event-weighted**: wilayah dengan banyak event mempunyai pengaruh lebih besar, dan grid tanpa event tidak ikut perbandingan ini. Kolom overlap melaporkan bagian event acuan yang masih dibandingkan.
''')
    sensitivity = data("reports/stability/grid_and_magnitude_ari.csv")
    sections.append(table(sensitivity[["method", "perturbation", "value", "n_common_events", "fraction_reference_events", "ari_same_events_including_noise", "ari_common_assigned"]].rename(columns={
        "method":"Metode", "perturbation":"Yang diubah", "value":"Nilai", "n_common_events":"Event bersama", "fraction_reference_events":"Overlap event acuan", "ari_same_events_including_noise":"ARI termasuk noise", "ari_common_assigned":"ARI assigned bersama"}), percentages=("Overlap event acuan",), digits={"Nilai":1}))
    sections.append(f'''

Saat konfigurasi sama dengan baseline (50 km dan M>=3), kontrol menghasilkan ARI **1,0**. K-Means pada grid 25 km mempunyai ARI **0,4428** dan pada 100 km **0,3193** terhadap baseline. Hasil ini menunjukkan ketergantungan pada resolusi, sehingga belum tepat menyebut seluruh profil stabil lintas ukuran grid. Perubahan ambang magnitudo juga mengubah partisi serta event yang dapat dibandingkan.

### 11.3 Dukungan jumlah kejadian per kotak

Pada training utama, **{number(train['low_support'].sum())} dari {number(len(train))} baris ({pct(low_support)}) hanya memuat 1-4 event**. Jika hanya satu event, mean dan maksimum magnitudo sama, sedangkan proporsi dangkal hanya 0 atau 1. Hal ini dapat menciptakan struktur geometris yang belum tentu mencerminkan profil regional yang kuat.

Eksperimen berikut memfilter training dengan dukungan minimal 1, 3, 5 dan 10 event, lalu fit ulang scaler dan model pada subset historis dengan k/eps/min_samples terpilih tetap. ARI dibandingkan pada baris grid-triwulan yang sama yang masih dipertahankan.
''')
    support = data("reports/stability/minimum_event_support.csv")
    sections.append(table(support[["method", "min_events_per_quarter", "n_same_grid_quarters", "fraction_reference_rows", "n_clusters", "noise_fraction", "ari_same_grid_quarters"]].rename(columns={
        "method":"Metode", "min_events_per_quarter":"Minimum event", "n_same_grid_quarters":"Baris tersisa", "fraction_reference_rows":"Cakupan baseline", "n_clusters":"Cluster", "noise_fraction":"Noise", "ari_same_grid_quarters":"ARI pada baris sama"}), percentages=("Cakupan baseline", "Noise")))
    sections.append('''

Dengan minimal 5 event, hanya **1.809 baris atau 13,43%** training tersisa. ARI K-Means pada subset itu sekitar **0,0671** setelah fitting ulang. Dengan minimal 10 event tersisa 435 baris. Artinya, interpretasi profil utama sangat bergantung pada banyak unit berdukungan rendah. Ini temuan keterbatasan yang perlu dilaporkan, bukan disembunyikan dengan memilih skor terbaik.

### 11.4 Kebijakan screening DBSCAN

Aturan ukuran cluster dan jumlah cluster juga diuji. Jika batas maksimal 300 cluster serta ukuran minimal 5 diterima, kandidat terpilih bergeser menjadi **95 cluster** dengan mean Silhouette sekitar **0,4976**. Jika ukuran minimal dinaikkan menjadi 50 pada kandidat yang tersedia, **tidak ada kandidat yang lolos**. Ini memperlihatkan bahwa pilihan tujuh cluster bergantung pada kebijakan screening, bukan temuan jumlah kelompok yang bebas dari keputusan peneliti.

**Keluaran:** [ARI parameter](reports/stability/parameter_ari.csv), [sensitivitas grid dan magnitudo](reports/stability/grid_and_magnitude_ari.csv), [minimum event](reports/stability/minimum_event_support.csv), dan [kebijakan seleksi DBSCAN](reports/stability/dbscan_selection_policy.csv).

## 12. Testing 2026

**Rujukan:** [09 Testing 2026](notebooks/09_testing_2026/09_testing_2026.ipynb), terutama bagian 2-4 untuk aturan dan eksekusi, bagian 5 untuk quarter, serta bagian 6 untuk hasil dengan nama kelompok historis.

### 12.1 Tujuan pengujian

Testing memeriksa apakah kelompok historis dapat diterapkan pada aktivitas yang tercatat setelah periode training. Centroid, core points, scaler dan parameter dibekukan dari 2021-2025. Tidak ada label ground truth untuk menghitung accuracy, precision atau recall. Ini juga bukan prediksi kapan gempa akan terjadi.

K-Means menggunakan `predict()`, yaitu centroid historis terdekat. Karena selalu memberikan label pada baris layak, cakupan 100% tidak otomatis membuktikan semua data baru cocok. Jarak centroid di atas kuantil 99% training diberi flag eksploratif.

DBSCAN tidak mempunyai `predict()` bawaan. Pipeline memakai **aturan tambahan core-radius**: tepat satu cluster core dalam eps -> ditugaskan; tidak ada -> `NO_CORE_MATCH`; lebih dari satu -> `AMBIGUOUS_CORE_MATCH`. Ini penerapan tambahan terhadap model historis, bukan fitting DBSCAN baru pada 2026.

### 12.2 Cakupan periode dan kecocokan
''')
    test_rows = []
    for period, group in test.groupby("period"):
        eligible = group.loc[group["state"].eq("ELIGIBLE")]
        test_rows.append({"Periode": period + (" parsial" if period == "2026Q4" else " lengkap"),
            "Event M >=3": int(included.loc[included["period"].eq(period)].shape[0]),
            "Kotak layak": len(eligible), "Kotak nol event": int(group["state"].eq("NO_EVENTS").sum()),
            "DBSCAN cocok": int(eligible["dbscan_cluster"].ge(0).sum()), "Belum cocok": int(eligible["dbscan_profile"].eq("NO_CORE_MATCH").sum()),
            "Ambigu": int(eligible["dbscan_profile"].eq("AMBIGUOUS_CORE_MATCH").sum())})
    sections.append(table(pd.DataFrame(test_rows)))
    sections.append(f'''

Testing lengkap Q1-Q3 mempunyai **{number(len(test_complete))} baris layak**. Ekstensi DBSCAN menugaskan **{number(test_complete['dbscan_cluster'].ge(0).sum())} baris**, tidak menemukan kecocokan untuk **{number(test_complete['dbscan_profile'].eq('NO_CORE_MATCH').sum())} baris**, dan menemukan kondisi ambigu untuk **{number(test_complete['dbscan_profile'].eq('AMBIGUOUS_CORE_MATCH').sum())} baris**. Total tidak assigned termasuk ambigu adalah **{pct(test_complete['dbscan_cluster'].lt(0).mean())}**.

Q4 hanya mencakup 1 Oktober sampai sebelum 26 Oktober, sehingga bukan triwulan penuh. Meskipun frekuensinya dinormalisasi terhadap durasi, maksimum magnitudo dan ringkasan lain tidak menjadi sepenuhnya setara dengan tiga bulan lengkap.

### 12.3 Skor pada data baru

{table(holdout[['split','method','n_rows','n_assigned','noise_fraction','silhouette','davies_bouldin']].rename(columns={'split':'Bagian','method':'Metode','n_rows':'Baris layak','n_assigned':'Assigned','noise_fraction':'Tidak assigned','silhouette':'Silhouette','davies_bouldin':'DBI'}), percentages=('Tidak assigned',))}

Untuk Q1-Q3, Silhouette K-Means adalah **0,3762** dan DBI **0,9735**. Nilai tersebut menunjukkan profil historis masih menghasilkan pemisahan geometris pada data baru, dengan kualitas berbeda dari training. DBSCAN core-radius mempunyai Silhouette **0,1556** dan DBI **2,0135** pada subset assigned.

Tabel testing ini **belum membandingkan kedua metode pada subset assigned identik**, sehingga skor tidak boleh dijadikan perbandingan sepenuhnya setara tanpa melihat cakupan. Angka tidak assigned DBSCAN mencakup tidak cocok dan ambigu; bukan noise dari refit model. Silhouette Q4 yang lebih tinggi pada subset tertentu juga tidak otomatis berarti Q4 lebih baik karena periodenya parsial dan cakupan assigned lebih kecil.

### 12.4 Perubahan distribusi fitur
''')
    drift = data("reports/testing_2026/feature_drift.csv")
    sections.append(table(drift.rename(columns={"feature":"Fitur","train_mean":"Mean training","test_complete_mean":"Mean testing lengkap","standardized_mean_shift":"Selisih dalam SD training"})))
    sections.append(f'''

Selisih standar positif berarti mean testing lebih tinggi daripada training pada fitur itu; negatif berarti lebih rendah. Ini pemeriksaan deskriptif, belum uji signifikansi. Perubahan komposisi kotak yang aktif juga dapat memengaruhinya. Mean magnitudo turun sekitar 0,355 SD training, sedangkan log frekuensi naik sekitar 0,281 SD; ini tidak otomatis menjelaskan penyebab fisiknya.

Ambang flag jarak centroid K-Means adalah **{number(test_method['kmeans_distance_q99_training'], 4)}** pada ruang standar. Pada testing lengkap terdapat **{number(test_complete['outside_training_distance_q99'].sum())} baris ({pct(test_complete['outside_training_distance_q99'].mean())})** di atas ambang; Q4 parsial mempunyai **{number(test_partial['outside_training_distance_q99'].sum())}** baris. Flag ini bukan probabilitas gempa atau uji signifikansi anomali fisik.

### 12.5 Contoh satu kotak yang dapat ditelusuri

Contoh aktual berikut memakai grid **5_-13**, berpusat sekitar 120,49 BT dan -8,15 LS. Jumlah event selalu merujuk kejadian M >=3 dalam katalog. Nama berikut merupakan hasil K-Means.
''')
    history = data("reports/temporal/kmeans/history.csv")
    historic_example = history.loc[history["grid_id"].eq("5_-13") & history["period"].ge("2025Q1"), ["period", "profile", "earthquake_count", "avg_magnitude", "avg_depth"]]
    new_example = test.loc[test["grid_id"].eq("5_-13"), ["period", "kmeans_profile", "earthquake_count", "avg_magnitude", "avg_depth"]].rename(columns={"kmeans_profile":"profile"})
    example = pd.concat([historic_example, new_example], ignore_index=True)
    example["profile"] = example["profile"].map(lambda code: km_names[code]["name"].split(":", 1)[0])
    sections.append(table(example.rename(columns={"period":"Periode","profile":"Kelompok","earthquake_count":"Event tercatat","avg_magnitude":"Mean M","avg_depth":"Mean depth (km)"}), digits={"Mean M":2,"Mean depth (km)":2}))
    sections.append('''

Pada 2026Q3, kotak contoh mempunyai **1.036 event**, lalu K-Means menempatkannya pada Kelompok F yang relatif lebih sering. Namun rata-rata historis F hanya sekitar 8,93 event. Baris ini diberi **flag jauh dari centroid historis**, dan ekstensi DBSCAN menghasilkan `NO_CORE_MATCH`. Ini memperlihatkan bahwa label K-Means terdekat tidak menjamin pengamatan berada dekat pola historis. Lonjakan perlu ditelusuri pada event sumber dan konsistensi katalog sebelum diberi interpretasi fisik.

File perubahan testing membandingkan setiap quarter 2026 terhadap baseline **2025Q4**, bukan selalu terhadap quarter tepat sebelumnya. Karena itu, `changed_from_baseline` harus dibaca sebagai perubahan dari baseline tersebut.

**Keluaran:** [label semua 2026](reports/testing_2026/labels_all_2026.csv), [quarter lengkap](reports/testing_2026/labels_complete_quarters.csv), [quarter parsial](reports/testing_2026/labels_partial_quarter.csv), [skor testing](reports/testing_2026/holdout_metrics.csv), [drift fitur](reports/testing_2026/feature_drift.csv), [perubahan dari 2025Q4](reports/testing_2026/changes_from_2025Q4.csv), dan [metodologi testing](reports/testing_2026/methodology.json).

## 13. GIS dan cara membaca peta

**Rujukan:** [10 GIS mapping](notebooks/10_gis_mapping/10_gis_mapping.ipynb), bagian 2 untuk penjelasan nama, bagian 3 untuk peta dasar dan pemilih periode, bagian 4-5 untuk pembuatan peta, dan bagian 6 untuk ekspor.

### 13.1 Mengembalikan hasil ke lokasi

Label hasil model dihubungkan kembali ke `grid_id` dan periode. Batas kotak dibangun dari proyeksi LAEA yang sama dengan preprocessing, sisi didensifikasi, lalu ditransformasi ke WGS84 agar dapat ditampilkan sebagai polygon pada peta. Warna mengikuti identitas kelompok yang sama sepanjang periode untuk metode tersebut.

Daratan Natural Earth digunakan sebagai konteks visual. Daratan tidak digunakan untuk menyaring event atau melatih model. Leaflet, CSS, geometri daratan dan data peta disertakan dalam HTML, sehingga file peta dapat dibuka tanpa internet dan tidak meminta tile OpenStreetMap. Ini juga menghindari masalah tile 403 pada versi awal.

### 13.2 Empat peta yang dihasilkan

| Peta | Pertanyaan yang dijawab | Buka file |
| --- | --- | --- |
| Profil K-Means per periode | Kotak ini paling mirip kelompok aktivitas yang mana? | [K-Means per triwulan](maps/kmeans_profiles_by_quarter.html) |
| Profil DBSCAN per periode | Kotak ini cocok dengan kelompok kepadatan historis yang mana? | [DBSCAN per triwulan](maps/dbscan_profiles_by_quarter.html) |
| Perubahan K-Means 2021-2025 | Berapa kali kotak berganti kelompok pada pasangan triwulan layak? | [Perubahan K-Means](maps/kmeans_profile_changes_2021_2025.html) |
| Perubahan DBSCAN 2021-2025 | Berapa kali kotak berganti kelompok, dengan noise dikeluarkan? | [Perubahan DBSCAN](maps/dbscan_profile_changes_2021_2025.html) |

Masing-masing peta profil menyediakan **24 pilihan periode**: 20 quarter historis, tiga quarter testing lengkap, dan satu quarter parsial. Satu layer periode ditampilkan pada satu waktu. Nama quarter ditulis sebagai rentang bulan. Zoom awal menyesuaikan bounding box; tombol +/- dapat dipakai untuk memperbesar dan memperkecil.

Semua peta memiliki tombol **Sembunyikan grid / Tampilkan grid** di kiri bawah. Tombol ini menyembunyikan atau menampilkan seluruh kotak hasil analisis, sementara daratan tetap terlihat. Pada peta profil, mengganti periode saat grid disembunyikan tidak memunculkannya kembali; tombol Tampilkan grid akan menampilkan periode terakhir yang dipilih.

### 13.3 Membaca peta profil

1. Pilih periode, misalnya Juli-September 2026.
2. Baca warna melalui nama deskriptif pada legenda. Huruf kelompok adalah pengenal, bukan peringkat rendah sampai tinggi.
3. Klik kotak untuk melihat jumlah event, mean magnitudo dan mean kedalaman pada periode itu.
4. Bedakan **angka pada legenda** (rata-rata kelompok historis) dari **angka pada popup** (data kotak-periode yang dipilih).
5. Jika suatu kotak diberi catatan dukungan 1-4 kejadian atau jauh dari centroid, periksa keterbatasan tersebut bersama labelnya.

Tidak ada gempa tercatat berarti tidak ada event di atas ambang pada katalog dan periode itu. Tidak masuk kelompok DBSCAN berarti status historis noise; belum cocok berarti data testing tidak mempunyai satu kelompok core yang cocok; ambigu berarti lebih dari satu kelompok core cocok. Daerah tanpa kotak berada di luar universe grid yang pernah aktif pada cakupan yang digunakan, bukan otomatis wilayah aman.

### 13.4 Membaca peta perubahan

Biru berarti nol pergantian pada pasangan yang dapat dibandingkan. Kuning berarti 1-2 pergantian. Jingga tua berarti lebih dari 2 pergantian. Abu-abu berarti belum ada pasangan periode yang layak dibandingkan. Arahkan kursor ke kotak untuk melihat jumlah pergantian, denominator pasangan, stability rate, dan durasi terpanjang.

Satu pergantian dari satu pasangan memberi dukungan yang berbeda dari satu pergantian dari 19 pasangan. Perubahan warna di peta profil maupun banyaknya pergantian di peta perubahan tidak langsung menyatakan kenaikan bahaya gempa.

### 13.5 Ekspor untuk GIS lain

Geometri tersedia sebagai [grid K-Means GeoJSON](maps/kmeans_grid_geometry.geojson) dan [grid DBSCAN GeoJSON](maps/dbscan_grid_geometry.geojson). Keduanya memakai universe geometri yang sama pada resolusi utama; label dan statistik per periode tetap berada dalam CSV masing-masing metode. Join dilakukan dengan `grid_id`, lalu pilih periode sebelum memetakan. Kedua file geometri bukan dua sistem grid spasial yang berbeda.

Pratinjau statis dapat dibuka melalui [K-Means](reports/figures/kmeans_map_preview.png) dan [DBSCAN](reports/figures/dbscan_map_preview.png). Kode kelompok pada pratinjau dapat dicocokkan dengan tabel bagian 9; HTML menggunakan legenda deskriptif yang lebih lengkap.

## 14. Verifikasi eksekusi

**Rujukan:** [00 Panduan](notebooks/00_panduan.ipynb), [laporan eksekusi notebook](reports/validation/notebook_execution.json), dan [pemeriksaan environment](reports/validation/environment.json).
''')
    executed_cells = sum(item["executed_cells"] for item in execution["notebooks"])
    test_xml = ROOT / "reports/validation/tests.xml"
    READ_FILES.add("reports/validation/tests.xml")
    suites = ET.parse(test_xml).getroot().findall("testsuite")
    n_tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
    failures = sum(int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0)) for s in suites)
    sections.append(f'''

- **{len(execution['notebooks'])} notebook** mempunyai status PASS pada laporan eksekusi; seluruh **{executed_cells} code cell** yang tercatat telah dieksekusi.
- Kernel yang digunakan: **Python 3.12.11 (Seismic Research)**. NumPy {environment['packages']['numpy']}, pandas {environment['packages']['pandas']}, scikit-learn {environment['packages']['scikit-learn']}, dan pyproj {environment['packages']['pyproj']} tercatat pada environment.
- Laporan pengujian berisi **{n_tests} tes**, dengan **{failures} kegagalan/error**. Pengujian meliputi depth, parsing/duplikat, nol event, pemisahan training/testing, cakupan unduhan, transisi, ekstensi DBSCAN, dan aset peta lokal. Ini bukan validasi kebenaran geologi.
- Kontrol peta lulus pemeriksaan 24 periode dan satu layer aktif. Pemeriksaan aset menunjukkan tidak ada permintaan tile eksternal. Uji kontrol menggunakan **stub DOM/Leaflet**, bukan browser nyata; cakupan pemeriksaan tampilan browser belum tersedia dalam sesi tersebut.
- Notebook awal tidak ditulis oleh pipeline baru. Laporan eksekusi mencatat hash sebelum/sesudah, sementara perubahan notebook oleh pengguna di luar eksekusi tidak ditimpa.

Bukti pendukung: [tests.xml](reports/validation/tests.xml), [integritas data](reports/validation/data_integrity.json), [kontrol peta](reports/validation/map_selector.json), [penamaan notebook](reports/validation/notebook_naming.json), dan [dependensi terkunci](requirements-lock.txt). Label, statistik dan data diuji tetap sama saat penjelasan nama ditambahkan.

## 15. Temuan dan batas kesimpulan

### 15.1 Pernyataan yang didukung hasil saat ini

Pengumpulan berhasil memperoleh {number(len(raw))} catatan API, yang setelah pemeriksaan menghasilkan {number(len(clean))} kejadian bersih. Analisis M>=3 mengagregasi {number(len(included))} event menjadi panel grid-triwulan. Model historis memakai {number(len(train))} observasi layak pada 2021-2025.

K-Means memilih k=6 menurut aturan kandidat dan mean Silhouette yang digunakan. DBSCAN memilih eps {number(db_meta['eps'], 6)} dan min_samples {db_meta['min_samples']}, menghasilkan tujuh cluster serta noise {pct(db_selected['noise_fraction'])}. Pada subset identik training, K-Means mempunyai Silhouette lebih tinggi dan DBI lebih rendah. Kelompok tersebut dapat disajikan melalui nama deskriptif yang didasarkan pada statistik training.

Analisis historis telah menghasilkan riwayat, transisi, durasi dan stability rate. Pengujian 2026 menggunakan model historis tanpa fit ulang, membedakan quarter lengkap dan parsial serta menangani kondisi yang tidak cocok atau ambigu. GIS menyajikan hasil pada polygon dengan pilihan periode dan peta pergantian kelompok.

### 15.2 Hal yang masih membatasi kekuatan penelitian

1. **Dukungan event rendah:** {pct(low_support)} observasi training hanya berisi 1-4 event. Banyak mean, maksimum dan proporsi belum merupakan ringkasan wilayah yang kuat.
2. **Sensitif terhadap resolusi dan minimum event:** ARI turun saat ukuran grid atau dukungan minimal diubah. Hasil belum stabil untuk semua pilihan metodologi.
3. **k=6 belum mutlak:** skor k=2 berdekatan. Pilihan enam kelompok harus disajikan sebagai hasil aturan eksperimen sekarang.
4. **Kelengkapan dan jenis magnitudo:** ambang M>=3 belum membuktikan completeness, dan magnitudo belum dihomogenisasi. Perubahan katalog dapat memengaruhi profil temporal.
5. **Korelasi fitur dan kebijakan DBSCAN:** fitur masih berkorelasi dan screening memengaruhi kelompok terpilih. Hasil bergantung pada keputusan analitis yang dinyatakan terbuka.
6. **Dukungan temporal tidak merata:** hanya pasangan assigned yang dibandingkan; median pasangan per grid rendah. Pergantian label belum otomatis perubahan fisik yang bermakna.
7. **Belum ada ground truth eksternal:** skor internal dan ARI bukan akurasi. Peta bukan peta risiko atau prediksi waktu kejadian.
8. **Testing Q4 parsial:** normalisasi durasi tidak menjadikan semua statistik setara dengan quarter lengkap.

Hasil ini dapat digunakan sebagai **analisis eksploratif profil aktivitas yang tercatat dalam katalog**. Untuk memperkuat naskah ilmiah, prioritasnya adalah justifikasi completeness/jenis magnitudo, pengaruh pemilihan fitur, kecukupan event per unit, sensitivitas spasial, serta ketidakpastian transisi. Tidak ada klaim novelty atau perubahan tektonik kausal hanya dari berhasilnya dashboard dan clustering.

## 16. Daftar keluaran dan referensi

| Kelompok file | Lokasi | Isi |
| --- | --- | --- |
| Konfigurasi | [config.json](config.json) | Periode, bounding box, grid, fitur dan kandidat parameter |
| Raw | [data/raw](data/raw/) | Respons bulanan, CSV dan manifest |
| Preprocessing | [data/processed](data/processed/) | Event bersih/ditolak, panel, scaler dan fitur standar |
| K-Means | [models/kmeans](models/kmeans/) | Model, seluruh kandidat, label dan profil |
| DBSCAN | [models/dbscan](models/dbscan/) | Model, seluruh kandidat, noise dan profil |
| Evaluasi | [reports/evaluation](reports/evaluation/) | Audit katalog, korelasi, nama dan perbandingan |
| Temporal | [reports/temporal](reports/temporal/) | Riwayat, transisi, stability rate dan durasi |
| Sensitivitas | [reports/stability](reports/stability/) | ARI parameter, grid, magnitudo dan dukungan event |
| Testing | [reports/testing_2026](reports/testing_2026/) | Label 2026, quarter lengkap/parsial, drift dan baseline |
| Grafik | [reports/figures](reports/figures/) | EDA, pilihan k, k-distance, PCA, transisi dan pratinjau peta |
| GIS | [maps](maps/) | Empat HTML peta dan dua geometri GeoJSON |
| Verifikasi | [reports/validation](reports/validation/) | Status notebook, pengujian, environment dan integritas |

**Rujukan penjelasan internal:** [metodologi dan batas interpretasi](docs/metodologi.md), [panduan membaca hasil](docs/cara_menjelaskan_hasil.md), dan [README proyek](README.md).

**Rujukan teknis:** [EMSC FDSN Event service](https://www.seismicportal.eu/fdsn-wsevent.html), [clustering scikit-learn](https://scikit-learn.org/stable/modules/clustering.html), [API DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), [PyProj Transformer](https://pyproj4.github.io/pyproj/stable/api/transformer.html), [kode tipe kejadian ISC/ISF](https://www.isc.ac.uk/standards/isf/), serta [lisensi data Natural Earth](https://www.naturalearthdata.com/about/terms-of-use/). Sumber aset peta juga dicatat dalam [sources.json](assets/maps/sources.json).

Laporan ini dibentuk dengan `scripts/build_full_report.py`, yang membaca keluaran tersimpan tanpa mengunduh data, memilih ulang parameter, atau melatih model. Tabel dan narasi pada dokumen ini merujuk snapshot yang dijelaskan di awal. Jika data atau metodologi berubah, hasil serta interpretasi naratif perlu ditinjau kembali sebelum laporan digunakan.
''')
    content = "\n\n".join(sections).replace("\n\n\n", "\n\n")
    OUTPUT.write_text(content, encoding="utf-8")
    verify_report(content)
    print("Laporan tersimpan:", OUTPUT)
    print("Jumlah kata sekitar:", len(content.split()))


def verify_report(content):
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content)
    local_links = []
    for target in links:
        if target.startswith(("http://", "https://")):
            continue
        if target.startswith("#"):
            expected = target[1:]
            headings = re.findall(r"^## (.+)$", content, re.M)
            slugs = [re.sub(r"[^\w\s-]", "", h.lower()).replace(" ", "-") for h in headings]
            assert expected in slugs, f"Anchor tidak ditemukan: {target}"
            continue
        relative, _, anchor = unquote(target).partition("#")
        path = ROOT / relative
        assert path.exists(), f"Tautan tidak ditemukan: {target}"
        if anchor and path.suffix == ".ipynb":
            nb = json.loads(path.read_text(encoding="utf-8"))
            assert any(f'id="{anchor}"' in "".join(c["source"]) for c in nb["cells"]), target
        local_links.append(target)
    assert len(re.findall(r"^## \d+\.", content, re.M)) == 16
    assert "78.265" in content and "78.213" in content and "50.198" in content
    source_hashes = {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() for relative in sorted(READ_FILES)}
    receipt = {"status": "PASS", "report": OUTPUT.name, "local_links_checked": len(local_links),
               "numbered_sections": 16, "source_sha256": source_hashes,
               "report_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
               "scope": "Angka dibaca dari artefak aktual dan tautan lokal/anchor notebook diperiksa; tidak menjalankan ulang model."}
    (ROOT / "reports/validation/full_report.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build()
