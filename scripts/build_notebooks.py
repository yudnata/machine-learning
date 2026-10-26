"""Buat notebook dengan kode tahap terlihat; file sumber hanya membantu pemeliharaan."""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import nbformat as nbf
import notebook_eda_content as eda_content
import notebook_dbscan_simple as dbscan_content
import notebook_evaluation_content as evaluation_content

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = '''from pathlib import Path
import os
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
candidates = []
for parent in [Path.cwd(), *Path.cwd().parents]:
    candidates.extend([parent, parent / "seismic_research_2021_2025"])
ROOT = next((p for p in candidates if (p / "config.json").is_file() and (p / "src/common.py").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Buka notebook dari folder proyek seismic_research_2021_2025.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from IPython.display import display, Image, Markdown, FileLink
from src.common import read_config
cfg = read_config(ROOT)
print("Interpreter:", sys.executable)
print("Python:", sys.version.split()[0])
print("Proyek:", ROOT)
print("Training:", cfg["train_start"], "sampai sebelum", cfg["train_end_exclusive"])
print("Testing sampai sebelum (UTC):", cfg["test_end_exclusive"])
'''


def source(module, names):
    text = (ROOT / f"src/{module}.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    selected = {node.name: ast.get_source_segment(text, node) for node in tree.body if isinstance(node,(ast.FunctionDef, ast.AsyncFunctionDef))}
    return "\n\n".join(selected[name] for name in names)


def imports(module):
    text = (ROOT / f"src/{module}.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    items = [ast.get_source_segment(text,node) for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom)) and not (isinstance(node,ast.ImportFrom) and node.module=="__future__")]
    return "\n".join(items).replace("from .", "from src.")


def section(title, explanation="", code=None):
    return (title, explanation, code)


COLLECTION_RESULTS = '''from contextlib import redirect_stdout
from io import StringIO

collection_log = StringIO()
with redirect_stdout(collection_log):
    raw, coverage = collect(ROOT)
jumlah = lambda nilai: f"{int(nilai):,}".replace(",", ".")
waktu = pd.to_datetime(raw["time"], utc=True, format="mixed", errors="coerce")
historis = waktu.lt(pd.Timestamp(cfg["train_end_exclusive"], tz="UTC"))
display(Markdown(f"""### Hasil pengambilan data

**Berhasil mengambil {jumlah(len(raw))} catatan dari katalog EMSC.**

| Periode | Jumlah catatan mentah |
| --- | ---: |
| 2021-2025, calon data training | {jumlah(historis.sum())} |
| 2026, calon data testing | {jumlah((~historis & waktu.notna()).sum())} |
| Total | {jumlah(len(raw))} |

Seluruh **{jumlah(len(coverage))} interval bulanan** berhasil diambil.
Batas akhir pengambilan: sebelum **{cfg['test_end_exclusive']} pukul 00.00 UTC**.
Tahun 2026 belum satu tahun penuh.

Jumlah ini masih **catatan mentah**, belum jumlah gempa final untuk analisis.
Jenis kejadian, duplikat, dan batas magnitudo diperiksa pada notebook 02.
Satu baris tabel berikut mewakili satu catatan kejadian, bukan satu kotak peta.
"""))
display(raw[["eventid", "time", "latitude", "longitude", "magnitude", "depth", "event_type", "region"]].head(10).rename(columns={
    "eventid": "ID kejadian", "time": "Waktu UTC", "latitude": "Lintang", "longitude": "Bujur",
    "magnitude": "Magnitudo", "depth": "Kedalaman (km)", "event_type": "Kode jenis", "region": "Wilayah"}))
display(FileLink(str(ROOT / "data/raw/events.csv"), result_html_prefix="Seluruh data mentah: "))
'''

COLLECTION_ANNUAL = '''annual_raw = raw.assign(tahun=waktu.dt.year).groupby("tahun").size().rename("Catatan mentah").reset_index()
annual_raw["Peran"] = annual_raw["tahun"].map(lambda tahun: "Training" if tahun < 2026 else "Testing (tahun belum lengkap)")
display(annual_raw.rename(columns={"tahun": "Tahun"}))
display(Markdown("### Jenis catatan dari sumber"))
display(raw["event_type"].fillna("Tidak tersedia").value_counts().rename_axis("Kode jenis sumber").reset_index(name="Jumlah catatan"))
display(Markdown("Kode jenis berasal dari EMSC. Hanya jenis yang diizinkan pada konfigurasi yang diteruskan oleh preprocessing; jumlah mentah tidak langsung dianggap jumlah gempa yang dipakai model."))
display(Markdown("### Rincian pengambilan per bulan"))
display(coverage[["start", "end_exclusive", "events_returned", "complete"]].rename(columns={
    "start": "Mulai UTC", "end_exclusive": "Sebelum UTC", "events_returned": "Catatan diambil", "complete": "Lengkap"}))
assert annual_raw["Catatan mentah"].sum() == len(raw)
assert coverage["events_returned"].sum() == len(raw)
'''

PREPROCESSING_RESULTS = '''panel, events, metadata = preprocess(ROOT)
jumlah = lambda nilai: f"{int(nilai):,}".replace(",", ".")
included = pd.read_csv(ROOT / "data/processed/events_with_grid.csv")
rejected = pd.read_csv(ROOT / "data/processed/events_rejected.csv")
below_threshold = len(events) - len(included)
display(Markdown(f"""### Hasil pembersihan dan pemilihan data

**{jumlah(len(included))} kejadian gempa memenuhi batas magnitudo minimal {cfg['min_magnitude']:g} dan masuk ke pembentukan grid.**

| Tahap | Jumlah catatan kejadian |
| --- | ---: |
| Masuk dari collection | {jumlah(metadata['n_raw'])} |
| Tidak diteruskan saat cleaning | {jumlah(len(rejected))} |
| Tersisa sesudah cleaning | {jumlah(len(events))} |
| Tidak memenuhi batas magnitudo (termasuk M hilang, jika ada) | {jumlah(below_threshold)} |
| Memenuhi batas magnitudo dan masuk pembentukan grid | {jumlah(len(included))} |

Data yang berada di bawah batas magnitudo tetap ada dalam `events_clean.csv`.
Untuk penelitian utama, `events_with_grid.csv` hanya menyimpan kejadian yang memenuhi ambang tersebut.
**Satu baris kedua file ini masih berarti satu kejadian gempa.**
"""))
assert metadata["n_raw"] == len(events) + len(rejected)
assert len(events) == below_threshold + len(included)
display(Markdown("### Jumlah gempa per tahun sebelum dan sesudah seleksi"))
raw_before = pd.read_csv(ROOT / "data/raw/events.csv")
annual = pd.concat([
    raw_before.assign(year=pd.to_datetime(raw_before["time"], utc=True, format="mixed", errors="coerce").dt.year).groupby("year").size().rename("Dari collection"),
    events.groupby("year").size().rename("Setelah cleaning"),
    included.groupby("year").size().rename("Lolos batas magnitudo")], axis=1).fillna(0).astype(int).rename_axis("Tahun").reset_index()
display(annual)
display(Markdown("### Contoh 10 kejadian yang benar-benar masuk pembentukan grid"))
display(included[["eventid", "time", "latitude", "longitude", "magnitude", "depth", "grid_id", "period"]].head(10).rename(columns={
    "eventid": "ID kejadian", "time": "Waktu UTC", "latitude": "Lintang", "longitude": "Bujur", "magnitude": "Magnitudo",
    "depth": "Kedalaman (km)", "grid_id": "ID kotak", "period": "Triwulan"}))
display(Markdown("### Mengapa ada catatan yang tidak diteruskan?"))
if rejected.empty:
    display(Markdown("Tidak ada catatan yang ditolak saat cleaning."))
else:
    reasons = {"event_type_not_accepted": "Jenis kejadian tidak termasuk jenis gempa yang dipakai",
               "invalid_time_coordinate_or_outside_scope": "Waktu/koordinat tidak valid atau di luar cakupan",
               "duplicate_older_revision": "Duplikat atau revisi lama"}
    display(rejected["reason"].map(lambda value: reasons.get(value, value)).value_counts().rename_axis("Alasan").reset_index(name="Jumlah catatan"))
'''

PREPROCESSING_PANEL = '''display(Markdown("""### Data kejadian diubah menjadi ringkasan kotak per triwulan

Misalnya, 10 gempa dalam kotak yang sama selama Januari-Maret menjadi **satu baris ringkasan**, dengan jumlah kejadian 10.
Karena itu jumlah baris panel berbeda dari jumlah gempa.

Kotak tanpa kejadian pada suatu triwulan tetap disimpan agar riwayat waktu tidak terputus.
Namun, baris nol kejadian tidak dimasukkan ke model clustering.
"""))
split_names = {"train": "Training 2021-2025", "test": "Testing 2026 (triwulan lengkap)", "test_partial": "Testing 2026 (triwulan belum lengkap)"}
state_names = {"ELIGIBLE": "Ada kejadian dan fitur layak untuk model", "NO_EVENTS": "Nol kejadian pada triwulan ini", "INCOMPLETE_FEATURES": "Ada kejadian, fitur belum lengkap"}
panel_summary = panel.groupby(["split", "state"]).agg(jumlah_baris=("row_id", "size"), jumlah_gempa=("earthquake_count", "sum")).reset_index()
panel_summary["split"] = panel_summary["split"].map(split_names)
panel_summary["state"] = panel_summary["state"].map(state_names)
display(panel_summary.rename(columns={"split": "Bagian data", "state": "Keadaan kotak", "jumlah_baris": "Baris kotak-triwulan", "jumlah_gempa": "Jumlah gempa di dalamnya"}))
assert panel["earthquake_count"].sum() == len(included)
display(Markdown(f"Total panel: **{jumlah(len(panel))} baris kotak-triwulan**, merangkum **{jumlah(len(included))} kejadian gempa**."))
display(Markdown("### Contoh data training yang masuk model (sebelum standardisasi)"))
ready = panel.loc[panel["split"].eq("train") & panel["state"].eq("ELIGIBLE")]
display(ready[["row_id", "earthquake_count", "frequency_per_month", "avg_magnitude", "max_magnitude", "avg_depth", "shallow_fraction"]].head(10).rename(columns={
    "row_id": "Kotak | triwulan", "earthquake_count": "Jumlah gempa", "frequency_per_month": "Gempa per bulan", "avg_magnitude": "Rata-rata M",
    "max_magnitude": "M terbesar", "avg_depth": "Rata-rata kedalaman (km)", "shallow_fraction": "Proporsi dangkal (0-1)"}).round(3))
display(Markdown("### Contoh hasil standardisasi untuk baris yang sama"))
scaled = pd.read_csv(ROOT / "data/processed/features_scaled.csv")
display(scaled.set_index("row_id").loc[ready["row_id"].head(10)].round(3))
display(Markdown("Lima kolom angka di atas dipakai oleh K-Means dan DBSCAN. Nilainya sudah disetarakan skalanya berdasarkan training; angka negatif berarti di bawah rata-rata training, bukan magnitudo atau kedalaman negatif."))
display(Markdown("### Contoh kotak tanpa kejadian"))
display(panel.loc[panel["state"].eq("NO_EVENTS"), ["grid_id", "period", "earthquake_count", "avg_magnitude", "avg_depth", "state"]].head(5))
display(Markdown("`NaN` pada M dan kedalaman berarti tidak ada nilai yang dapat dihitung karena jumlah gempa nol. Angka ini tidak diisi nol dan baris tersebut tidak masuk clustering."))
display(Markdown("### Buka seluruh datanya"))
for filename, label in [
    ("events_clean.csv", "Kejadian setelah cleaning"), ("events_rejected.csv", "Catatan ditolak beserta alasan"),
    ("events_with_grid.csv", "Kejadian lolos batas magnitudo dan ID kotaknya"), ("panel.csv", "Seluruh ringkasan kotak-triwulan"),
    ("features_scaled.csv", "Lima fitur yang sudah disetarakan skalanya")]:
    display(FileLink(str(ROOT / "data/processed" / filename), result_html_prefix=label + ": "))
'''


SPECS = [
    ("00_panduan.ipynb", "Panduan dan pemeriksaan environment", "Mulai di sini. Unit penelitian adalah grid-triwulan. Model dan skala fitur dipelajari hanya dari 2021-2025; 2026 merupakan temporal holdout.", None, [
        section("Periksa seluruh pustaka dan interpreter", "Kernel yang disiapkan: **Python 3.12.11 (Seismic Research)**. Pilih nama ini melalui Select Kernel di VS Code. Jika memakai mesin lain, jalankan `setup_kernel.ps1` terlebih dahulu.",
                'import runpy\nenvironment = runpy.run_path(str(ROOT / "scripts/verify_environment.py"))["verify"]()\nassert environment["imports"] == "PASS"'),
        section("Rancangan dan urutan eksekusi", "Jalankan notebook 01 sampai 10 secara berurutan. Setiap notebook dapat dibuka sendiri dan membaca artefak tahap sebelumnya dari disk. Nomor folder memudahkan pengurutan di Explorer; heading bernomor tampil di Outline/Table of Contents.",
                'import json\ndisplay(cfg)\ndisplay(Markdown((ROOT / "docs/metodologi.md").read_text(encoding="utf-8")))'),
        section("Lokasi input dan keluaran", "`data/raw`: JSON asli EMSC dan manifest. `data/processed`: panel dan fitur. `models/kmeans` serta `models/dbscan`: model terpisah. `reports`: tabel, grafik, transisi, sensitivitas, testing. `maps`: peta polygon. Notebook asli di folder induk dipertahankan.",
                'display(Markdown((ROOT / "README.md").read_text(encoding="utf-8")))'),
    ]),
    ("01_data_collection/01_data_collection.ipynb", "Data collection EMSC", "Input: API EMSC. Output: JSON bulanan, events.csv, dan collection_manifest.json. Rentang utama 2021-2025 ditambah testing 2026 hingga batas tanggal eksplisit.", "collection", [
        section("Definisi atribut dan kedalaman", "Kedalaman berasal dari **properties.depth (km)**. geometry.coordinates[2] disimpan sebagai geometry_z untuk audit; tidak dijadikan kedalaman. Katalog mencakup bounding box Indonesia dan sekitarnya, bukan batas administratif negara.", source("collection",["parse_feature"])),
        section("Pengunduhan bulanan, retry, cache, dan batas API", "Setiap bulan disimpan agar pengulangan tidak mengunduh ulang. Jika respons mencapai limit, interval dipecah hingga tidak terpotong. Kegagalan menghentikan proses; tidak dianggap nol kejadian. Endpoint menerima waktu UTC lengkap. Data disaring lagi secara lokal pada preprocessing.", source("collection",["collect"])),
        section("Jalankan collection dan lihat jumlah data", "Output sel ini menampilkan total catatan, pembagian training/testing, dan 10 contoh data. Unduhan baru membutuhkan internet; cache yang lengkap digunakan kembali.", COLLECTION_RESULTS),
        section("Audit depth dan cakupan", "Pastikan unduhan mencakup semua bulan dan bandingkan depth dengan geometry_z. Jangan mengubah tanda secara otomatis tanpa definisi sumber.", 'display(raw[["eventid","depth","geometry_z","depth_source"]].head(10))\nassert coverage["complete"].all()\nassert raw["depth_source"].eq("properties.depth").all()\nprint("Manifest:", ROOT / "data/raw/collection_manifest.json")'),
        section("Jumlah per tahun, jenis catatan, dan rincian bulanan", "Tabel ini menjelaskan dari tahun mana data diperoleh. Jumlah per tahun di sini masih sebelum cleaning dan seleksi magnitudo.", COLLECTION_ANNUAL),
    ]),
    ("02_data_preprocessing/02_data_preprocessing.ipynb", "Data preprocessing dan panel grid-triwulan", "Input: data mentah beserta manifest. Output: data bersih, panel train/test/test_partial, fitur scaled, dan scaler yang fit hanya pada training.", "preprocessing", [
        section("Validasi cakupan collection", "Nol kejadian hanya dapat dibentuk setelah semua interval API berhasil. Manifest diperiksa terhadap rentang tanggal, kesinambungan interval, dan hash CSV.", source("preprocessing",["validate_collection"])),
        section("Cleaning dan audit data", "UTC dan format waktu campuran diparsing eksplisit. Duplikat ID menyimpan revisi terbaru; ID kosong memakai kunci atribut. Depth di luar 0-800 km ditandai hilang. Magnitudo/depth hilang tidak otomatis diisi nol.", source("preprocessing",["clean_events"])),
        section("Proyeksi dan panel lengkap per triwulan", "Grid utama 50 km memakai LAEA berpusat di Indonesia, dengan origin tetap. Ini 50 km pada bidang proyeksi, bukan jaminan panjang sisi geodesik tepat 50 km. Universe training hanya grid aktif historis. Semua 20 triwulan disertakan. Testing menambahkan grid baru tanpa mengubah training. State NO_EVENTS terpisah dari INCOMPLETE_FEATURES; depth coverage minimal 80%.", source("preprocessing",["grid_panel"])),
        section("Standardisasi training dan penyimpanan", "Fitur: log1p(frekuensi/bulan), rata-rata M, maksimum M, rata-rata depth, dan proporsi depth <70 km dari event berkedalaman valid. Koordinat, count, density dan frekuensi mentah tidak digandakan sebagai fitur. StandardScaler fit pada baris training ELIGIBLE saja; test hanya transform. Tidak ada target supervised.", source("preprocessing",["preprocess"])),
        section("Jalankan dan lihat data hasil preprocessing", "Lihat jumlah sebelum/sesudah cleaning, jumlah yang memenuhi batas magnitudo, dan contoh kejadian yang digunakan. M >=3.0 adalah ambang eksploratif, belum klaim katalog lengkap. Analisis sensitivitas ambang ada pada notebook 08.", PREPROCESSING_RESULTS),
        section("Pemeriksaan kebocoran dan nol kejadian", "Statistik M/depth untuk nol event tetap NaN. Setiap grid historis memiliki 20 periode. Tahun 2026 tidak ikut fit scaler.", 'historical = panel.loc[panel["split"].eq("train")]\nassert historical.groupby("grid_id")["period"].nunique().eq(20).all()\nempty = panel.loc[panel["state"].eq("NO_EVENTS")]\nassert empty[["avg_magnitude","max_magnitude","avg_depth"]].isna().all().all()\nassert set(historical["year"]) == set(range(2021,2026))\nprint("Pemeriksaan panel lulus.")'),
        section("Lihat data kotak-triwulan dan fitur yang siap dimodelkan", "Bagian ini memperlihatkan perbedaan data kejadian gempa, ringkasan kotak-triwulan, dan fitur yang dipakai model. Seluruh CSV dapat dibuka melalui tautan di bawah tabel.", PREPROCESSING_PANEL),
    ]),
    ("03_eda/03_eda.ipynb", "EDA: mengenali data sebelum pengelompokan", "Di sini kita menjawab: berapa gempa yang tercatat, seperti apa magnitudo dan kedalamannya, serta apa keterbatasan datanya? Setiap hasil dilengkapi cara membaca dan maknanya. Hanya data 2021-2025 digunakan; 2026 disimpan untuk pengujian.", "analysis", [
        section("Kode pemeriksaan dan penyimpanan tabel", "Fungsi berikut menyiapkan tabel pemeriksaan dari hasil preprocessing. Hasil yang mudah dibaca mulai bagian 3. EDA tidak melatih model atau menentukan nama kelompok.", source("analysis",["eda"])),
        section("Jalankan EDA: jumlah data dan perbandingan tahun", "Mulai membaca hasil di sini. Tabel membedakan data setelah cleaning, data yang lolos batas magnitudo, dan baris kotak-triwulan untuk model.", eda_content.RESULTS),
        section("Apakah lima ciri aktivitas saling berkaitan?", "Satu baris di sini adalah satu kotak-triwulan training. Lima cirinya: frekuensi gempa yang dipadatkan dengan log1p, rata-rata magnitudo, magnitudo terbesar, rata-rata kedalaman, serta proporsi kejadian dengan kedalaman kurang dari 70 km. Proporsi 0,8 berarti 80% kejadian berkedalaman valid termasuk dangkal. Tabel korelasi memperlihatkan hubungan antarciri, bukan akurasi model.", eda_content.CORRELATION),
        section("Seperti apa magnitudo, kedalaman, dan jumlah per triwulan?", "Grafik bagian ini hanya memakai kejadian yang memenuhi batas magnitudo utama. Baca keterangan setelah grafik untuk memahami sumbu dan angka ringkasannya.", eda_content.DISTRIBUTIONS),
        section("Apa pengaruh batas magnitudo terhadap jumlah data?", "Batas minimal menentukan kejadian mana yang disertakan. Di sini kita membandingkan jumlah data, belum membandingkan hasil kelompok.", eda_content.THRESHOLDS),
        section("Apakah setiap kotak memiliki cukup banyak kejadian?", "Jumlah gempa berbeda dari jumlah baris model. Bagian ini memeriksa berapa kejadian yang mendasari ringkasan satu kotak selama tiga bulan.", eda_content.SUPPORT),
        section("Apakah data kosong dan pencatatan sudah diperiksa?", "Kita memeriksa nilai yang hilang, perubahan puncak histogram, dan jenis magnitudo dari sumber. Pemeriksaan ini membantu mengenali keterbatasan; tidak membuktikan katalog mencatat seluruh gempa yang terjadi.", eda_content.QUALITY),
        section("Kesimpulan EDA dan langkah berikutnya", "Ringkasan ini menghubungkan isi data dengan kehati-hatian yang diperlukan saat membaca hasil K-Means dan DBSCAN.", eda_content.SUMMARY),
    ]),
    ("04_kmeans/04_kmeans.ipynb", "K-Means: pencarian k dan profiling", "Satu model pada gabungan 20 triwulan historis menjaga makna label antarperiode. Label profil tidak diurutkan menjadi rendah/sedang/tinggi.", "modeling", [
        section("Definisi evaluasi clustering", "Silhouette dirata-rata dari tiga sampel acak, masing-masing maksimal 2.000 baris, untuk pemilihan parameter; SD sampel disimpan. DBI dihitung pada seluruh baris yang ditugaskan. Ini metrik internal, bukan akurasi.", source("modeling",["cluster_metrics"])),
        section("Label, profil asli, dan proyeksi visual", "Centroid dipelajari di ruang fitur standar; profiling ditampilkan dalam satuan asli. PCA 2D hanya membantu melihat hasil, bukan ruang pelatihan atau peta geografis.", source("modeling",["full_labels","profile_table","save_cluster_plot"])),
        section("Pencarian k dan fitting historis", "Uji k=2..8 dengan n_init=20. Pilih rata-rata Silhouette tertinggi, lalu DBI terendah jika sama. Elbow dan ukuran cluster tetap diperiksa. Selisih kecil antarkandidat bukan bukti keunggulan statistik. Tidak memaksa k=3.", source("modeling",["fit_kmeans"])),
        section("Jalankan K-Means", code='search, labels = fit_kmeans(ROOT)\ndisplay(search)\ndisplay(pd.read_csv(ROOT / "models/kmeans/profiles.csv"))'),
        section("Penamaan dan interpretasi kelompok (setelah clustering)", "Di bagian inilah nama ramah pembaca diberikan. Model sudah selesai membentuk kelompok pada bagian sebelumnya. Fungsi berikut membaca statistik training lalu menambahkan penjelasan: cenderung dangkal jika mean kedalaman <70 km; lebih sering untuk kelompok dengan mean jumlah kejadian terbesar. Huruf A/B/C adalah pengenal, bukan urutan bahaya. Aturan penulisan nama ini tidak menentukan anggota cluster, tidak melatih model lagi, dan tidak menggunakan testing 2026. Kode asli tetap disimpan untuk audit.", 'from src.mapping import profile_color\n\n' + source("mapping",["profile_descriptions","profile_interpretation_table"])),
        section("Tabel nama kelompok dan angka dasarnya", "Baca satu baris: kode keluaran algoritma -> nama deskriptif -> statistik yang mendasari nama. Angka adalah rata-rata grid-triwulan pada 2021-2025; data suatu kotak pada 2026 dapat berbeda. Tabel yang sama dipakai dalam evaluasi dan peta.", 'interpretation = profile_interpretation_table(ROOT, "kmeans", cfg)\nwith pd.option_context("display.max_colwidth", None):\n    display(interpretation)\ninterpretation.to_csv(ROOT / "reports/evaluation/kmeans_profile_interpretation.csv", index=False)\nprint("Penamaan selesai setelah clustering; kode serta anggota kelompok tetap dipertahankan.")'),
        section("Lihat elbow, evaluasi, dan cluster", code='display(Image(filename=str(ROOT / "reports/figures/kmeans_selection.png")))\ndisplay(Image(filename=str(ROOT / "reports/figures/kmeans_pca.png")))\ndisplay(labels["profile"].value_counts().rename("n_grid_quarters"))'),
    ]),
    ("05_dbscan/05_dbscan.ipynb", "DBSCAN: memilih eps dan membaca hasil", "Ikuti tiga langkah utama seperti saat membaca grafik K-Means: lihat kurva jarak, bandingkan kandidat parameter, lalu baca hasil kelompok. Data yang dipakai tetap training 2021-2025; lokasi tidak menjadi fitur kemiripan.", "modeling", [
        section("Kode perhitungan kelompok dan skor", "Kode berikut mendukung proses pengelompokan. Untuk membaca hasil, mulai bagian 4 lalu ikuti grafik bagian 7-9. Satu baris data adalah satu kotak selama tiga bulan.", source("modeling",["cluster_metrics","full_labels","profile_table","save_cluster_plot"])),
        section("Kode pencarian parameter DBSCAN", "DBSCAN mencoba eps dan min_samples, kemudian menghasilkan jumlah kelompok. Seluruh percobaan disimpan. Penjelasan grafik ada di bagian 7-9; aturan pemeriksaan lengkap tersedia di bagian 10.", source("modeling",["fit_dbscan"])),
        section("Jalankan atau baca hasil DBSCAN", "Jika hasil tersimpan tersedia dan sesuai dengan preprocessing, gunakan hasil tersebut. Ubah LATIH_ULANG = True untuk mengulang seluruh pencarian; jika hasil belum ada, pelatihan dijalankan otomatis. Data atau konfigurasi yang berubah harus diproses ulang sesuai urutan notebook.", dbscan_content.RUN),
        section("Penamaan dan interpretasi kelompok (setelah clustering)", "DBSCAN sudah membentuk kelompok pada bagian sebelumnya. Di sini kita membaca cirinya lalu memberi nama deskriptif. Nama tidak menentukan anggota kelompok atau mengubah noise. Aturan nama sama dengan K-Means, tetapi Kelompok A DBSCAN tidak sama dengan Kelompok A K-Means. Statistik pembentuk nama hanya memakai training 2021-2025; tidak memakai testing 2026.", 'from src.mapping import profile_color\n\n' + source("mapping",["profile_descriptions","profile_interpretation_table"])),
        section("Tabel nama kelompok dan angka dasarnya", "Nama kelompok, jumlah kejadian, magnitudo, dan kedalaman ditampilkan berdampingan agar interpretasinya dapat diperiksa. Baris tidak masuk kelompok tetap dipisahkan sebagai status DBSCAN.", 'interpretation = profile_interpretation_table(ROOT, "dbscan", cfg)\nwith pd.option_context("display.max_colwidth", None):\n    display(interpretation)\ninterpretation.to_csv(ROOT / "reports/evaluation/dbscan_profile_interpretation.csv", index=False)\nprint("Nama hanya menjelaskan hasil; keanggotaan dan status DBSCAN tidak diubah.")'),
        section("Langkah 1: lihat kurva k-distance", "Satu kurva menunjukkan jarak antarpola. Garis oranye menandai eps model terpilih. Kurva ini membantu memahami eps; tidak dipakai untuk menentukan jumlah kelompok seperti elbow K-Means.", dbscan_content.DISTANCE),
        section("Langkah 2: bandingkan eps dengan grafik skor", "Grafik garis ini menampilkan kandidat eps pada min_samples terpilih. Kiri: Silhouette lebih tinggi lebih baik. Kanan: Davies-Bouldin lebih rendah lebih baik. Bintang oranye menandai pilihan model.", dbscan_content.EPS_SCORES),
        section("Langkah 3: baca hasil kelompok dan noise", "Setelah memilih parameter, lihat berapa kelompok terbentuk dan berapa baris tidak tergabung. Ini hasil model, bukan kategori risiko.", dbscan_content.RESULT),
        section("Lampiran: seluruh percobaan dan rincian skor", "Bagian ini untuk pemeriksaan lebih lanjut. Tabel 12 percobaan dapat dibuka dengan mengeklik judul tabel di bawah. Alur utama cukup dibaca pada bagian 7-9.", dbscan_content.DETAILS),
        section("Ringkasan alur DBSCAN", "Hubungkan pemilihan parameter dengan hasil yang akan dibandingkan pada notebook evaluasi model.", dbscan_content.CONCLUSION),
    ]),
    ("06_model_evaluation/06_model_evaluation.ipynb", "Evaluasi dan perbandingan model", "Bandingkan struktur cluster, proporsi data ditugaskan, ukuran kelompok, interpretasi profil, dan kestabilan. Tidak memilih model hanya dari satu skor.", "analysis", [
        section("Perbandingan seluruh data dan subset identik", "Baris evaluasi pertama memakai seluruh training layak; DBSCAN menghitung metrik jarak tanpa noise. Baris kedua membatasi kedua metode pada subset training assigned DBSCAN yang sama. Silhouette tabel evaluasi memakai sampel tetap seed=42; tabel seleksi memakai rata-rata tiga seed. ARI adalah kesepakatan partisi, bukan akurasi.", source("analysis",["evaluate"])),
        section("Jalankan evaluasi dan baca ringkasan perbandingan", "Mulai membaca hasil di sini. Skor utama memakai baris yang sama untuk kedua model. Noise dan cakupan pada seluruh training dibaca terpisah agar tidak tertukar dengan akurasi.", evaluation_content.SUMMARY),
        section("Bandingkan karakteristik profil", "Nama deskriptif disusun setelah clustering pada notebook 04 dan 05. Tabel di sini memakai aturan nama yang sama. Kelompok A K-Means dan Kelompok A DBSCAN adalah kelompok berbeda; angka/huruf yang sama tidak menyamakan keduanya.", 'from src.mapping import profile_interpretation_table\nfor method in ["kmeans","dbscan"]:\n    display(Markdown(f"### Kelompok {method.upper()}"))\n    with pd.option_context("display.max_colwidth", None):\n        display(profile_interpretation_table(ROOT, method, cfg))'),
        section("Grafik perbandingan Silhouette dan Davies-Bouldin", "Grafik ini memakai subset identik, yaitu baris training yang ditugaskan DBSCAN. Silhouette dan DBI ditampilkan pada sumbu terpisah karena arah penilaiannya berbeda.", evaluation_content.SCORES),
        section("Grafik cakupan: masuk kelompok dan noise", "Seluruh training layak menjadi penyebut di sini. Grafik menjawab berapa banyak data yang ditugaskan, bukan berapa banyak prediksi yang benar.", evaluation_content.COVERAGE),
        section("Grafik jumlah anggota setiap kelompok", "Bandingkan ukuran kelompok dan dukungan datanya. Huruf yang sama pada kedua metode tidak berarti kelompok yang sama.", evaluation_content.SIZES),
        section("Grafik ciri kelompok dalam satuan asli", "Grafik menampilkan rata-rata jumlah kejadian, magnitudo, dan kedalaman agar ciri kelompok mudah dibaca. Grafik pada kolom yang sama menggunakan skala sumbu yang sama.", evaluation_content.PROFILES),
        section("Grafik kesepakatan pengelompokan dan ARI", "Periksa bagaimana baris yang sama dibagi oleh dua metode. Tidak ada label jawaban benar, sehingga grafik ini tidak digunakan untuk menghitung akurasi.", evaluation_content.AGREEMENT),
        section("Kesimpulan evaluasi, batas hasil, dan langkah berikutnya", "Baca kesimpulan bersama cakupan data, ciri kelompok, dan uji sensitivitas. Satu skor tidak cukup untuk membuktikan kebenaran geologis suatu pengelompokan.", evaluation_content.CONCLUSION),
    ]),
    ("07_temporal_analysis/07_temporal_analysis.ipynb", "Transisi dan kestabilan profil historis", "Lacak profil setiap grid selama 2021Q1-2025Q4 menggunakan label satu model historis. Ukuran transisi adalah deskriptif; bukan bukti perubahan tektonik kausal.", "analysis", [
        section("Definisi pasangan, pergantian, dan durasi", "Hanya dua triwulan berurutan yang sama-sama mempunyai profil cluster dihitung sebagai pasangan comparable. Noise, NO_EVENTS, dan fitur tidak lengkap memutus run. Stability rate = 1 - jumlah pergantian / pasangan comparable. Jika tidak ada pasangan, hasil NaN. State transition disimpan terpisah agar nol/noise tetap dapat diperiksa.", source("analysis",["temporal_tables"])),
        section("Ekspor riwayat, matriks dan durasi", "Simpan history, grid_stability, profile_transitions, state_transitions, transition_probabilities dan profile_spells untuk masing-masing metode.", source("analysis",["temporal"])),
        section("Jalankan analisis temporal", code='stability_by_grid = temporal(ROOT)\nfor method, summary in stability_by_grid.items():\n    display(Markdown(f"### {method.upper()}"))\n    display(summary.sort_values("n_profile_changes",ascending=False).head(15))\n    display(Image(filename=str(ROOT / f"reports/figures/{method}_transitions.png")))'),
        section("Riwayat grid yang dapat dipilih", "Nama deskriptif yang dibuat pada notebook 04 dipakai kembali di sini; tidak ada pengelompokan baru. Ganti GRID_ID sesuai tabel dan periksa perubahan nama bersama jumlah kejadian, magnitudo serta kedalaman.", 'from src.mapping import profile_descriptions\nhistory = pd.read_csv(ROOT / "reports/temporal/kmeans/history.csv")\ndescriptions = profile_descriptions(ROOT, "kmeans", cfg)\nhistory["nama_kelompok"] = history["profile"].map(lambda code: descriptions[code]["name"])\nGRID_ID = history.groupby("grid_id")["earthquake_count"].sum().idxmax()\nwith pd.option_context("display.max_colwidth", None):\n    display(history.loc[history["grid_id"].eq(GRID_ID),["grid_id","period","nama_kelompok","earthquake_count","avg_magnitude","avg_depth","low_support","changed"]].sort_values("period"))'),
    ]),
    ("08_stability/08_stability.ipynb", "Uji kestabilan dan sensitivitas", "Semua eksperimen di notebook ini memakai data 2021-2025. Tahun 2026 tidak digunakan untuk memilih parameter.", "analysis", [
        section("ARI parameter, grid, dan ambang magnitudo", "Uji seed dan k K-Means; eps serta min_samples DBSCAN; grid 25/50/100 km; M minimum 2.5/3/3.5/4. Untuk grid berbeda, ARI dibandingkan pada **event ID historis yang sama** setelah setiap event diberi label grid-triwulannya. Hasil bersifat event-weighted, mengecualikan grid kosong dan melaporkan overlap. Bukan ARI langsung antara baris grid yang berlainan. Parameter terpilih dipertahankan, scaler di-fit ulang pada training alternatif.", source("analysis",["stability"])),
        section("Jalankan eksperimen sensitivitas", "ARI tinggi dapat didominasi kelompok besar/noise. Karena itu ARI seluruh data dan subset assigned dilaporkan bersama coverage. Tidak ada cutoff ARI universal untuk menyatakan pola stabil.", 'parameter_ari, spatial_ari = stability(ROOT)\ndisplay(parameter_ari)\ndisplay(spatial_ari)'),
        section("Sensitivitas kebijakan pemilihan DBSCAN", "Tabel ini mengulang seleksi dari kandidat training dengan beberapa batas ukuran cluster dan fragmentasi. Ini menunjukkan seberapa bergantung hasil pada kebijakan screening yang dipilih.", 'display(pd.read_csv(ROOT / "reports/stability/dbscan_selection_policy.csv"))'),
        section("Sensitivitas jumlah event per grid-triwulan", "Profil dari satu atau dua event mudah mencerminkan diskretisasi fitur dan variasi sampling. Uji dukungan minimal 1/3/5/10 event: fit ulang scaler dan model hanya pada training yang lolos, dengan k/eps/min_samples tetap. ARI dibandingkan pada baris grid-triwulan yang sama dan coverage dilaporkan. Jika hanya sedikit baris lolos, klaim stabilitas tidak dapat digeneralisasi ke semua grid.", 'display(pd.read_csv(ROOT / "reports/stability/minimum_event_support.csv"))'),
    ]),
    ("09_testing_2026/09_testing_2026.ipynb", "Testing temporal 2026", "Training tetap 2021-2025. Model, scaler, threshold dan parameter dibekukan sebelum transformasi data 2026. Tidak ada label ground truth untuk menghitung accuracy.", "analysis", [
        section("Aturan ekstensi DBSCAN", "Scikit-learn DBSCAN tidak menyediakan predict bawaan. Ekstensi ini menugaskan titik hanya jika core points dalam radius eps berasal dari satu cluster. Tidak ada kecocokan -> NO_CORE_MATCH; beberapa cluster -> AMBIGUOUS_CORE_MATCH. Ini aturan tambahan yang dapat ditolak, bukan refit DBSCAN pada 2026.", source("analysis",["extend_dbscan"])),
        section("Prediksi K-Means, drift, dan periode parsial", "K-Means selalu memberi centroid terdekat; jarak melampaui kuantil 99% training ditandai sebagai flag eksploratif, bukan uji signifikansi. Q1-Q3 2026 dilaporkan terpisah dari Q4 parsial. Frekuensi Q4 dinormalisasi lama pengamatan, tetapi statistik magnitudo dan maksimum tetap tidak sebanding penuh dengan triwulan lengkap. Pergantian dibanding baseline 2025Q4, bukan otomatis pergantian dari triwulan tepat sebelumnya.", source("analysis",["testing_2026"])),
        section("Jalankan testing 2026", code='holdout_metrics, test_labels = testing_2026(ROOT)\ndisplay(holdout_metrics)\ndisplay(test_labels.groupby(["period","split","kmeans_profile"]).size().rename("n").reset_index())\ndisplay(pd.read_csv(ROOT / "reports/testing_2026/feature_drift.csv"))'),
        section("Audit kecocokan dan quarter", code='display(test_labels.groupby(["period","dbscan_profile"]).size().rename("n").reset_index())\nassert set(test_labels.loc[test_labels["split"].eq("test"),"period"]) == {"2026Q1","2026Q2","2026Q3"}\nprint("Q4 disimpan pada labels_partial_quarter.csv; tidak digabung ke evaluasi quarter lengkap.")'),
        section("Baca hasil testing dengan nama kelompok historis", "Nama berasal dari ciri kelompok training yang dijelaskan pada notebook 04/05. Data 2026 hanya dicocokkan dengan kelompok historis; angka pada baris di bawah adalah jumlah kotak, bukan jumlah event gempa.", 'from src.mapping import profile_descriptions\nfor method in ["kmeans","dbscan"]:\n    descriptions = profile_descriptions(ROOT, method, cfg)\n    named_test = test_labels.copy()\n    named_test["nama_kelompok"] = named_test[f"{method}_profile"].map(lambda code: descriptions[code]["name"])\n    display(Markdown(f"### Hasil pengujian {method.upper()}"))\n    with pd.option_context("display.max_colwidth", None):\n        display(named_test.groupby(["period","split","nama_kelompok"]).size().rename("jumlah_kotak").reset_index())'),
    ]),
    ("10_gis_mapping/10_gis_mapping.ipynb", "GIS polygon dan peta perubahan", "Peta profil menampilkan satu triwulan pada satu waktu melalui pilihan tahun-triwulan. Layer membedakan profil, nol kejadian, fitur tak lengkap, noise dan kecocokan core ambigu.", "mapping", [
        section("Geometri polygon dan arti nama kelompok", "Algoritma membentuk kelompok dari kemiripan lima fitur. Nama ramah pembaca dibuat sesudah fitting untuk merangkum ciri kelompok. Huruf A/B/C hanya pengenal; penamaan tidak mengubah label, anggota atau model. Istilah cenderung dangkal memakai rata-rata depth <70 km, mengikuti definisi preprocessing; lebih sering berarti mean jumlah event terbesar di antara cluster metode itu. Angka pada legenda berasal dari training 2021-2025; nilai kotak yang diklik adalah data periode terpilih.", 'COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#882255"]\nSPECIAL = {"NO_EVENTS":"#ededed","INCOMPLETE_FEATURES":"#999999","NOISE":"#333333","NO_CORE_MATCH":"#333333","AMBIGUOUS_CORE_MATCH":"#b58900"}\n\n' + source("mapping",["polygon_coordinates","profile_color","readable_period","profile_descriptions"])),
        section("Peta dasar lokal dan pemilihan periode", "Daratan Natural Earth dan Leaflet disertakan langsung dalam setiap HTML, sehingga peta dapat dibuka tanpa internet dan tidak meminta tile OpenStreetMap. Daratan hanya konteks visual, tidak mengubah seleksi event atau clustering. Zoom awal menyesuaikan bounding box penelitian; gunakan tombol +/- atau sentuhan untuk zoom. Tombol Sembunyikan grid / Tampilkan grid di kiri bawah menyembunyikan seluruh kotak hasil, sementara daratan tetap terlihat. Pergantian periode mempertahankan status grid; ketika ditampilkan kembali, grid mengikuti periode terbaru yang dipilih. Tombol juga tersedia pada peta perubahan.", source("mapping",["create_base_map","attach_grid_toggle","attach_selector"])),
        section("Peta profil dan jumlah perubahan", "Peta perubahan hanya menggunakan 2021-2025. Jumlah pergantian ditampilkan bersama jumlah pasangan comparable supaya satu perubahan dari satu pasangan tidak disamakan dengan satu dari 19 pasangan.", source("mapping",["create_maps"])),
        section("Buat dan buka peta", code='map_paths = create_maps(ROOT)\nfor path in map_paths:\n    display(FileLink(str(path)))\nprint("Peta HTML disimpan dalam:", ROOT / "maps")'),
        section("Ekspor GIS dan riwayat", "Geometri grid tersedia sebagai GeoJSON; join dengan CSV hasil berdasarkan grid_id. Untuk riwayat lengkap per grid buka notebook 07 dan pilih GRID_ID. GeoJSON tidak mengandung klaim batas administratif Indonesia.", 'display(pd.DataFrame([{"file":p.name,"size_MB":round(p.stat().st_size/1024**2,2)} for p in sorted((ROOT / "maps").iterdir())]))'),
    ]),
]


def build():
    notebook_root=ROOT / "notebooks"
    for index,(relative,title,description,module,sections) in enumerate(SPECS):
        if len(sys.argv)>1 and index not in [int(value) for value in sys.argv[1:]]:
            continue
        path=notebook_root / relative
        path.parent.mkdir(parents=True,exist_ok=True)
        links=[]
        for position,label in [(0,"Panduan"),(index-1,"Sebelumnya"),(index+1,"Berikutnya")]:
            if 0<=position<len(SPECS) and position!=index:
                target=os.path.relpath(notebook_root / SPECS[position][0],path.parent).replace("\\","/")
                links.append(f"[{label}]({target})")
        toc="\n".join(f"- [{i}. {title}](#s{i})" for i,(title,_,_) in enumerate([section("Inisialisasi dan dependensi"),*sections],1))
        results_hint = {
            1: "**Lihat hasil:** [jumlah catatan dan contoh data](#s4) | [jumlah per tahun](#s6). Hasil ada di bawah sel kode pada bagian tersebut.",
            2: "**Lihat hasil:** [jumlah gempa sebelum/sesudah preprocessing dan contoh data](#s6) | [data kotak-triwulan dan fitur untuk model](#s8). Hasil ada di bawah sel kode pada bagian tersebut.",
            3: "**Mulai membaca:** [jumlah data dan perbandingan tahun](#s3) | [grafik magnitudo dan kedalaman](#s5) | [kesimpulan EDA](#s9). Hasil ada di bawah sel kode. Baca berurutan dari bagian 3 sampai 9 untuk mengikuti penjelasannya.",
            5: "**Baca tiga grafik utama secara berurutan:** [1. Kurva k-distance](#s7) → [2. Grafik eps dan skor](#s8) → [3. Hasil kelompok](#s9). [Nama kelompok](#s6) | [Lampiran teknis](#s10). Grafik langsung tampil di bawah sel kode.",
            6: "**Mulai membaca:** [ringkasan hasil](#s3) | [grafik skor](#s5) | [grafik cakupan/noise](#s6) | [ukuran kelompok](#s7) | [ciri kelompok](#s8) | [kesepakatan dua metode](#s9) | [kesimpulan](#s10). Grafik dan tabel tampil di bawah sel kode dan hasilnya tersimpan di notebook.",
        }.get(index, "")
        cells=[nbf.v4.new_markdown_cell(f"# {index:02d}. {title}\n\n{description}\n\n{results_hint}\n\n{' | '.join(links)}\n\n### Daftar bagian\n\n{toc}")]
        cells.append(nbf.v4.new_markdown_cell('<a id="s1"></a>\n## 1. Inisialisasi dan dependensi\n\nJalankan dari atas ke bawah dengan kernel Python 3.12.11 (Seismic Research). Konfigurasi tersentral di config.json.'))
        initial=BOOTSTRAP + ("\n"+imports(module) if module else "")
        cells.append(nbf.v4.new_code_cell(initial))
        for number,(heading,explanation,code) in enumerate(sections,2):
            cells.append(nbf.v4.new_markdown_cell(f'<a id="s{number}"></a>\n## {number}. {heading}\n\n{explanation}'))
            if code:
                cells.append(nbf.v4.new_code_cell(code))
        cells.append(nbf.v4.new_markdown_cell("## Referensi dan batas interpretasi\n\n[Dokumentasi EMSC](https://www.seismicportal.eu/fdsn-wsevent.html), [clustering scikit-learn](https://scikit-learn.org/stable/modules/clustering.html), dan docs/metodologi.md. Hasil adalah profil aktivitas katalog dalam bounding box dan periode yang dipilih. Interpretasi risiko, prediksi kejadian gempa, serta klaim perubahan fisik memerlukan bukti tambahan."))
        nb=nbf.v4.new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3.12.11 (Seismic Research)","language":"python","name":"seismic-python312"},"language_info":{"name":"python","version":"3.12.11"},"toc":{"base_numbering":1,"number_sections":False},"research":{"stage":index,"training":"2021-2025","testing":"2026"}})
        nbf.validate(nb)
        nbf.write(nb,path)
        print(path.relative_to(ROOT))


if __name__=="__main__":
    build()
