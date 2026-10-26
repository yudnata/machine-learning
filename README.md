# Penelitian aktivitas seismik: training 2021-2025, testing 2026

**Mulai dari [00_panduan.ipynb](notebooks/00_panduan.ipynb).** Notebook lama `../ml_20206 (1).ipynb` tidak diubah. Seluruh pekerjaan baru berada dalam folder ini.

Hasil aktual dan batas interpretasinya dicatat pada [ringkasan hasil eksekusi](reports/ringkasan_hasil.md).

Untuk memahami tujuan, perjalanan data, arti hasil, dan keluaran akhir yang diharapkan, baca [penjelasan utama proyek](LAPORAN_ALUR_DAN_HASIL_PENELITIAN.md). Tabel parameter, seluruh skor, dan detail sumber dipisahkan ke [laporan teknis dan angka](LAPORAN_TEKNIS_DAN_ANGKA.md).

Untuk membaca legenda dan menyiapkan presentasi, buka [cara menjelaskan hasil kepada dosen](docs/cara_menjelaskan_hasil.md). Nama kelompok di peta adalah ringkasan statistik setelah clustering; kelompok tetap dibentuk algoritma. Angka legenda merangkum training, sedangkan popup menunjukkan data kotak pada periode pilihan.

Tombol **Sembunyikan grid / Tampilkan grid** tersedia di kiri bawah semua peta HTML. Peta daratan tetap terlihat ketika kotak hasil analisis disembunyikan.

## Notebook dan urutan menjalankan

| No. | Notebook | Keluaran utama |
|---|---|---|
| 00 | [Panduan](notebooks/00_panduan.ipynb) | Pemeriksaan kernel, pustaka, dan metodologi |
| 01 | [Data collection](notebooks/01_data_collection/01_data_collection.ipynb) | Cache JSON bulanan dan manifest EMSC |
| 02 | [Data preprocessing](notebooks/02_data_preprocessing/02_data_preprocessing.ipynb) | Panel grid-triwulan, fitur, scaler historis |
| 03 | [EDA](notebooks/03_eda/03_eda.ipynb) | Audit katalog, distribusi, korelasi |
| 04 | [K-Means](notebooks/04_kmeans/04_kmeans.ipynb) | Seleksi k, centroid, label, profil |
| 05 | [DBSCAN](notebooks/05_dbscan/05_dbscan.ipynb) | k-distance, pencarian eps/min_samples, noise |
| 06 | [Model evaluation](notebooks/06_model_evaluation/06_model_evaluation.ipynb) | Perbandingan cakupan dan subset identik |
| 07 | [Temporal analysis](notebooks/07_temporal_analysis/07_temporal_analysis.ipynb) | Riwayat, transisi, durasi, kestabilan per grid |
| 08 | [Stability](notebooks/08_stability/08_stability.ipynb) | ARI parameter, grid, magnitudo, kebijakan seleksi |
| 09 | [Testing 2026](notebooks/09_testing_2026/09_testing_2026.ipynb) | Hasil Q1-Q3 dan Q4 parsial, drift, core matching |
| 10 | [GIS mapping](notebooks/10_gis_mapping/10_gis_mapping.ipynb) | Peta polygon dengan pemilih quarter dan peta perubahan |

Setiap notebook memuat penjelasan, kode proses yang terlihat, section bernomor, daftar isi, tautan navigasi, dan pemeriksaan hasil. Gunakan Outline pada VS Code atau Table of Contents Jupyter untuk melompat/melipat section. Penomoran folder menjaga urutan di Explorer. Jalankan **Run All** sesuai nomor; tiap notebook memakai kernel baru dan membaca hasil tahap sebelumnya dari disk. Memindahkan urutan cell dapat memutus dependensi, sehingga pengurutan dimaksudkan sebagai navigasi tahapan, bukan eksekusi bebas urutan.

## Kernel Python 3.12.11

Interpreter pilihan pengguna: `C:\Users\yudhi\.local\bin\python3.12.exe`, versi 3.12.11 dari uv. Semua dependensi proyek diinstal di interpreter tersebut, termasuk ipykernel. Kernel terdaftar sebagai **Python 3.12.11 (Seismic Research)**.

Di VS Code: buka notebook, klik **Select Kernel**, pilih **Python 3.12.11 (Seismic Research)**. Jika daftar belum berubah, jalankan **Developer: Reload Window**. Jika Pylance masih memakai interpreter lain, **Python: Select Interpreter** lalu pilih `.local\bin\python3.12.exe`. Jangan memilih Python312 versi 3.12.6 yang merupakan instalasi lain. Notebook mencetak `sys.executable` agar pilihan bisa diperiksa.

Untuk mengulang setup di mesin ini:

```powershell
.\setup_kernel.ps1
```

Python tersebut dikelola uv, sehingga instalasi global yang secara eksplisit diminta menggunakan opsi `--break-system-packages`. `requirements.txt` berisi dependensi proyek; `requirements-lock.txt` mencatat versi environment saat diverifikasi. `reports/validation/environment.json` menyimpan hasil import dan interpreter yang benar-benar dipakai.

## Struktur hasil

```text
notebooks/              notebook utama per tahap/metode
data/raw/monthly/       respons JSON asli, bisa dipakai offline
data/processed/         events bersih, panel, scaled features, scaler
models/kmeans/          model, search, profil, label training
models/dbscan/          model, search, profil, label training
reports/evaluation/     audit katalog dan perbandingan model
reports/temporal/       history, transisi, durasi per metode
reports/stability/      ARI dan sensitivitas kebijakan
reports/testing_2026/   test lengkap dan quarter parsial terpisah
reports/figures/        grafik EDA dan clustering
reports/validation/     bukti eksekusi notebook, environment, dan pengujian
maps/                   HTML interaktif serta geometri GeoJSON
src/                    fungsi pendukung dan kontrak data bersama
scripts/                reproduksi notebook dan pemeriksaan
tests/                  pengujian integritas metodologi
docs/                   keputusan metodologi dan catatan sumber
```

## Konfigurasi dan cara membaca hasil

Pengaturan ada di `config.json`. Training 2021-2025, testing sebelum 2026-10-26 00:00 UTC. **2026Q4 belum lengkap** dan dipisah dari Q1-Q3. Grid utama 50 km pada proyeksi LAEA; sensitivitas 25/100 km. Ambang M>=3.0 masih eksploratif. Jangan menyamakan batas unduhan lengkap dengan katalog yang pasti lengkap secara seismologis.

Jika konfigurasi berubah, jalankan ulang notebook 01 bila cakupan unduhan berubah, lalu 02 sampai 10. Hash konfigurasi/data mendeteksi pemakaian preprocessing/model lama. Peta dan tabel ringkasan harus dihasilkan ulang setelah tahap model berubah. Jangan mencampurkan file hasil dari dua konfigurasi.

**Baca [metodologi dan batas interpretasi](docs/metodologi.md)** sebelum menggunakan hasil dalam laporan. Profil aktivitas bukan peta risiko. Tidak ada label ground truth atau accuracy. DBSCAN pada 2026 menggunakan aturan core-radius tambahan yang menangani kondisi ambigu. Pergantian profil perlu ditinjau bersama dukungan data dan hasil sensitivitas.

## Reproduksi dan verifikasi

```powershell
& "$env:USERPROFILE\.local\bin\python3.12.exe" -m pytest -q
& "$env:USERPROFILE\.local\bin\python3.12.exe" scripts/execute_notebooks.py
```

Skrip kedua menjalankan semua notebook pada kernel yang terdaftar dan menyimpan output di dalam notebook. Data collection menggunakan cache; menghapus/mengganti cache adalah keputusan eksplisit saat memperbarui snapshot katalog. Setiap HTML peta menyertakan Leaflet dan daratan Natural Earth, sehingga bisa dibuka tanpa internet serta tanpa permintaan tile OpenStreetMap. Daratan hanya konteks visual; hasil clustering tetap sama. Sumber dan lisensi ada di `assets/maps/sources.json` dan `LEAFLET_LICENSE.txt`. Jalankan `scripts/prepare_map_assets.py` hanya jika aset lokal hilang; tahap persiapan itu memerlukan internet.
