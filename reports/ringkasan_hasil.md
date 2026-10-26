# Ringkasan hasil eksekusi

Snapshot ini dihasilkan dari data EMSC nyata yang tersimpan di data/raw/monthly. Semua 11 notebook berhasil dijalankan pada kernel Python 3.12.11 (Seismic Research). Notebook asli tidak ditulis oleh pipeline ini.

- Baris mentah API: 78,265.
- Baris bersih sesuai cakupan/tipe event: 78,213; ditolak: 52.
- Observasi grid-triwulan training layak: 13,471.
- Cakupan historis: 20 triwulan, 2021Q1-2025Q4.
- Proporsi baris training dengan 1-4 event: 86.57%; ini batas penting interpretasi profil.
- K-Means terpilih: k=6.
- DBSCAN terpilih: eps=0.505041, min_samples=20, 7 cluster, noise=9.04%.
- Status DBSCAN hanya diagnostik: False.
- Testing lengkap: 2026Q1-Q3; 2026Q4 dipisahkan sebagai parsial.

## Evaluasi historis

```text
 model                                                  evaluation_scope  n_rows  n_clusters  n_assigned  noise_fraction  largest_cluster_fraction_assigned  smallest_cluster_size  silhouette  davies_bouldin  silhouette_n  dbscan_diagnostic_only
kmeans seluruh baris training layak; noise dikeluarkan dari metrik jarak   13471           6       13471        0.000000                           0.410363                    536    0.398720        0.932800          2000                   False
dbscan seluruh baris training layak; noise dikeluarkan dari metrik jarak   13471           7       12253        0.090416                           0.396638                     28    0.126064        2.011581          2000                   False
kmeans             subset identik: baris training yang ditugaskan DBSCAN   12253           6       12253        0.000000                           0.450257                    352    0.425670        0.866523          2000                   False
dbscan             subset identik: baris training yang ditugaskan DBSCAN   12253           7       12253        0.000000                           0.396638                     28    0.126064        2.011581          2000                   False
```

Metrik perbandingan harus dibaca bersama scope. Tabel parameter search memakai rata-rata tiga sampel Silhouette; tabel ringkasan di atas memakai seed 42. K-Means/DBSCAN tidak memiliki accuracy tanpa label acuan.

## Testing temporal

```text
       split                method  ambiguous_fraction  n_rows  n_clusters  n_assigned  noise_fraction  largest_cluster_fraction_assigned  smallest_cluster_size  silhouette  davies_bouldin  silhouette_n
        test                kmeans            0.000000    2625           6        2625        0.000000                           0.468952                     91    0.376161        0.973501          2000
        test dbscan_core_extension            0.000381    2625           7        2277        0.132571                           0.433904                      5    0.155648        2.013470          2000
test_partial                kmeans            0.000000     298           6         298        0.000000                           0.355705                      8    0.397524        0.870319           298
test_partial dbscan_core_extension            0.000000     298           2         172        0.422819                           0.953488                      8    0.585860        0.455596           172
```

DBSCAN testing adalah ekstensi core-radius. Noise_fraction di tabel testing berarti seluruh titik yang tidak ditugaskan (termasuk ambigu), bukan hasil refit DBSCAN; ambiguous_fraction dilaporkan terpisah. K-Means selalu memberi label, sehingga assignment 100% bukan bukti semua pengamatan cocok dengan profil historis.

## Sensitivitas resolusi dan magnitudo

```text
method  perturbation  value  n_common_events  fraction_reference_events  ari_same_events_including_noise  ari_common_assigned  common_assigned_fraction  n_grid_quarters
kmeans       grid_km   25.0            36146                   1.000000                         0.442776             0.442776                  1.000000            21348
dbscan       grid_km   25.0            36146                   1.000000                         0.218412             0.114520                  0.698307            21348
kmeans       grid_km   50.0            36146                   1.000000                         1.000000             1.000000                  1.000000            13471
dbscan       grid_km   50.0            36146                   1.000000                         1.000000             1.000000                  0.727328            13471
kmeans       grid_km  100.0            36146                   1.000000                         0.319266             0.319266                  1.000000             6539
dbscan       grid_km  100.0            36146                   1.000000                         0.318332             0.181273                  0.584684             6539
kmeans min_magnitude    2.5            36146                   1.000000                         0.645576             0.645576                  1.000000            15219
dbscan min_magnitude    2.5            36146                   1.000000                         0.724377             0.811623                  0.702318            15219
kmeans min_magnitude    3.0            36146                   1.000000                         1.000000             1.000000                  1.000000            13471
dbscan min_magnitude    3.0            36146                   1.000000                         1.000000             1.000000                  0.727328            13471
kmeans min_magnitude    3.5            20508                   0.567366                         0.453681             0.453681                  1.000000             9947
dbscan min_magnitude    3.5            20508                   0.567366                         0.401753             0.273106                  0.660962             9947
kmeans min_magnitude    4.0            10903                   0.301638                         0.235065             0.235065                  1.000000             6429
dbscan min_magnitude    4.0            10903                   0.301638                         0.362440             0.258077                  0.595066             6429
```

ARI dihitung pada event ID bersama dan bersifat event-weighted. Perbedaan grid, threshold magnitudo, dan cakupan overlap harus dibaca sebelum menyatakan pola stabil. ARI bukan validasi terhadap kondisi tektonik sebenarnya.

## Sensitivitas dukungan event

```text
method  min_events_per_quarter  n_same_grid_quarters  fraction_reference_rows  n_clusters  noise_fraction  ari_same_grid_quarters  ari_common_assigned  common_assigned_fraction
kmeans                       1                 13471                 1.000000           6        0.000000                1.000000             1.000000                  1.000000
dbscan                       1                 13471                 1.000000           7        0.090416                1.000000             1.000000                  0.909584
kmeans                       3                  4042                 0.300052           6        0.000000                0.330899             0.330899                  1.000000
dbscan                       3                  4042                 0.300052           2        0.381000                0.638837             1.000000                  0.619000
kmeans                       5                  1809                 0.134288           6        0.000000                0.067089             0.067089                  1.000000
dbscan                       5                  1809                 0.134288           2        0.533997                0.440148             0.000000                  0.466003
kmeans                      10                   435                 0.032292           6        0.000000                0.007853             0.007853                  1.000000
dbscan                      10                   435                 0.032292           1        0.800000                0.437773             1.000000                  0.200000
```

Mayoritas unit mempunyai dukungan event rendah. Pada satu event, mean dan maksimum M sama serta proporsi dangkal hanya 0/1; struktur cluster dapat dipengaruhi diskretisasi itu. Jangan menyebut semua transisi sebagai perubahan regional yang kuat tanpa pemeriksaan dukungan, kestabilan dan kualitas katalog.

## Diagnostik completeness katalog

```text
 year  max_curvature_histogram_mode  analysis_threshold  threshold_below_histogram_mode                                                                               note
 2021                           3.1                 3.0                            True diagnostik histogram 0.1, bukan estimasi Mc tervalidasi atau bukti katalog lengkap
 2022                           3.0                 3.0                           False diagnostik histogram 0.1, bukan estimasi Mc tervalidasi atau bukti katalog lengkap
 2023                           3.0                 3.0                           False diagnostik histogram 0.1, bukan estimasi Mc tervalidasi atau bukti katalog lengkap
 2024                           3.0                 3.0                           False diagnostik histogram 0.1, bukan estimasi Mc tervalidasi atau bukti katalog lengkap
 2025                           2.5                 3.0                           False diagnostik histogram 0.1, bukan estimasi Mc tervalidasi atau bukti katalog lengkap
```

Ambang M>=3.0 belum membuktikan katalog lengkap; jenis magnitudo belum dihomogenisasi. Lihat docs/metodologi.md untuk batas seluruh keputusan. Temuan ini masih analisis eksploratif aktivitas dalam katalog, bukan peta risiko atau bukti final perubahan seismik fisik.

## Bukti verifikasi

reports/validation/environment.json, notebook_execution.json, data_integrity.json dan tests.xml mencatat interpreter, eksekusi, integritas, serta pengujian. Peta HTML tersedia di maps; riwayat per grid dan matriks transisi ada di reports/temporal.
