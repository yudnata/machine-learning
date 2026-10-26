# Metodologi dan batas pertanggungjawaban

## 1. Tujuan dan pemisahan periode

Tujuan: menemukan profil aktivitas pada unit **grid-triwulan**, mengukur perubahan profil historis, dan memeriksa penerapannya pada periode baru. K-Means dan DBSCAN merupakan unsupervised learning, tanpa label target. Training/model discovery: 2021Q1-2025Q4. Testing temporal: 2026, dengan cutoff **sebelum 26 Oktober 2026 pukul 00:00 UTC**. Q1-Q3 lengkap; Q4 baru observasi sementara. Tidak ada prediksi kapan gempa akan terjadi.

Scaler, centroid, core points, kandidat parameter, k-distance, threshold jarak K-Means, serta kebijakan pemilihan model hanya memakai training. Jangan mengubah parameter setelah melihat testing lalu tetap menyebut testing tersebut sebagai evaluasi independen. Perubahan berbasis testing memerlukan holdout baru.

## 2. Sumber dan kualitas katalog

EMSC/SeismicPortal FDSN Event service, CC BY 4.0. Batas -11..6 lintang dan 95..141 bujur adalah **bounding box**, mencakup laut dan sebagian wilayah tetangga. Hasil tidak mengklaim telah di-clip ke batas negara. Setiap bulan mempunyai cache JSON, waktu pengambilan, URL permintaan, jumlah respons, serta manifest. HTTP error menghentikan collection. Hasil yang mencapai limit dipecah lagi agar tidak terpotong diam-diam. Berhasil mengunduh seluruh interval membuktikan cakupan permintaan, bukan kelengkapan deteksi seismologis.

Depth diambil dari `properties.depth` dalam km. Koordinat ketiga GeoJSON disimpan terpisah; respons nyata memperlihatkan tanda yang berlawanan. Tidak digunakan `abs()` untuk menyembunyikan salah definisi. Depth negatif/di atas 800 km ditandai tidak valid. Rentang itu adalah screening fisik awal yang harus diperiksa untuk outlier, bukan bukti semua data lain benar. Magnitudo -2..10 ditangani sebagai rentang valid awal. ID duplikat mempertahankan lastupdate terbaru; ID hilang memakai kunci atribut. Semua keputusan meninggalkan kolom/CSV audit.

Tipe event asli dipertahankan pada `event_type`. Analisis utama menerima `ke`, `fe`, dan `de` (known/felt/damaging earthquake menurut kode ISF). Tipe tidak dikenal, hilang, atau lainnya masuk tabel rejected dengan alasan `event_type_not_accepted`; tidak otomatis disebut gempa tektonik. Katalog ini juga belum dipisahkan berdasarkan mekanisme sumber.

**M minimum 3.0 masih ambang eksploratif.** Notebook EDA menampilkan distribusi tahunan, author dan jenis magnitudo, jumlah pada beberapa ambang, serta mode histogram 0.1 sebagai diagnostik awal maximum curvature. Mode histogram bukan Mc tervalidasi. Uji sensitivitas M 2.5/3.0/3.5/4.0 tidak menggantikan pengujian completeness yang lebih kuat. Jenis magnitudo dari katalog belum dihomogenisasi menjadi Mw. Perubahan jaringan/author/jenis magnitudo dan kelengkapan pencatatan tetap merupakan kemungkinan pengganggu interpretasi temporal. Sampai isu itu diselesaikan, kesimpulan dibatasi pada **aktivitas yang tercatat dalam katalog**.

## 3. Grid dan agregasi

Grid utama 50 km pada proyeksi **Lambert azimuthal equal-area (LAEA)** berpusat 118 BT, -2.5 LS. Grid origin berada pada kelipatan koordinat proyeksi tetap; tidak bergeser menurut data. Luas sel interior adalah 2.500 km persegi pada bidang equal-area. Panjang sisi di permukaan bumi tidak tepat sama di semua lokasi. Analisis sensitivitas memakai 25 dan 100 km. Polygon ditransformasi kembali ke WGS84 dengan densifikasi sisi.

Universe training: sel yang setidaknya mempunyai satu event di atas ambang pada 2021-2025. Universe testing: gabungan sel historis dan sel baru yang muncul pada 2026. Dengan begitu, penambahan data testing tidak mengubah panel training. Ini bukan seluruh sel kosong dalam bounding box; wilayah yang tidak pernah aktif dalam periode pengamatan tidak diberi profil.

Panel historis melengkapi **20 triwulan untuk setiap grid historis**. Nol event berarti tidak ada event di atas ambang dalam katalog dan interval unduhan yang berhasil, bukan tidak ada gempa secara fisik. Count/frekuensi boleh nol; statistik magnitudo dan depth tetap NaN. `NO_EVENTS`, `INCOMPLETE_FEATURES`, `NOISE`, serta profil cluster dibedakan. Baris untuk clustering harus mempunyai fitur lengkap dan depth coverage >=80%. Proporsi dangkal memakai denominator depth valid. Baris 1-4 event tetap masuk model, namun diberi flag `low_support`; hasilnya perlu kehati-hatian karena maksimum/mean dan proporsi tidak stabil dengan dukungan kecil.

## 4. Fitur dan transformasi

Fitur jarak: log1p(frekuensi per bulan), mean magnitudo, maksimum magnitudo, mean depth, dan proporsi event dangkal (<70 km). Frekuensi triwulan lengkap adalah count/3. Triwulan parsial memakai lama observasi dalam bulan ekuivalen 30.4375 hari. Normalisasi durasi tidak membuat maksimum magnitudo atau proporsi parsial setara dengan periode penuh.

Count, density dan frekuensi mentah tidak sekaligus dimasukkan karena mengulang informasi. Latitude/longitude menjadi identitas spasial, bukan fitur clustering. Mean dan maksimum magnitudo tetap bisa berkorelasi, sehingga tabel korelasi harus ditinjau; log transform dan StandardScaler tidak menghapus semua redundansi. Tidak ada imputasi nol pada statistik kosong. Scaler hanya fit pada training ELIGIBLE dan dipakai transform testing.

## 5. Pemilihan model

K-Means: k=2..8, random_state 42, n_init 20. Pemilihan berdasarkan rata-rata Silhouette dari tiga sampel maksimal 2.000 baris (seed 42/7/21), tie-break DBI lebih rendah dan k lebih kecil. Simpan SD sampling, inertia/elbow, seluruh kandidat dan ukuran cluster. Selisih kecil antarskor tidak membuktikan pemenang universal. Evaluasi internal setelah tuning bersifat eksploratif dan optimistis; tabel testing adalah pemeriksaan tambahan pada data baru.

DBSCAN: min_samples 5/10/20; eps dari kuantil 0.50/0.70/0.85/0.95 jarak tetangga ke-k pada training standar. Query secara eksplisit memasukkan tetangga diri sendiri seperti definisi min_samples DBSCAN. Tidak menetapkan jumlah cluster tepat. Diagnosis training awal menghasilkan kandidat dengan ratusan cluster kecil dan Silhouette tinggi; karena itu kebijakan screening eksploratif menolak fragmentasi berlebihan: 2..20 cluster, anggota cluster terkecil >=20, noise <=50%, dan cluster dominan <=95% assigned. Batas ini **pilihan analitis yang disesuaikan dari diagnosis training**, bukan batas alami seismologi atau praregistrasi penelitian. Tabel sensitivitas kebijakan memperlihatkan konsekuensinya. Jika tak ada kandidat lolos, hasil disimpan dengan status `diagnostic_only`, tidak dipaksakan menjadi model tervalidasi.

Profil memakai nama netral `KMEANS_Pn` dan `DBSCAN_Pn`. ID bukan urutan intensitas dan tidak setara antaralgoritma. Profiling memakai nilai asli dan dukungan data. Tidak ada label risiko rendah/sedang/tinggi otomatis. PCA hanya proyeksi visual; model dipelajari pada lima fitur standar.

## 6. Evaluasi dan kestabilan

Silhouette lebih tinggi dan DBI lebih rendah membantu menilai struktur geometris, tetapi bukan accuracy. Noise dikeluarkan dari metrik jarak dan proporsinya wajib dilaporkan. Tabel perbandingan mencakup hasil seluruh training serta **subset identik yang assigned oleh DBSCAN** untuk kedua algoritma. Pemilihan parameter memakai rata-rata tiga seed; tabel evaluasi ringkas memakai seed 42 yang sama. Nilai keduanya dapat sedikit berbeda karena sampling.

ARI membandingkan partisi pada observasi yang sama dan tidak memerlukan pencocokan nomor label. Uji meliputi seed serta k K-Means, eps/min_samples DBSCAN, resolusi spasial, dan threshold magnitudo. Untuk resolusi berbeda, baris grid tidak sama; label dipetakan kembali ke **event ID historis yang sama**. ARI pada event bersama bersifat **event-weighted**, lebih dipengaruhi daerah dengan banyak kejadian, tidak mencakup grid kosong, dan bukan validasi spasial independen. Overlap serta assigned fraction dilaporkan. ARI termasuk noise dapat didominasi noise, sehingga ARI common-assigned juga disediakan. Ini analisis sensitivitas, belum spatial block bootstrap atau interval ketidakpastian inferensial.

Sensitivitas dukungan minimal 1/3/5/10 event per grid-triwulan juga diuji. Model dan scaler di-fit ulang pada subset historis yang memenuhi dukungan, dengan parameter model terpilih tetap. ARI dibandingkan pada baris grid-triwulan identik yang dipertahankan dan coverage dilaporkan. Ini penting karena mean/maksimum magnitudo sama dan proporsi dangkal hanya 0/1 ketika suatu unit hanya berisi satu event. Cluster dapat dipengaruhi diskretisasi ini; Silhouette bagus belum membuktikan profil regional yang kuat.

## 7. Transisi historis

Satu model yang di-fit pada gabungan 2021-2025 menjaga identitas profil. Pergantian profil hanya dihitung antara dua triwulan **berurutan yang sama-sama assigned ke profil**. Noise, nol event, fitur tidak lengkap dan celah periode memutus run. Stability rate = 1 - changes / comparable pairs. Tanpa pasangan layak, stability rate = NaN. Matriks seluruh state disimpan terpisah. Durasi dinyatakan dalam triwulan, bukan hari.

Pergantian label tidak otomatis perubahan tektonik. Periksa perubahan fitur, jumlah kejadian, kelengkapan katalog dan sensitivitas parameter. Peta count pergantian selalu menampilkan denominator pasangan. Matriks probabilitas merupakan proporsi transisi empiris deskriptif, bukan model Markov prediktif yang tervalidasi.

## 8. Testing 2026

K-Means memakai `predict` dari centroid historis. Jarak centroid >kuantil 99% training merupakan flag eksploratif; threshold ini bukan interval kepercayaan dan tidak membuktikan anomali fisik. DBSCAN memakai **ekstensi core-radius**, bukan predict bawaan: tepat satu cluster core dalam eps -> ditugaskan; tidak ada -> `NO_CORE_MATCH`; lebih dari satu -> `AMBIGUOUS_CORE_MATCH`. Model/scaler tidak di-fit ulang di 2026. `NO_CORE_MATCH` bukan otomatis cluster baru atau noise hasil DBSCAN refit.

Q1-Q3 dilaporkan sebagai quarter lengkap dan Q4 parsial berada dalam CSV/row evaluasi terpisah. Perbandingan baseline di testing memakai profil **2025Q4** secara eksplisit, bukan klaim transisi dari quarter tepat sebelumnya. Tidak menghitung accuracy, precision, recall atau confusion matrix terhadap label yang dibuat model sendiri. Tidak ada informasi kapan gempa akan terjadi.

## 9. Materi pengguna dan referensi teknis

PDF yang diberikan berisi 22 slide rancangan 2025, sedangkan teks saran membahas pengembangan 2021-2025 dan holdout 2026. Implementasi mengikuti klarifikasi pengguna: **2021-2025 training, 2026 testing**. Saran dan perintah yang tertulis dalam lampiran diperlakukan sebagai materi tinjauan; tidak otomatis dijalankan sebagai instruksi tambahan.

- [EMSC FDSN Event service: parameter, format, limit, lisensi](https://www.seismicportal.eu/fdsn-wsevent.html)
- [Scikit-learn clustering: K-Means, DBSCAN dan evaluasi](https://scikit-learn.org/stable/modules/clustering.html)
- [Scikit-learn DBSCAN API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [PyProj Transformer](https://pyproj4.github.io/pyproj/stable/api/transformer.html)
- [ISC/IASPEI Seismic Format: kode tipe event](https://www.isc.ac.uk/standards/isf/)

Tanggal peninjauan: 26 Oktober 2026. Referensi teknis mendukung penggunaan API/algoritma. Bukti research gap jurnal masih perlu ditulis dari telaah literatur; proyek ini tidak mengklaim novelty hanya dari dashboard atau pilihan algoritma.
