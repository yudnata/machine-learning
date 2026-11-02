# Perbandingan jumlah kelompok untuk naskah jurnal

Dokumen ini mengklarifikasi interpretasi hasil yang sudah ada. Jika ada pernyataan lain yang menyebut hasil Silhouette sebagai “seri” secara statistik, atau menyatakan penyebab perbedaannya hanya jumlah kelompok, gunakan penjelasan di bawah. Tidak ada model yang dilatih ulang untuk dokumen ini.

## 1. Pertanyaan dan rancangan perbandingan

Ada dua pertanyaan yang berbeda:

1. **Perbandingan konfigurasi terpilih:** bagaimana hasil terbaik menurut aturan seleksi yang diterapkan pada masing-masing metode? K-Means terpilih pada k=6 dan K-Medoids pada k=2. DBSCAN tidak menerima k sebagai masukan: kombinasi eps dan min_samples yang terpilih menghasilkan 7 kelompok dan noise. Ketiga hasil boleh dibandingkan, tetapi jumlah kelompok dan cakupan berbeda sehingga perbedaan skor tidak dapat diatribusikan hanya pada algoritma.
2. **Perbandingan pada granularitas tetap:** bagaimana hasil K-Means dan K-Medoids ketika jumlah kelompok ditetapkan sama? Gunakan k=6 sebagai analisis terkontrol yang sudah tersedia, dan tunjukkan juga kurva k=2–8. Kontrol ini membandingkan implementasi, inisialisasi, fungsi objektif, dan hasil fitting yang digunakan dalam penelitian, bukan efek algoritma murni yang terbebas dari seluruh faktor lain.

Pada K-Means dan K-Medoids, kandidat k=2–8 dinilai dengan rata-rata Silhouette pada tiga sampel, lalu DBI dan k lebih kecil dipakai sebagai pemecah nilai yang sama. DBSCAN diseleksi dari kandidat yang melewati batas noise, dominasi, fragmentasi, dan dukungan, kemudian berdasarkan Silhouette. Oleh karena itu **aturan seleksi tidak identik untuk ketiga metode**. Tujuh kelompok DBSCAN adalah hasil parameter dan kebijakan penyaringan tersebut, bukan k yang ditetapkan seperti pada K-Means atau PAM. Memaksa semua metode memiliki tujuh kelompok akan mengubah pertanyaan, dan tidak menjamin perbandingan DBSCAN lebih adil.

## 2. Hasil yang dapat dilaporkan

Tabel berikut memakai 12.253 baris training yang ditugaskan DBSCAN. Semua model dilatih pada 13.471 baris training, lalu label aslinya dievaluasi pada subset ini tanpa fitting ulang. Silhouette dihitung pada satu sampel tetap hingga 2.000 baris dari subset; DBI dan Calinski–Harabasz memakai seluruh subset. Karena hasil pada subset dipengaruhi oleh baris noise yang dikeluarkan, hasil ini tidak mewakili seluruh training tanpa syarat.

| Perbandingan | Model | Jumlah kelompok | Silhouette | DBI | Calinski–Harabasz |
|---|---|---:|---:|---:|---:|
| Konfigurasi terpilih | K-Means | 6 | 0,4257 | 0,8665 | 7.442 |
| Konfigurasi terpilih | K-Medoids | 2 | 0,4298 | 1,1103 | 6.007 |
| Konfigurasi terpilih | DBSCAN | 7 | 0,1261 | 2,0116 | 1.776 |
| Kontrol k=6 | K-Medoids | 6 | 0,3472 | 1,0078 | 7.261 |

Pada konfigurasi terpilih, Silhouette K-Medoids lebih tinggi secara numerik sebesar 0,0041; K-Means lebih baik pada DBI dan Calinski–Harabasz. Perbedaan kecil ini **belum membuktikan kesetaraan statistik**. Simpangan baku sekitar 0,011 berasal dari hanya tiga skor sampel pada tahap seleksi dan data/cakupannya berbeda dari perbandingan subset; nilai tersebut bukan ukuran ketidakpastian selisih berpasangan pada subset. Bootstrap baris yang sudah ada juga belum memperhitungkan korelasi spasial dan temporal.

Pada k=6 dan subset evaluasi yang sama, K-Means memiliki Silhouette lebih tinggi, DBI lebih rendah, dan Calinski–Harabasz lebih tinggi daripada K-Medoids. Pada masing-masing k=2–8, hasil pencarian pada seluruh training juga menunjukkan rata-rata Silhouette K-Means lebih tinggi daripada K-Medoids. Ini mendukung pilihan K-Means pada ukuran dan konfigurasi yang diuji, tetapi bukan pembuktian bahwa K-Means selalu lebih unggul, dan perbandingan k berbeda tidak membuktikan bahwa k adalah satu-satunya penyebab selisih.

K-Medoids k=2 tetap merupakan hasil sah dari aturan seleksinya; dua kelompoknya terutama mencerminkan perbedaan kedalaman. K-Medoids k=6 adalah **analisis kontrol**, bukan hasil yang boleh disebut k optimum menurut aturan Silhouette tersebut. Medoid adalah observasi nyata yang dapat ditelusuri. K-Means k=6 dapat dipilih sebagai model utama jika tujuan penelitian mencakup profil yang lebih rinci dan kriteria internal pada granularitas yang sama, dengan pilihan enam kelompok dinyatakan sebagai keputusan yang bergantung pada tujuan dan rentang kandidat yang diuji.

DBSCAN menghasilkan noise 1.218 dari 13.471 baris (9,04%). Laporkan angka ini bersama skor dan tampilkan profil/ukuran kelompok. Skor pada subset assigned bersifat bersyarat; penyingkiran noise dapat memengaruhi skor seluruh model. Silhouette dan DBI cenderung menguntungkan kelompok yang kompak/konveks, sedangkan CH berbasis dispersi terhadap rata-rata; ketiganya bukan ukuran netral untuk seluruh bentuk kelompok. Hasil DBSCAN yang rendah pada ukuran ini tidak membuktikan bahwa struktur kepadatan tidak ada atau DBSCAN secara umum buruk. Lihat [panduan evaluasi clustering scikit-learn](https://scikit-learn.org/stable/modules/clustering.html) dan [dokumentasi DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).

## 3. Contoh narasi hasil untuk jurnal

> Pemilihan jumlah kelompok K-Means dan K-Medoids dilakukan secara terpisah pada kandidat k=2–8 berdasarkan rata-rata Silhouette tiga sampel, sehingga diperoleh masing-masing k=6 dan k=2. Parameter DBSCAN dipilih dari kandidat yang memenuhi batas kualitas dan cakupan yang ditentukan; kombinasi terpilih menghasilkan tujuh kelompok dan 9,04% noise. Pada 12.253 unit kotak-triwulan yang ditugaskan DBSCAN, Silhouette konfigurasi terpilih adalah 0,4257 untuk K-Means, 0,4298 untuk K-Medoids, dan 0,1261 untuk DBSCAN. Pada subset yang sama, DBI masing-masing sebesar 0,8665, 1,1103, dan 2,0116. Karena jumlah kelompok berbeda, perbandingan tambahan dilakukan pada k=6: Silhouette K-Means dan K-Medoids masing-masing 0,4257 dan 0,3472, dengan DBI 0,8665 dan 1,0078. K-Means k=6 dipilih untuk analisis utama karena memberikan profil yang lebih rinci daripada K-Medoids k=2 dan memperoleh ukuran internal yang lebih baik dalam kontrol k=6. Kesimpulan ini terbatas pada fitur, kandidat, dan ukuran evaluasi yang digunakan; tanpa label acuan, skor tersebut tidak merupakan akurasi atau validasi geologi.

## 4. Penguatan sebelum disubmit

- Tetapkan tujuan tingkat rincian profil dan jelaskan alasan praktis enam kelompok, termasuk ukuran kelompok dan karakteristiknya. Sajikan k=2 sebagai alternatif yang lebih kasar, bukan salah.
- Jadikan tabel konfigurasi terpilih dan kontrol k=6 dua panel terpisah. Sajikan kurva k=2–8, cakupan dan noise DBSCAN, serta analisis sensitivitas kebijakan DBSCAN; hasil seleksi bergantung pada screening.
- Evaluasi 2026 dengan model dan standardisasi yang dibekukan, pada subset penugasan yang identik untuk semua metode dan dengan cakupan setiap metode. Ini evaluasi temporal eksternal untuk geometri partisi, bukan akurasi, dan ekstensi core-radius DBSCAN adalah aturan tambahan.
- Jika ingin inferensi perbedaan, rancang evaluasi berpasangan dengan resampling blok spasial/temporal yang sesuai dan jelaskan pilihan blok serta keterbatasannya. Tiga subsampel dan bootstrap baris independen tidak cukup untuk menyimpulkan kesetaraan. Hindari menyesuaikan desain setelah melihat hasil tanpa menyebutnya eksploratif.
- Bahas ketidakpastian katalog, ketercukupan event per kotak-triwulan, korelasi fitur dan sensitivitas resolusi grid. Tidak ada klaim sebab tektonik atau validasi seismologis dari skor internal saja.

Sumber hasil: [perbandingan model](../reports/evaluation/model_comparison.csv), [pencarian K-Means](../models/kmeans/parameter_search.csv), [pencarian K-Medoids](../models/kmedoids/parameter_search.csv), [evaluasi model](../notebooks/07_model_evaluation/07_model_evaluation.ipynb), dan [sensitivitas kebijakan DBSCAN](../reports/stability/dbscan_selection_policy.csv).
