# Cara menjelaskan hasil kepada dosen

## Apakah kategorinya kita tentukan sendiri?

Kelompoknya dibentuk oleh algoritma clustering. Peneliti memilih data, fitur, ukuran grid, algoritma, dan aturan evaluasi parameter. Pilihan tersebut memengaruhi hasil dan harus dijelaskan. Setelah model membentuk kelompok, peneliti membaca statistik kelompok lalu memberi nama deskriptif supaya hasil mudah dipahami.

Contoh aktual: K-Means menemukan kelompok berkode `KMEANS_P5`. Pada training 2021-2025, kelompok itu mempunyai rata-rata sekitar 8,9 kejadian per kotak per triwulan, magnitudo 3,7, dan kedalaman 38 km. Karena rata-rata jumlah kejadiannya terbesar di antara enam kelompok K-Means, tampilan peta menamainya **Kelompok F: Kejadian lebih sering; cenderung dangkal**.

Nama itu merangkum hasil yang sudah terbentuk. Nama tersebut tidak dipakai untuk menentukan keanggotaan. Tidak ada aturan seperti "kalau jumlah gempa >5, otomatis Kelompok F". K-Means membandingkan seluruh lima fitur yang ditransformasi dan distandardisasi terhadap centroid historis.

## Di notebook mana nama diberikan?

- **04 K-Means, bagian 6: Penamaan dan interpretasi kelompok (setelah clustering).** Bagian ini mendefinisikan aturan pada fungsi `profile_descriptions()`. Bagian 7 menjalankan penamaan dan menampilkan tabel kode, nama, serta statistik dasarnya.
- **05 DBSCAN, bagian 5: Penamaan dan interpretasi kelompok (setelah clustering).** Bagian ini mendefinisikan aturan nama. Bagian 6 menjalankannya dan menampilkan tabel nama serta statistik dasar untuk DBSCAN, termasuk status tidak masuk kelompok.
- Notebook **06** memakai nama yang sama untuk perbandingan profil; **07** untuk riwayat; **09** untuk testing; **10** untuk legenda dan popup peta.

Kode pendukung bersama ada di `src/mapping.py` pada fungsi `profile_descriptions()` dan `profile_interpretation_table()`. Notebook 04/05 menampilkan definisi fungsi itu secara penuh agar proses penamaannya dapat diperiksa. Semua nama dihitung dari statistik training, tidak dari data testing. Tabel hasil penamaan disimpan sebagai `reports/evaluation/kmeans_profile_interpretation.csv` dan `dbscan_profile_interpretation.csv`.

## Kalimat untuk presentasi

> Kami mengelompokkan kotak wilayah berdasarkan kemiripan aktivitas gempa, meliputi frekuensi kejadian, rata-rata dan maksimum magnitudo, rata-rata kedalaman, serta proporsi kejadian dangkal. Kelompok dibentuk oleh K-Means dan DBSCAN dari data 2021-2025. Setelah itu, kami memberi nama deskriptif berdasarkan karakteristik rata-rata setiap kelompok. Model historis kemudian diterapkan pada data 2026 untuk melihat kecocokan profil pada periode baru.

K-Means menghasilkan enam kelompok menurut aturan seleksi yang digunakan. Jumlah enam bukan ditetapkan agar mewakili enam kategori bahaya; kandidat k=2 sampai 8 dibandingkan memakai evaluasi training. Hasil k=6 dan k=2 mempunyai Silhouette yang berdekatan, sehingga enam kelompok masih pilihan eksploratif yang perlu dibaca bersama sensitivitas.

DBSCAN menghasilkan tujuh kelompok dalam konfigurasi terpilih, ditambah status untuk data yang tidak masuk kelompok. Pengujian DBSCAN pada 2026 menggunakan aturan tambahan kedekatan ke core points historis. Huruf kelompok hanya pengenal; Kelompok A K-Means tidak sama dengan Kelompok A DBSCAN.

## Cara membaca peta profil

1. Pilih periode, misalnya **Juli-September 2026**.
2. Setiap kotak mewakili sekitar 50 x 50 km pada proyeksi penelitian.
3. Warna menunjukkan kelompok dengan karakteristik serupa. Legenda menyajikan nama serta rata-rata kelompok pada 2021-2025.
4. Klik kotak untuk membaca jumlah kejadian, magnitudo, dan kedalaman pada **periode yang dipilih**. Angka ini dapat berbeda dari rata-rata historis kelompok.
5. Pilih periode lain dan amati kotak dengan kode yang sama untuk memeriksa perubahan ciri aktivitasnya.

Nama "cenderung dangkal" merangkum rata-rata kedalaman kelompok di bawah 70 km, mengikuti definisi penelitian. Nama "lebih sering" merujuk pada rata-rata jumlah kejadian terbesar di antara kelompok algoritma tersebut. Ini aturan penulisan nama setelah clustering, bukan aturan pembentukan cluster atau kategori risiko.

Tidak ada gempa tercatat berarti tidak ada kejadian di atas ambang magnitudo penelitian pada katalog dan periode tersebut. Tanpa kelompok berarti data belum cukup atau belum cocok dengan kelompok historis menurut metode. Kedua status itu tidak menentukan keamanan wilayah.

## Cara membaca peta perubahan

Peta perubahan menjawab **berapa kali suatu kotak berganti kelompok sepanjang 2021-2025**, bukan berapa banyak gempa terjadi. Perubahan hanya dihitung pada dua triwulan berurutan yang sama-sama mempunyai kelompok. Arahkan kursor untuk melihat berapa pasangan triwulan yang dapat dibandingkan. Nol pergantian dari satu pasangan berbeda dukungan buktinya dengan nol dari 19 pasangan.

Pergantian kelompok dapat mencerminkan perubahan frekuensi, magnitudo, kedalaman, atau campurannya. Pergantian itu perlu diperiksa bersama jumlah kejadian dan kualitas katalog sebelum disebut perubahan seismik yang kuat.

## Batas kesimpulan

Hasil ini memetakan profil aktivitas yang tercatat dalam katalog, bukan tingkat risiko, prediksi kapan gempa terjadi, atau klasifikasi aman/tidak aman. Sekitar 86,6% baris training hanya memuat 1-4 kejadian. Sensitivitas terhadap resolusi dan dukungan event masih besar. Penamaan yang mudah dibaca membantu presentasi, tetapi tidak menambah kekuatan bukti statistik.
