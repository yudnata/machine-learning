"""Penjelasan proyek untuk pembaca pertama, berdasarkan hasil yang tersimpan."""
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

from build_full_report import ROOT, NOTEBOOKS, number, notebook


def build():
    meta = json.loads((ROOT / "data/processed/metadata.json").read_text())
    km = pd.read_csv(ROOT / "models/kmeans/profiles.csv")
    panel = pd.read_csv(ROOT / "data/processed/panel.csv")
    train = panel.loc[panel["split"].eq("train") & panel["state"].eq("ELIGIBLE")]
    example = pd.read_csv(ROOT / "reports/temporal/kmeans/history.csv")
    example = example.loc[example["grid_id"].eq("5_-13") & example["period"].ge("2025Q1")].copy()
    test = pd.read_csv(ROOT / "reports/testing_2026/labels_all_2026.csv")
    first_test = test.loc[test["grid_id"].eq("5_-13") & test["period"].eq("2026Q1")].iloc[0]
    comparison = pd.read_csv(ROOT / "reports/evaluation/model_comparison.csv").iloc[2:]
    score_km = comparison.loc[comparison["model"].eq("kmeans")].iloc[0]
    score_db = comparison.loc[comparison["model"].eq("dbscan")].iloc[0]
    selected_k = json.loads((ROOT / "models/kmeans/metadata.json").read_text())["selected_k"]
    db_labels = pd.read_csv(ROOT / "models/dbscan/training_labels.csv")
    db_eligible = db_labels.loc[db_labels["cluster"].notna()]
    noise_fraction = db_eligible["cluster"].eq(-1).mean()
    low_support_fraction = train["low_support"].mean()
    descriptions = json.loads((ROOT / "reports/evaluation/kmeans_profile_explanations.json").read_text(encoding="utf-8"))
    rows = []
    periods = {"2025Q1":"Januari-Maret 2025", "2025Q2":"April-Juni 2025", "2025Q3":"Juli-September 2025", "2025Q4":"Oktober-Desember 2025"}
    for row in example.itertuples():
        name = descriptions[row.profile]["name"].split(":", 1)[0]
        rows.append(f"| {periods[row.period]} | {row.earthquake_count} | {name} |")
    test_name = descriptions[first_test["kmeans_profile"]]["name"].split(":", 1)[0]
    rows.append(f"| Januari-Maret 2026, data pengujian | {int(first_test['earthquake_count'])} | {test_name} |")
    examples = "\n".join(rows)
    profile_e = km.loc[km["profile"].eq("KMEANS_P4")].iloc[0]
    profile_f = km.loc[km["profile"].eq("KMEANS_P5")].iloc[0]
    profile_b = km.loc[km["profile"].eq("KMEANS_P1")].iloc[0]
    eligible_test = test.loc[test["split"].eq("test") & test["state"].eq("ELIGIBLE")]
    n_matched = eligible_test["dbscan_cluster"].ge(0).sum()
    n_no_match = eligible_test["dbscan_profile"].eq("NO_CORE_MATCH").sum()
    n_ambiguous = eligible_test["dbscan_profile"].eq("AMBIGUOUS_CORE_MATCH").sum()
    content = f'''# Memahami Proyek dan Hasil Analisis Gempa

**Bacaan utama untuk memahami keseluruhan proyek.** Angka di sini berasal dari hasil proyek yang tersimpan. Rincian rumus, parameter, dan seluruh skor tersedia di [laporan teknis pendamping](LAPORAN_TEKNIS_DAN_ANGKA.md).

## 1. Gambaran proyek dalam satu kalimat

**Kita ingin mengetahui wilayah mana yang mempunyai pola aktivitas gempa yang mirip, bagaimana polanya berubah selama 2021-2025, dan apakah pola lama masih cocok untuk data 2026.**

Pola yang dimaksud adalah gabungan dari seberapa sering gempa tercatat, berapa magnitudonya, dan seberapa dalam sumbernya. Misalnya, dua wilayah sama-sama jarang mengalami gempa tercatat, tetapi satu didominasi kejadian dangkal sementara satu lagi jauh lebih dalam. Keduanya bisa masuk kelompok berbeda.

Nama proyek ini dapat dibaca sebagai **pencarian dan pelacakan profil aktivitas seismik**. Kata profil berarti ringkasan ciri aktivitas suatu wilayah pada suatu periode.

## 2. Hasil akhir yang diharapkan

Hasil penelitian seharusnya membuat kita mampu menjawab tiga hal berikut.

| Pertanyaan penelitian | Bentuk hasil yang diharapkan |
| --- | --- |
| Tipe aktivitas apa yang ditemukan? | Kelompok dengan ciri yang dapat dijelaskan, misalnya lebih sering dan cenderung dangkal, atau jarang tetapi lebih dalam. |
| Bagaimana suatu wilayah berubah dari waktu ke waktu? | Riwayat wilayah yang memperlihatkan kapan kelompoknya tetap, kapan berganti, dan seberapa kuat dukungan datanya. |
| Apakah pola historis masih cocok pada data baru? | Hasil penerapan kelompok 2021-2025 pada 2026, termasuk pengamatan yang belum cocok. |

**Peta adalah alat untuk melihat jawaban itu. Temuan penelitiannya berasal dari data dan analisis yang menjelaskan peta.**

Contoh bentuk temuan yang ingin disusun adalah: suatu wilayah awalnya mirip kelompok dengan kejadian lebih sering, kemudian berpindah ke kelompok lain, dan pola tersebut tetap atau berubah ketika diperiksa pada periode baru. Pernyataan seperti itu perlu angka pendukung dan pemeriksaan kestabilan.

Penelitian ini belum menghasilkan dasar untuk mengatakan suatu wilayah aman, berisiko tinggi, atau akan mengalami gempa pada tanggal tertentu.

## 3. Unit yang kita bandingkan: satu kotak selama tiga bulan

Wilayah dibagi menjadi kotak sekitar **50 x 50 km**. Data setiap kotak diringkas per tiga bulan, yang disebut **triwulan**.

| Kode periode | Artinya |
| --- | --- |
| Q1 | Januari-Maret |
| Q2 | April-Juni |
| Q3 | Juli-September |
| Q4 | Oktober-Desember |

Bayangkan kita memilih satu kotak dan hanya melihat Januari-Maret. Semua kejadian di dalamnya diringkas menjadi satu catatan: jumlah gempa, rata-rata dan maksimum magnitudo, rata-rata kedalaman, serta proporsi kejadian dangkal. Untuk April-Juni, kita membuat catatan baru pada kotak yang sama.

**Jadi, yang dikelompokkan model adalah keadaan satu kotak pada satu triwulan.** Satu wilayah bisa berganti kelompok karena aktivitasnya pada periode berikutnya mempunyai ciri berbeda.

Jumlah gempa juga berbeda dari jumlah catatan model. Sepuluh gempa di satu kotak pada satu triwulan dapat menjadi **satu catatan ringkasan**, bukan sepuluh baris model.

## 4. Perjalanan data dari awal sampai menjadi hasil

### Langkah 1: Mengambil catatan gempa

Kita mengunduh data dari EMSC, yaitu sumber katalog kejadian gempa. Collection memperoleh **{number(meta['n_raw'])} catatan** untuk wilayah dan rentang waktu penelitian.

**Hasil tahap ini:** daftar kejadian, lengkap dengan waktu, lokasi, magnitudo, dan kedalaman. Angka ini masih mencakup catatan yang perlu diperiksa.

Notebook: {notebook(1, '01 Data collection')}.

### Langkah 2: Membersihkan dan merangkum data

Kita memeriksa tipe kejadian, waktu, koordinat, duplikat, dan kedalaman. Hasilnya **{number(meta['n_clean'])} kejadian lolos pemeriksaan**. Sebanyak {number(meta['n_rejected'])} catatan tidak diterima karena tipe kejadiannya tidak memenuhi ketentuan analisis.

Analisis utama lalu memakai magnitudo **minimal 3,0**. Tersisa **50.198 kejadian** untuk diringkas menurut kotak dan triwulan. Dari data 2021-2025, model memakai **{number(len(train))} catatan kotak-triwulan** yang mempunyai fitur lengkap.

Nilai-nilai fitur disetarakan skalanya supaya kedalaman dalam kilometer tidak otomatis lebih berpengaruh hanya karena angkanya lebih besar. Ini disebut standardisasi.

**Hasil tahap ini:** tabel keadaan setiap kotak pada setiap periode. Kotak tanpa kejadian tetap bisa diberi jumlah 0; rata-rata magnitudo dan kedalamannya tetap kosong karena tidak ada kejadian untuk dirata-ratakan.

Notebook: {notebook(2, '02 Data preprocessing')}.

### Langkah 3: Memahami isi data sebelum membentuk kelompok

Kita memeriksa distribusi magnitudo dan kedalaman, jumlah kejadian dari waktu ke waktu, serta hubungan antarfitur. Pemeriksaan ini disebut **EDA**.

**Hasil tahap ini:** gambaran isi dan masalah data. Salah satu temuan pentingnya adalah rata-rata dan maksimum magnitudo masih saling berkaitan. Kita juga belum membuktikan bahwa pencatatan katalog sama lengkapnya sepanjang lima tahun.

Notebook: {notebook(3, '03 EDA')}.

### Langkah 4: Mencari kelompok dengan dua metode

**K-Means** mencoba membentuk kelompok yang mempunyai ciri mirip dengan pusat kelompoknya. Kode mencoba beberapa jumlah kelompok, lalu memilih **{selected_k} kelompok** menurut aturan evaluasi yang digunakan.

**DBSCAN** mencari kelompok berdasarkan kepadatan kemiripan data. Metode ini juga dapat membiarkan suatu catatan tidak masuk kelompok. Pada pengaturan yang terpilih, hasilnya **7 kelompok**, dengan sekitar **{number(noise_fraction*100, 2)}%** catatan training tidak masuk kelompok DBSCAN.

**Hasil tahap ini:** setiap catatan yang dapat dikelompokkan memperoleh kode kelompok. Dua metode dapat memberi hasil berbeda karena cara membentuk kelompoknya berbeda.

Notebook: {notebook(4, '04 K-Means')} dan {notebook(5, '05 DBSCAN')}.

### Langkah 5: Menilai hasil kelompok

Kita memeriksa apakah anggota kelompok cukup mirip dan apakah kelompok satu cukup terpisah dari kelompok lain.

Pada 12.253 catatan training yang sama-sama dibandingkan, hasilnya:

| Penilaian | K-Means | DBSCAN | Cara membaca |
| --- | --- | --- | --- |
| Silhouette | {number(score_km['silhouette'], 3)} | {number(score_db['silhouette'], 3)} | Lebih tinggi berarti pemisahan kelompok lebih baik menurut ukuran ini. |
| Davies-Bouldin | {number(score_km['davies_bouldin'], 3)} | {number(score_db['davies_bouldin'], 3)} | Lebih rendah berarti kelompok lebih terpisah menurut ukuran ini. |

**Hasil tahap ini:** dalam percobaan sekarang, K-Means menghasilkan struktur kelompok yang lebih baik menurut kedua ukuran tersebut. Skor ini **bukan persentase akurasi** dan belum membuktikan kelompok benar secara geologi.

Jumlah enam kelompok juga masih pilihan eksperimen. Skor untuk dua kelompok berdekatan, sehingga enam belum menjadi satu-satunya jawaban yang pasti.

Notebook: {notebook(6, '06 Model evaluation')}.

### Langkah 6: Membaca ciri kelompok lalu memberi nama

Algoritma awalnya menghasilkan kode seperti `KMEANS_P5`. Kita kemudian membaca rata-rata cirinya dan memberi nama deskriptif supaya mudah dipahami.

Beberapa contoh aktual dari enam kelompok K-Means:

| Kelompok | Gambaran hasil historis |
| --- | --- |
| E | Rata-rata sekitar {number(profile_e['earthquake_count'], 1)} kejadian per kotak dalam tiga bulan; kedalaman sekitar {number(profile_e['avg_depth'])} km; magnitudo sekitar {number(profile_e['avg_magnitude'], 1)}. |
| F | Kejadian relatif lebih sering: rata-rata sekitar {number(profile_f['earthquake_count'], 1)} per kotak dalam tiga bulan; kedalaman sekitar {number(profile_f['avg_depth'])} km. |
| B | Rata-rata sumber jauh lebih dalam, sekitar {number(profile_b['avg_depth'])} km. |

**Kelompok dibentuk algoritma; nama penjelasannya dibuat setelah kita membaca hasil.** Peneliti tetap memilih data, fitur, dan aturan model, sehingga keputusan penelitian ikut memengaruhi hasil. Huruf A sampai F hanya pengenal, bukan peringkat bahaya.

Aturan nama dapat dibaca pada {notebook(4, 'notebook 04 bagian 6-7', 6)} dan {notebook(5, 'notebook 05 bagian 5-6', 5)}. Contoh E, F, dan B di atas hanya sebagian kelompok; tabel lengkap ada di notebook tersebut.

### Langkah 7: Mengikuti perubahan kotak yang sama

Kita menyusun riwayat kelompok setiap kotak selama **20 triwulan pada 2021-2025**. Dari riwayat itu dihitung jumlah pergantian dan lama bertahan di kelompok yang sama.

**Hasil tahap ini:** tabel perjalanan tiap wilayah, matriks perpindahan antarkelompok, dan ukuran kestabilan. Hanya dua triwulan berurutan yang sama-sama mempunyai kelompok dapat dibandingkan. Nol kejadian atau tidak masuk kelompok tidak otomatis dihitung sebagai pergantian profil.

Notebook: {notebook(7, '07 Temporal analysis')}.

### Langkah 8: Memeriksa apakah hasil mudah berubah

Kita mencoba pengaturan lain, termasuk kotak 25, 50, dan 100 km serta jumlah kejadian minimum per kotak. Tujuannya melihat apakah kesimpulan terlalu bergantung pada pilihan pengaturan.

**Hasil tahap ini:** K-Means cukup konsisten ketika titik awal perhitungannya diubah, tetapi kelompok berubah cukup banyak ketika ukuran kotak atau syarat jumlah kejadian diubah. Artinya, hasil belum kuat untuk semua pengaturan.

Notebook: {notebook(8, '08 Stability')}.

### Langkah 9: Menguji pola historis pada 2026

Kelompok dan skala fitur yang sudah dibentuk dari 2021-2025 dipakai kembali untuk data 2026. Jadi, model tidak dibentuk ulang memakai 2026.

Pada Januari-September 2026 terdapat **{number(len(eligible_test))} catatan kotak-triwulan yang layak**. K-Means memberi semuanya kelompok terdekat. Aturan pencocokan DBSCAN menemukan satu kelompok yang cocok untuk **{number(n_matched)} catatan**, belum cocok untuk **{number(n_no_match)}**, dan ambigu untuk **{number(n_ambiguous)}**.

**Hasil tahap ini:** kita mengetahui pengamatan baru mana yang masih cocok dengan pola historis dan mana yang perlu diperiksa lebih lanjut. Belum cocok tidak otomatis berarti kelompok baru atau kondisi berbahaya.

Q1-Q3 2026 merupakan periode lengkap. Q4 baru sampai sebelum 26 Oktober 2026 sehingga dipisahkan sebagai periode belum lengkap.

Notebook: {notebook(9, '09 Testing 2026')}.

### Langkah 10: Menampilkan hasil di peta

Label kelompok dikembalikan ke kotak wilayah dan periodenya. Karena lokasi kotaknya tetap, kita dapat memilih periode berbeda dan memeriksa wilayah yang sama.

**Hasil tahap ini:** peta kelompok aktivitas per periode dan peta berapa kali kelompok suatu wilayah berganti. Tombol **Sembunyikan grid / Tampilkan grid** membantu melihat daratan dengan atau tanpa kotak hasil analisis.

Notebook: {notebook(10, '10 GIS mapping')}. {notebook(0, 'Notebook 00')} menjadi panduan menjalankan seluruh tahapan.

## 5. Contoh nyata: mengikuti satu wilayah

Contoh ini memakai kotak berkode **5_-13**, berpusat sekitar 120,49 BT dan -8,15 LS. Semua angka kejadian di bawah adalah gempa M >=3 yang tercatat dalam katalog.

| Periode | Jumlah kejadian | Hasil K-Means |
| --- | --- | --- |
{examples}

Cara membacanya:

- Pada Januari-Maret 2025, wilayah ini mempunyai 10 kejadian dan masuk Kelompok F yang relatif lebih sering.
- Pada April-Juni, hasilnya berganti ke Kelompok E, lalu tetap di E sampai akhir 2025.
- Pada Januari-Maret 2026, model historis masih menempatkannya di Kelompok E.

**Inilah bentuk keluaran yang ingin kita pahami: perjalanan ciri aktivitas suatu wilayah.** Nama kelompok tidak ditentukan hanya dari jumlah gempa. Model juga memakai magnitudo dan kedalaman; enam kejadian tidak otomatis berarti Kelompok F.

Dari contoh ini kita belum menyimpulkan wilayah menjadi lebih aman atau lebih berbahaya. Kita memeriksa ciri apa yang berubah, berapa banyak event yang mendukungnya, dan apakah hasilnya bertahan saat pengaturan diuji.

## 6. Empat peta dan cara memakainya

| Peta | Gunanya |
| --- | --- |
| [Kelompok K-Means](maps/kmeans_profiles_by_quarter.html) | Melihat kelompok aktivitas tiap kotak pada periode terpilih. |
| [Kelompok DBSCAN](maps/dbscan_profiles_by_quarter.html) | Melihat hasil metode lain, termasuk catatan yang belum masuk atau belum cocok dengan kelompok. |
| [Perubahan K-Means](maps/kmeans_profile_changes_2021_2025.html) | Melihat jumlah pergantian kelompok historis pada tiap kotak. |
| [Perubahan DBSCAN](maps/dbscan_profile_changes_2021_2025.html) | Melihat pergantian berdasarkan kelompok DBSCAN. |

Mulailah dari **peta K-Means**, pilih satu periode, lalu klik satu kotak. Popup menunjukkan data kotak itu. Bandingkan dengan periode lain pada lokasi yang sama. Peta perubahan dapat dibuka sesudah itu untuk melihat riwayat secara ringkas.

Angka legenda adalah rata-rata kelompok historis. Angka pada kotak yang diklik adalah data wilayah pada periode pilihan, sehingga keduanya tidak harus sama.

Kotak dengan angka **0** berarti tidak ada event di atas ambang pada periode itu, tetapi wilayah tersebut masuk panel penelitian. Wilayah **tanpa kotak** tidak termasuk panel pada periode tersebut. Itu menjelaskan mengapa sebagian area tampak kosong; grid yang ditampilkan belum menutupi seluruh cakupan. Kedua tampilan itu tidak menentukan keamanan wilayah.

## 7. Hasil kita sekarang dan hasil akhir yang masih perlu diperkuat

**Yang sudah dihasilkan:** data bersih, kelompok aktivitas, penjelasan ciri kelompok, riwayat perubahan, hasil pengujian 2026, serta peta untuk membacanya. Semua 11 notebook sudah pernah dijalankan dengan berhasil.

**Yang belum dapat dinyatakan kuat:** bahwa kelompok tersebut selalu stabil, setiap pergantian menunjukkan perubahan seismik yang nyata, atau enam kelompok merupakan satu-satunya pembagian yang benar.

Masalah terbesarnya adalah sekitar **{number(low_support_fraction*100, 1)}% catatan training hanya berisi 1-4 kejadian**. Jika sebuah ringkasan hanya berasal dari satu atau dua gempa, cirinya mudah berubah pada periode berikutnya. Hasil juga cukup dipengaruhi ukuran kotak. Kelengkapan pencatatan katalog masih perlu ditinjau.

Karena itu, **hasil akhir penelitian yang diharapkan adalah penjelasan pola dan perubahan yang disertai bukti kecukupan data serta kestabilannya**. Sekarang kita sudah mempunyai bahan analisis dan alat penyajian. Kesimpulan ilmiahnya harus mengikuti kekuatan bukti tersebut, termasuk jika hasil menunjukkan beberapa pola belum stabil.

## 8. Gambaran keluaran yang dapat dipahami pembaca laporan

Pembaca seharusnya bisa mengikuti penjelasan seperti ini:

> Kami menemukan beberapa kelompok wilayah-periode dengan ciri aktivitas gempa yang mirip. Kelompoknya dijelaskan menggunakan jumlah kejadian, magnitudo, dan kedalaman. Riwayat setiap wilayah menunjukkan kapan kelompoknya bertahan atau berganti. Model historis lalu diterapkan pada data 2026 untuk melihat kecocokan pada periode baru. Sebagian hasil masih sensitif terhadap sedikitnya kejadian dan ukuran wilayah, sehingga kesimpulannya dibatasi pada pola aktivitas yang tercatat dalam katalog.

Keluaran utamanya adalah **kelompok aktivitas yang dapat dijelaskan, riwayat perubahan wilayah, dan hasil kecocokan pada periode baru**. Peta, tabel, dan skor dipakai bersama untuk menunjukkan serta menilai hasil tersebut.

Jika ingin mengecek seluruh jumlah data, tabel kandidat model, skor, dan file sumbernya, lanjutkan ke [laporan teknis dan angka lengkap](LAPORAN_TEKNIS_DAN_ANGKA.md). Dokumen utama ini sengaja menekankan makna proses dan hasilnya.
'''
    output = ROOT / "LAPORAN_ALUR_DAN_HASIL_PENELITIAN.md"
    output.write_text(content, encoding="utf-8")
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content)
    for target in links:
        relative, _, anchor = unquote(target).partition("#")
        path = ROOT / relative
        assert path.exists(), target
        if anchor:
            nb = json.loads(path.read_text(encoding="utf-8"))
            assert any(f'id="{anchor}"' in "".join(cell["source"]) for cell in nb["cells"]), target
    assert all(path in content for path in NOTEBOOKS.values())
    assert len(example) == 4 and len(eligible_test) == 2625
    receipt = {"status":"PASS", "report":output.name, "technical_report":"LAPORAN_TEKNIS_DAN_ANGKA.md",
               "word_count":len(content.split()), "links_checked":len(links), "all_notebooks_referenced":True,
               "example_grid":"5_-13", "sha256":hashlib.sha256(output.read_bytes()).hexdigest()}
    (ROOT / "reports/validation/project_explanation.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    build()
