"""Isi penjelasan EDA; hasil dihitung dari artefak proyek saat notebook dijalankan."""

RESULTS = '''annual, thresholds = eda(ROOT)
_, panel, train, _, _, _ = load_prepared(ROOT)
clean = pd.read_csv(ROOT / "data/processed/events_clean.csv")
historic = clean.loc[clean["year"].between(2021, 2025)].copy()
used = historic.loc[historic["magnitude"].ge(cfg["min_magnitude"])].copy()
jumlah = lambda value: f"{int(value):,}".replace(",", ".")
angka = lambda value: f"{value:.2f}".replace(".", ",")
display(Markdown(f"""### Data apa yang kita periksa?

EDA berarti **mengenali isi dan kondisi data sebelum membuat kelompok**.

| Data 2021-2025 | Jumlah | Satu baris berarti |
| --- | ---: | --- |
| Setelah cleaning, sebelum batas magnitudo | {jumlah(len(historic))} | Satu kejadian gempa |
| Memenuhi magnitudo minimal {cfg['min_magnitude']:g} | {jumlah(len(used))} | Satu kejadian gempa |
| Data training yang layak untuk clustering | {jumlah(len(train))} | Satu kotak selama satu triwulan |

Kita memeriksa katalog sebelum batas magnitudo untuk memahami pencatatannya,
kemudian memeriksa data yang benar-benar masuk penelitian.
**Data 2026 tidak disertakan dalam EDA ini.**
"""))
annual_view = annual[["n_events", "min_magnitude", "median_magnitude"]].copy()
annual_view["Lolos M minimum"] = used.groupby("year").size().reindex(annual_view.index, fill_value=0)
annual_view = annual_view.rename_axis("Tahun").reset_index().rename(columns={
    "n_events": "Gempa setelah cleaning", "min_magnitude": "M terkecil tercatat", "median_magnitude": "M tengah (median)"})
display(annual_view)
fig, ax = plt.subplots(figsize=(10, 4))
annual_view.set_index("Tahun")[["Gempa setelah cleaning", "Lolos M minimum"]].plot.bar(ax=ax, rot=0, color=["#999999", "#0072B2"])
ax.set(title="Jumlah gempa tercatat per tahun: sebelum dan sesudah batas magnitudo", xlabel="Tahun", ylabel="Jumlah kejadian gempa")
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/eda_annual_readable.png", dpi=160)
plt.show()
plt.close(fig)
year_max = used.groupby("year").size().idxmax()
display(Markdown(f"""**Cara membaca:** pilih tahun, lalu bandingkan batang abu-abu (setelah cleaning) dengan biru (memenuhi batas magnitudo).
Median adalah nilai tengah setelah data diurutkan, berbeda dari rata-rata.

**Yang terlihat:** {year_max} memiliki kejadian tercatat terbanyak pada data M minimal {cfg['min_magnitude']:g}, yaitu {jumlah(used.groupby('year').size().max())} kejadian.
Ini menjelaskan isi katalog. Perubahan pencatatan atau cakupan sumber juga dapat memengaruhi jumlahnya,
sehingga kita belum dapat menyimpulkan wilayah menjadi lebih berbahaya.
"""))
assert len(used) == train["earthquake_count"].sum()
'''

CORRELATION = '''feature_names = {
    "log_frequency": "Frekuensi (skala log)", "avg_magnitude": "Rata-rata magnitudo",
    "max_magnitude": "Magnitudo terbesar", "avg_depth": "Rata-rata kedalaman",
    "shallow_fraction": "Proporsi gempa dangkal"}
corr = pd.read_csv(ROOT / "reports/evaluation/feature_correlation.csv", index_col=0)
display(corr.rename(index=feature_names, columns=feature_names).round(2))
fig, ax = plt.subplots(figsize=(9, 6))
image = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
labels = [feature_names[key] for key in corr.columns]
ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
ax.set_yticks(range(len(labels)), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        value = corr.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > .6 else "black")
fig.colorbar(image, ax=ax, label="Hubungan berlawanan (-1) sampai searah (+1)")
ax.set_title("Hubungan lima ciri pada kotak-triwulan training")
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/eda_correlation_readable.png", dpi=160)
plt.show()
plt.close(fig)
display(Markdown(f"""**Cara membaca:** angka mendekati +1 berarti dua ciri cenderung meningkat bersama;
angka mendekati -1 berarti arahnya berlawanan. Angka dekat 0 berarti hubungan lurusnya lemah.
Angka 1 pada diagonal hanya membandingkan ciri dengan dirinya sendiri.

**Hasil pada data ini:** rata-rata magnitudo dan magnitudo terbesar memiliki hubungan {angka(corr.loc['avg_magnitude', 'max_magnitude'])}.
Rata-rata kedalaman dan proporsi gempa dangkal memiliki hubungan {angka(corr.loc['avg_depth', 'shallow_fraction'])}.
Pada data ini, kotak dengan rata-rata kedalaman lebih besar cenderung memiliki proporsi gempa dangkal lebih kecil.

**Artinya untuk model:** sebagian ciri membawa informasi yang saling berkaitan.
Menyetarakan skala angka tidak menghilangkan hubungan ini.
Korelasi juga tidak membuktikan hubungan sebab-akibat.
"""))
'''

DISTRIBUTIONS = '''display(Markdown(f"Grafik berikut memakai **{jumlah(len(used))} kejadian dengan M minimal {cfg['min_magnitude']:g} pada 2021-2025**. Satu kejadian dapat dihitung satu kali dalam setiap grafik."))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
used["magnitude"].hist(ax=axes[0], bins=30, color="#0072B2", edgecolor="white")
axes[0].set(title="Magnitudo gempa yang masuk penelitian", xlabel="Magnitudo", ylabel="Jumlah kejadian gempa")
used["depth"].hist(ax=axes[1], bins=40, color="#0072B2", edgecolor="white")
axes[1].axvline(cfg["shallow_depth_km"], color="#9C3D10", linestyle="--", label=f"Batas dangkal: {cfg['shallow_depth_km']:g} km")
axes[1].set(title="Kedalaman gempa yang masuk penelitian", xlabel="Kedalaman (km)", ylabel="Jumlah kejadian gempa")
axes[1].legend()
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/eda_distributions_readable.png", dpi=160)
plt.show()
plt.close(fig)
valid_depth = used["depth"].dropna()
display(Markdown(f"""**Cara membaca:** batang yang lebih tinggi berarti lebih banyak gempa dalam rentang angka tersebut.
Grafik kiri menunjukkan magnitudo; grafik kanan menunjukkan kedalaman. Ketinggian batang adalah jumlah gempa, bukan tingkat bahaya.

**Ringkasan data yang masuk penelitian:** median magnitudo **{angka(used['magnitude'].median())}**,
median kedalaman **{angka(valid_depth.median())} km**.
Dari {jumlah(len(valid_depth))} kejadian dengan kedalaman tersedia,
**{angka(valid_depth.lt(cfg['shallow_depth_km']).mean() * 100)}%** berada pada kedalaman kurang dari {cfg['shallow_depth_km']:g} km.
"""))
quarters = used.groupby("period").size().reindex(pd.period_range("2021Q1", "2025Q4", freq="Q").astype(str), fill_value=0)
fig, ax = plt.subplots(figsize=(12, 4))
quarters.plot.bar(ax=ax, color="#0072B2", rot=60)
ax.set(title=f"Jumlah gempa M minimal {cfg['min_magnitude']:g} per triwulan", xlabel="Tahun dan triwulan", ylabel="Jumlah kejadian gempa")
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/eda_quarters_readable.png", dpi=160)
plt.show()
plt.close(fig)
display(Markdown("**Cara membaca:** Q1 = Januari-Maret, Q2 = April-Juni, Q3 = Juli-September, Q4 = Oktober-Desember. Grafik ini menunjukkan jumlah kejadian seluruh wilayah penelitian, bukan perubahan kelompok pada satu kotak. Perubahan kelompok per kotak diperiksa di notebook 07."))
'''

THRESHOLDS = '''threshold_view = thresholds.pivot(index="year", columns="min_magnitude", values="event_count")
threshold_view.columns = [f"M minimal {value:g}" for value in threshold_view.columns]
display(threshold_view.rename_axis("Tahun").reset_index())
totals = thresholds.groupby("min_magnitude")["event_count"].sum()
display(Markdown(f"""**Cara membaca:** pada tahun yang sama, pilih kolom batas magnitudo yang berbeda.
M minimal 3 berarti magnitudo 3 ke atas, bukan hanya tepat 3.

Dengan batas utama **M minimal {cfg['min_magnitude']:g}**, data historis yang tersisa adalah **{jumlah(len(used))} kejadian**.
Jika batas dinaikkan menjadi M minimal 4, tersisa **{jumlah(totals.loc[4])} kejadian**.
Makin tinggi batasnya, makin sedikit data yang digunakan.

**Keputusan penelitian:** batas utama tetap mengikuti konfigurasi.
Tabel ini memperlihatkan pengaruh batas tersebut terhadap jumlah data;
tabel ini belum membuktikan bahwa semua gempa di atas batas itu tercatat lengkap.
Pengaruh perubahan batas terhadap hasil kelompok diuji di notebook 08.
"""))
'''

SUPPORT = '''historical_panel = panel.loc[panel["split"].eq("train")]
empty_count = int(historical_panel["state"].eq("NO_EVENTS").sum())
few = train["earthquake_count"].between(1, cfg["low_support_count"] - 1)
display(Markdown(f"""### Berapa banyak data di dalam setiap kotak?

| Ringkasan historis | Jumlah |
| --- | ---: |
| Kotak geografis yang pernah aktif | {jumlah(historical_panel['grid_id'].nunique())} |
| Triwulan yang diperiksa | {jumlah(historical_panel['period'].nunique())} |
| Seluruh baris kotak-triwulan | {jumlah(len(historical_panel))} |
| Baris tanpa kejadian pada triwulan tersebut | {jumlah(empty_count)} |
| Baris yang layak masuk model | {jumlah(len(train))} |
| Baris training dengan 1-{cfg['low_support_count'] - 1} gempa | {jumlah(few.sum())} ({angka(few.mean() * 100)}%) |

**Contoh:** satu kotak dengan 10 gempa pada Januari-Maret adalah satu baris model dengan jumlah gempa 10.
Kotak yang sama pada April-Juni menjadi baris lain.

**Temuan yang perlu diperhatikan:** banyak baris hanya merangkum sedikit kejadian.
Jika sebuah baris hanya memiliki satu gempa, rata-rata dan maksimum magnitudonya sama dengan magnitudo gempa itu.
Karena itu perubahan profil dapat dipengaruhi sedikit kejadian, dan tidak langsung membuktikan perubahan kondisi tektonik.
Kepekaan hasil terhadap jumlah kejadian diuji di notebook 08.
"""))
support = pd.cut(train["earthquake_count"], bins=[0, 1, 4, 9, float("inf")], labels=["1 gempa", "2-4 gempa", "5-9 gempa", "10 gempa atau lebih"])
display(support.value_counts(sort=False).rename_axis("Gempa dalam satu kotak-triwulan").reset_index(name="Jumlah baris training"))
display(train[["grid_id", "period", "earthquake_count", "avg_magnitude", "max_magnitude", "avg_depth"]].head(5).rename(columns={
    "grid_id": "ID kotak", "period": "Triwulan", "earthquake_count": "Jumlah gempa", "avg_magnitude": "Rata-rata M", "max_magnitude": "M terbesar", "avg_depth": "Rata-rata kedalaman (km)"}).round(2))
'''

QUALITY = '''missing = historic[["time", "latitude", "longitude", "magnitude", "depth"]].isna().sum()
display(missing.rename_axis("Kolom").reset_index(name="Nilai kosong setelah cleaning").replace({"Kolom": {"time": "Waktu", "latitude": "Lintang", "longitude": "Bujur", "magnitude": "Magnitudo", "depth": "Kedalaman"}}))
diagnostic = pd.read_csv(ROOT / "reports/evaluation/completeness_diagnostic.csv")
display(diagnostic[["year", "max_curvature_histogram_mode", "analysis_threshold"]].rename(columns={
    "year": "Tahun", "max_curvature_histogram_mode": "Magnitudo paling sering (bin 0,1)", "analysis_threshold": "Batas M penelitian"}).round(1))
display(Markdown("**Cara membaca:** magnitudo paling sering adalah puncak histogram dengan rentang 0,1. Perbedaan antartahun menjadi alasan untuk memeriksa konsistensi katalog. Angka ini bukan bukti bahwa katalog sudah lengkap. Tidak ada nilai kosong juga tidak berarti semua gempa yang terjadi berhasil dicatat."))
types = pd.read_csv(ROOT / "reports/evaluation/magnitude_types.csv")
display(types.pivot_table(index="year", columns="magnitude_type", values="n", aggfunc="sum", fill_value=0).rename_axis(index="Tahun", columns="Jenis M dari sumber").reset_index())
display(Markdown("**Jenis magnitudo:** kolom `m`, `mb`, `ml`, `mla`, dan `mw` adalah kode yang disediakan katalog. Data ini memuat beberapa jenis magnitudo dan belum disetarakan ke satu skala Mw. Karena itu, istilah yang digunakan dalam penelitian adalah magnitudo katalog, bukan seluruhnya Mw."))
'''

SUMMARY = '''display(Markdown(f"""### Apa yang kita ketahui setelah EDA?

1. Ada **{jumlah(len(used))} kejadian historis** yang memenuhi batas M minimal {cfg['min_magnitude']:g}.
   Kejadian ini diringkas menjadi **{jumlah(len(train))} baris training kotak-triwulan**.
2. Jumlah tercatat berbeda antartahun. Itu adalah pola dalam katalog, belum penjelasan penyebab fisiknya.
3. Beberapa ciri saling berkaitan, terutama rata-rata dan maksimum magnitudo, serta kedalaman dan proporsi dangkal.
4. **{angka(few.mean() * 100)}% baris training** hanya memiliki 1-{cfg['low_support_count'] - 1} gempa,
   sehingga profil dan perubahannya perlu ditafsirkan hati-hati.
5. Pilihan batas magnitudo memengaruhi data yang tersisa. Kelengkapan katalog dan perbedaan jenis magnitudo masih menjadi batas interpretasi.

**Langkah berikutnya:** notebook 04 dan 05 mengelompokkan kotak-triwulan berdasarkan lima ciri aktivitas.
EDA belum menghasilkan kelompok dan belum menghasilkan peta tingkat risiko.
Hasil clustering serta uji sensitivitas dibaca bersama temuan EDA ini.

### Buka tabel lengkap
"""))
for filename in ["catalogue_by_year.csv", "magnitude_threshold_audit.csv", "feature_correlation.csv", "completeness_diagnostic.csv", "magnitude_types.csv"]:
    display(FileLink(str(ROOT / "reports/evaluation" / filename)))
'''
