"""Tabel dan grafik evaluasi model; seluruh hasil memakai training historis."""

SUMMARY = '''comparison = evaluate(ROOT)
evaluation_cfg, evaluation_panel, evaluation_train, evaluation_X, _, _ = load_prepared(ROOT)
km_model, km_metadata = load_model(ROOT, "kmeans")
db_model, db_metadata = load_model(ROOT, "dbscan")
assert len(km_model.labels_) == len(db_model.labels_) == len(evaluation_train)
model_labels = {"kmeans": km_model.labels_, "dbscan": db_model.labels_}
model_names = {"kmeans": "K-Means", "dbscan": "DBSCAN"}
model_colors = {"kmeans": "#0072B2", "dbscan": "#B44A00"}
whole = comparison.loc[comparison["evaluation_scope"].str.startswith("seluruh")].set_index("model").loc[["kmeans", "dbscan"]]
fair = comparison.loc[comparison["evaluation_scope"].str.startswith("subset identik")].set_index("model").loc[["kmeans", "dbscan"]]
common = db_model.labels_ >= 0
assert fair["n_rows"].eq(common.sum()).all()
assert fair["n_assigned"].eq(common.sum()).all()
jumlah = lambda value: f"{int(value):,}".replace(",", ".")
angka = lambda value, digits=4: f"{value:.{digits}f}".replace(".", ",")
display(Markdown(f"""### Apa yang dibandingkan?

Kita ingin mengetahui **seberapa terpisah kelompok yang terbentuk**, berapa data yang masuk kelompok,
dan seperti apa ciri serta ukuran kelompoknya.

- Kedua model dibentuk memakai **{jumlah(len(evaluation_train))} baris training 2021-2025** dan lima fitur standar yang sama.
- Satu baris berarti satu kotak selama satu triwulan, bukan satu kejadian gempa.
- K-Means memberi kelompok kepada seluruh baris tersebut. DBSCAN dapat menyisakan noise.
- Untuk membandingkan skor secara adil, kita memakai **{jumlah(common.sum())} baris yang sama**, yaitu baris yang masuk kelompok DBSCAN.
  Label kedua model tetap hasil training semula; tidak ada pelatihan ulang pada subset ini.
- Data 2026 tidak digunakan di sini; hasil pengujiannya ada di notebook 09.

### Ringkasan skor pada baris yang sama

| Ukuran | K-Means | DBSCAN | Cara membaca |
| --- | ---: | ---: | --- |
| Silhouette | {angka(fair.loc['kmeans', 'silhouette'])} | {angka(fair.loc['dbscan', 'silhouette'])} | Makin tinggi, pemisahan menurut metrik ini lebih baik |
| Davies-Bouldin Index (DBI) | {angka(fair.loc['kmeans', 'davies_bouldin'])} | {angka(fair.loc['dbscan', 'davies_bouldin'])} | Makin kecil, kelompok lebih kompak dan terpisah menurut metrik ini |
| Baris yang dibandingkan | {jumlah(fair.loc['kmeans', 'n_rows'])} | {jumlah(fair.loc['dbscan', 'n_rows'])} | Identik untuk kedua model |

**Cara membaca hasil:** pada subset dan dua metrik ini, K-Means menunjukkan pemisahan kelompok lebih baik.
Ini belum membuktikan K-Means paling benar secara seismologi atau unggul pada semua tujuan.
Skor tersebut juga bukan akurasi prediksi gempa.
Silhouette memakai sampel tetap maksimal {jumlah(evaluation_cfg['silhouette_sample_size'])} baris, seed 42;
DBI memakai seluruh baris yang dibandingkan. Noise tidak dimasukkan ke skor jarak DBSCAN.
"""))
'''

SCORES = '''fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, metric, title in zip(axes, ["silhouette", "davies_bouldin"],
                            ["Silhouette: lebih tinggi lebih baik", "Davies-Bouldin: lebih rendah lebih baik"]):
    values = fair[metric].to_numpy()
    bars = ax.bar([model_names[key] for key in fair.index], values, color=[model_colors[key] for key in fair.index], width=.55)
    ax.bar_label(bars, labels=[angka(value) for value in values], padding=6)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Skor")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    if metric == "silhouette":
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="#444444", linewidth=.8)
    else:
        ax.set_ylim(0, max(values) * 1.25)
fig.suptitle(f"Perbandingan pada {jumlah(common.sum())} baris training yang sama", fontsize=14)
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/evaluation_scores_common_subset.png", dpi=160)
plt.show()
plt.close(fig)
display(Markdown("**Cara membaca:** kiri, cari batang yang lebih tinggi; kanan, cari batang yang lebih rendah. Sumbu kedua grafik berbeda karena makna skornya berbeda. Ini evaluasi pada subset identik, bukan seluruh baris K-Means dibandingkan dengan sebagian baris DBSCAN."))
display(Markdown("### Angka lengkap untuk dua cakupan evaluasi"))
details = comparison[["model", "evaluation_scope", "n_rows", "n_assigned", "silhouette", "davies_bouldin"]].copy()
details["model"] = details["model"].map(model_names)
details["evaluation_scope"] = details["evaluation_scope"].map(lambda text: "Subset identik; kedua model pada baris yang sama" if text.startswith("subset identik") else "Seluruh training; skor DBSCAN hanya pada anggota kelompok")
display(details.rename(columns={"model": "Model", "evaluation_scope": "Cakupan", "n_rows": "Baris masukan", "n_assigned": "Baris masuk kelompok", "silhouette": "Silhouette", "davies_bouldin": "DBI"}).round(4))
display(Markdown("Nilai Silhouette K-Means pada seluruh training dapat berbeda dari subset identik karena baris yang dinilai berbeda. Skor seleksi pada notebook 04/05 memakai rata-rata tiga sampel, sedangkan evaluasi ini memakai satu sampel tetap. Tidak ada model yang dilatih ulang di bagian ini."))
'''

COVERAGE = '''assigned_counts = whole["n_assigned"].to_numpy(dtype=int)
noise_counts = whole["n_rows"].to_numpy(dtype=int) - assigned_counts
assigned_percent = assigned_counts / len(evaluation_train) * 100
noise_percent = noise_counts / len(evaluation_train) * 100
fig, ax = plt.subplots(figsize=(10, 4))
names = [model_names[key] for key in whole.index]
ax.barh(names, assigned_percent, color="#0072B2", label="Masuk kelompok")
ax.barh(names, noise_percent, left=assigned_percent, color="#777777", label="Noise")
for i, (assigned, noise) in enumerate(zip(assigned_counts, noise_counts)):
    ax.text(assigned_percent[i] / 2, i, f"{jumlah(assigned)} baris ({angka(assigned_percent[i], 2)}%)", ha="center", va="center", color="white")
    if noise:
        ax.text(assigned_percent[i] + noise_percent[i] / 2, i, jumlah(noise) + chr(10) + angka(noise_percent[i], 2) + "%", ha="center", va="center", color="white", fontsize=9)
ax.invert_yaxis()
ax.set(xlim=(0, 100), xlabel="Persentase seluruh baris training layak (%)", title="Berapa banyak data yang masuk kelompok?")
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/evaluation_assignment_coverage.png", dpi=160)
plt.show()
plt.close(fig)
display(Markdown(f"""**Cara membaca:** panjang total setiap batang adalah {jumlah(len(evaluation_train))} baris training layak.
K-Means selalu memberi kelompok; DBSCAN menempatkan {jumlah(noise_counts[1])} baris sebagai noise.
Karena itu 100% penugasan K-Means tidak berarti akurasi 100%, dan noise DBSCAN tidak otomatis berarti data salah.
Kotak tanpa kejadian tidak termasuk penyebut grafik ini karena tidak masuk training clustering.
"""))
'''

SIZES = '''fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True)
cluster_counts = {}
for ax, method in zip(axes, ["kmeans", "dbscan"]):
    unique, counts = np.unique(model_labels[method], return_counts=True)
    ordered = sorted(zip(unique, counts), key=lambda item: (item[0] < 0, item[0]))
    cluster_counts[method] = ordered
    names = ["Noise" if label < 0 else f"Kelompok {chr(65 + int(label))} / P{int(label)}" for label, _ in ordered]
    values = [count for _, count in ordered]
    bars = ax.barh(names, values, color=["#777777" if label < 0 else model_colors[method] for label, _ in ordered])
    ax.bar_label(bars, labels=[jumlah(value) for value in values], padding=4)
    ax.invert_yaxis()
    ax.set_title(model_names[method])
    ax.set_xlabel("Jumlah baris kotak-triwulan")
maximum_count = max(count for entries in cluster_counts.values() for _, count in entries)
axes[0].set_xlim(0, maximum_count * 1.2)
fig.suptitle("Ukuran kelompok pada seluruh training; noise terpisah", fontsize=14)
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/evaluation_cluster_sizes.png", dpi=160)
plt.show()
plt.close(fig)
display(Markdown(f"""**Cara membaca:** lihat jumlah anggota masing-masing kelompok dan apakah ada kelompok sangat kecil atau dominan.
K-Means membentuk {int(whole.loc['kmeans', 'n_clusters'])} kelompok; DBSCAN membentuk {int(whole.loc['dbscan', 'n_clusters'])} kelompok di luar noise.
Kelompok terkecil DBSCAN memiliki {jumlah(whole.loc['dbscan', 'smallest_cluster_size'])} baris.
Huruf A pada K-Means tidak berarti kelompok yang sama dengan A pada DBSCAN.
Angka/huruf adalah pengenal, bukan peringkat bahaya. Jumlah kelompok lebih banyak juga tidak otomatis lebih baik.
"""))
'''

PROFILES = '''profile_frames = {method: pd.read_csv(ROOT / f"models/{method}/profiles.csv") for method in ["kmeans", "dbscan"]}
profile_columns = ["earthquake_count", "avg_magnitude", "avg_depth"]
profile_titles = ["Gempa per kotak-triwulan", "Magnitudo rata-rata", "Kedalaman rata-rata (km)"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for row, method in enumerate(["kmeans", "dbscan"]):
    frame = profile_frames[method].loc[lambda data: data["profile"].ne("NOISE")].copy()
    names = frame["profile"].map(lambda code: chr(65 + int(code.rsplit("P", 1)[1])))
    for col, (feature, title) in enumerate(zip(profile_columns, profile_titles)):
        ax = axes[row, col]
        bars = ax.bar(names, frame[feature], color=model_colors[method])
        ax.bar_label(bars, labels=[angka(value, 1) for value in frame[feature]], padding=3, fontsize=8)
        max_value = max(profile_frames[m].loc[lambda data: data["profile"].ne("NOISE"), feature].max() for m in profile_frames)
        ax.set_ylim(0, max_value * 1.25)
        ax.set_title(f"{model_names[method]}: {title}", fontsize=11)
        ax.set_xlabel("Kelompok")
fig.suptitle("Apa ciri kelompok yang dihasilkan?", fontsize=15)
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/evaluation_profile_characteristics.png", dpi=160)
plt.show()
plt.close(fig)
display(Markdown("**Cara membaca:** bandingkan kolom yang sama karena satuannya sama. Kolom kiri menunjukkan rata-rata jumlah gempa, tengah magnitudo, kanan kedalaman. Angka merangkum rata-rata baris kotak-triwulan anggota kelompok, bukan setiap kejadian memiliki nilai yang sama. A–F K-Means dan A–G DBSCAN tidak berpasangan berdasarkan huruf. Grafik menampilkan tiga ciri dalam satuan asli; model tetap memakai lima fitur standar. Noise dikeluarkan dari grafik ciri kelompok ini; statistiknya tersedia pada tabel profil."))
'''

AGREEMENT = '''agreement_counts = pd.crosstab(pd.Series(km_model.labels_[common], name="K-Means"), pd.Series(db_model.labels_[common], name="DBSCAN"))
assert agreement_counts.to_numpy().sum() == common.sum()
ari = adjusted_rand_score(km_model.labels_[common], db_model.labels_[common])
fig, ax = plt.subplots(figsize=(10, 6))
image = ax.imshow(agreement_counts, cmap="Blues", aspect="auto", vmin=0)
for i in range(len(agreement_counts.index)):
    for j in range(len(agreement_counts.columns)):
        count = agreement_counts.iloc[i, j]
        ax.text(j, i, jumlah(count), ha="center", va="center", color="white" if count > agreement_counts.to_numpy().max() * .55 else "black")
ax.set_xticks(range(len(agreement_counts.columns)), [f"DBSCAN {chr(65 + int(label))}" for label in agreement_counts.columns])
ax.set_yticks(range(len(agreement_counts.index)), [f"K-Means {chr(65 + int(label))}" for label in agreement_counts.index])
ax.set(xlabel="Kelompok DBSCAN", ylabel="Kelompok K-Means", title=f"Ke mana baris yang sama dikelompokkan? ARI = {angka(ari)}")
fig.colorbar(image, ax=ax, label="Jumlah baris kotak-triwulan")
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/evaluation_partition_overlap.png", dpi=160)
plt.show()
plt.close(fig)
display(Markdown(f"""**Cara membaca:** pilih satu baris K-Means, lalu lihat ke kelompok DBSCAN mana anggotanya tersebar.
Sel lebih gelap berarti lebih banyak baris yang memiliki pasangan kelompok tersebut.
Grafik memakai {jumlah(common.sum())} baris yang sama dan tidak memasukkan noise DBSCAN.

**Ini bukan confusion matrix akurasi:** tidak ada kelompok jawaban benar.
Diagonal tidak mempunyai arti benar/salah karena label dua metode tidak berpasangan berdasarkan huruf.
ARI sebesar **{angka(ari)}** merangkum kesepakatan pengelompokan:
1 berarti partisinya sama meskipun nama label berbeda; sekitar 0 setara kesepakatan acak menurut ukuran ini.
Hasil ini menunjukkan kedua metode tidak menghasilkan pembagian yang sama, bukan salah satunya memiliki akurasi {angka(ari * 100, 2)}%.
"""))
'''

CONCLUSION = '''display(Markdown(f"""### Apa yang dapat disimpulkan?

| Pertanyaan | Hasil penelitian ini |
| --- | --- |
| Mana yang lebih terpisah menurut Silhouette dan DBI pada baris identik? | K-Means: Silhouette {angka(fair.loc['kmeans', 'silhouette'])}, DBI {angka(fair.loc['kmeans', 'davies_bouldin'])} |
| Apakah semua data harus masuk kelompok? | K-Means menugaskan seluruh baris; DBSCAN menyisakan {angka(whole.loc['dbscan', 'noise_fraction'] * 100, 2)}% noise |
| Apakah kedua metode membagi data dengan cara yang sama? | Tidak sama; kesepakatan ARI {angka(ari)} pada baris yang masuk kelompok DBSCAN |
| Apakah salah satu model sudah terbukti memprediksi gempa dengan benar? | Belum; penelitian ini tidak memiliki label kebenaran untuk menghitung akurasi |

**Penilaian yang dapat dipertanggungjawabkan:** K-Means memiliki pemisahan lebih baik pada dua metrik jarak yang digunakan.
DBSCAN tetap memberi sudut pandang tambahan melalui kelompok berbasis kepadatan dan noise,
tetapi pemisahannya masih lemah menurut Silhouette. Kedua metrik ini tidak menangkap seluruh sifat kelompok kepadatan DBSCAN.
Kita belum dapat menyatakan salah satu metode paling benar secara geologi hanya dari grafik ini.

Ukuran kelompok, ciri aktivitas, dan banyaknya kejadian yang mendasari tiap baris tetap perlu diperiksa.
Hasil juga dipengaruhi pemilihan fitur, batas magnitudo, ukuran grid, dan parameter.
Nama deskriptif dibuat setelah clustering; nama tersebut tidak mengubah anggota kelompok dan tidak menunjukkan tingkat risiko.

### Setelah evaluasi ini

- [Notebook 07](../07_temporal_analysis/07_temporal_analysis.ipynb): lihat perubahan kelompok setiap kotak sepanjang waktu.
- [Notebook 08](../08_stability/08_stability.ipynb): periksa kepekaan hasil terhadap parameter dan rancangan data.
- [Notebook 09](../09_testing_2026/09_testing_2026.ipynb): cocokkan aktivitas 2026 dengan kelompok historis.
- [Notebook 10](../10_gis_mapping/10_gis_mapping.ipynb): baca hasil pada peta.

### Tabel lengkap dan grafik untuk dibuka terpisah
"""))
display(FileLink(str(ROOT / "reports/evaluation/model_comparison.csv")))
display(FileLink(str(ROOT / "reports/evaluation/agreement.json")))
for filename in ["evaluation_scores_common_subset.png", "evaluation_assignment_coverage.png", "evaluation_cluster_sizes.png", "evaluation_profile_characteristics.png", "evaluation_partition_overlap.png"]:
    display(FileLink(str(ROOT / "reports/figures" / filename)))
'''
