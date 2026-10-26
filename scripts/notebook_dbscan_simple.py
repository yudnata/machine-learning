"""Alur grafik utama DBSCAN yang ringkas; rincian perhitungan tetap tersedia."""

RUN = '''from src.common import load_model

LATIH_ULANG = False
saved_files = [ROOT / f"models/dbscan/{name}" for name in ["model.joblib", "metadata.json", "parameter_search.csv", "training_labels.csv"]]
if LATIH_ULANG or not all(path.is_file() for path in saved_files):
    search, labels = fit_dbscan(ROOT)
    print("Pencarian parameter dan pelatihan DBSCAN selesai.")
else:
    search = pd.read_csv(ROOT / "models/dbscan/parameter_search.csv")
    labels = pd.read_csv(ROOT / "models/dbscan/training_labels.csv")
    print("Membaca hasil training tersimpan. Ubah LATIH_ULANG menjadi True untuk mengulang pencarian.")

evaluation_cfg, evaluation_panel, evaluation_train, evaluation_X, _, _ = load_prepared(ROOT)
dbscan_model, dbscan_metadata = load_model(ROOT, "dbscan")
candidate_results = search.copy()
selected_mask = (np.isclose(search["eps"], dbscan_metadata["eps"], rtol=1e-10, atol=1e-12)
                 & search["min_samples"].eq(dbscan_metadata["min_samples"]))
assert selected_mask.sum() == 1
selected_result = search.loc[selected_mask].iloc[0]
assert len(dbscan_model.labels_) == len(evaluation_X)
final_metrics = cluster_metrics(evaluation_X, dbscan_model.labels_, cfg["silhouette_sample_size"], seed=42)
assert final_metrics["n_assigned"] == selected_result["n_assigned"]
assert final_metrics["n_clusters"] == selected_result["n_clusters"]
angka = lambda value, digits=4: f"{value:.{digits}f}".replace(".", ",")
jumlah = lambda value: f"{int(value):,}".replace(",", ".")
noise_rows = final_metrics["n_rows"] - final_metrics["n_assigned"]
display(Markdown(f"""### Ikuti tiga langkah grafik

1. **Bagian 7:** lihat kurva k-distance untuk memahami jarak antarpola.
2. **Bagian 8:** bandingkan nilai eps dan skor, seperti grafik pilihan k pada K-Means.
3. **Bagian 9:** baca jumlah kelompok, noise, dan ringkasan hasilnya.

**Hasil tersimpan:** eps **{angka(dbscan_metadata['eps'], 3)}**, min_samples **{dbscan_metadata['min_samples']}**.
Model memakai **{jumlah(len(evaluation_train))} baris kotak-triwulan 2021-2025**, tanpa data testing 2026.
"""))
'''

DISTANCE = '''with threadpool_limits(limits=2):
    neighbor_distances = NearestNeighbors(n_neighbors=int(dbscan_metadata["min_samples"]), n_jobs=1).fit(evaluation_X).kneighbors(evaluation_X)[0][:, -1]
sorted_distances = np.sort(neighbor_distances)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(np.arange(1, len(sorted_distances) + 1), sorted_distances, color="#0072B2", linewidth=2)
ax.axhline(dbscan_metadata["eps"], color="#B44A00", linestyle="--", linewidth=1.8,
           label=f"eps terpilih dari pencarian = {angka(dbscan_metadata['eps'], 3)}")
ax.set(title=f"1. Kurva k-distance untuk min_samples = {dbscan_metadata['min_samples']}",
       xlabel="Baris kotak-triwulan, diurutkan dari jarak terkecil", ylabel="Jarak pada lima fitur standar")
ax.set_ylim(bottom=0)
ax.legend(loc="upper left")
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/dbscan_simple_k_distance.png", dpi=160)
plt.show()
plt.close(fig)
display(Markdown(f"""**Bacanya:** bagian kurva yang rendah menunjukkan baris dengan pola yang lebih dekat ke tetangganya.
Ketika kurva menanjak, baris-baris tersebut lebih jauh dari tetangganya.

Garis oranye menunjukkan **eps {angka(dbscan_metadata['eps'], 3)} yang dipilih dari pencarian parameter**.
Dalam penelitian ini, kandidat eps berasal dari beberapa kuantil jarak, kemudian diperiksa skor dan ukuran kelompoknya.
Garis ini bukan hasil deteksi titik siku otomatis. Angka jarak bukan kilometer.
Perhitungan tetangga mengikuti min_samples DBSCAN, termasuk baris itu sendiri.
"""))
'''

EPS_SCORES = '''simple_candidates = search.loc[search["min_samples"].eq(dbscan_metadata["min_samples"])].sort_values("eps")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, metric, title in zip(axes, ["silhouette", "davies_bouldin"],
                            ["Silhouette: cari yang lebih tinggi", "Davies-Bouldin: cari yang lebih rendah"]):
    ax.plot(simple_candidates["eps"], simple_candidates[metric], color="#0072B2", linewidth=1.5)
    valid = simple_candidates[metric].notna()
    accepted = simple_candidates["accepted"] & valid
    rejected = ~simple_candidates["accepted"] & valid
    ax.scatter(simple_candidates.loc[accepted, "eps"], simple_candidates.loc[accepted, metric], color="#0072B2", s=55, label="Lolos aturan")
    ax.scatter(simple_candidates.loc[rejected, "eps"], simple_candidates.loc[rejected, metric], color="#777777", marker="x", s=70, label="Tidak lolos aturan")
    ax.scatter(dbscan_metadata["eps"], selected_result[metric], color="#B44A00", s=160, marker="*", zorder=4, label="Terpilih")
    ax.annotate(angka(selected_result[metric]), (dbscan_metadata["eps"], selected_result[metric]), xytext=(8, 12), textcoords="offset points", color="#803700")
    ax.axvline(dbscan_metadata["eps"], color="#B44A00", linestyle="--", alpha=.5)
    ax.set(title=title, xlabel="eps yang dicoba", ylabel="Skor")
    ax.set_xticks(simple_candidates["eps"], [angka(value, 3) for value in simple_candidates["eps"]])
    ax.margins(x=.15, y=.25)
    eps_span = simple_candidates["eps"].max() - simple_candidates["eps"].min()
    ax.set_xlim(simple_candidates["eps"].min() - .1 * eps_span, simple_candidates["eps"].max() + .1 * eps_span)
    ax.grid(alpha=.2)
axes[0].legend(loc="lower right", fontsize=9)
fig.suptitle(f"2. Bandingkan eps pada min_samples = {dbscan_metadata['min_samples']}", fontsize=14)
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/dbscan_simple_eps_scores.png", dpi=160)
plt.show()
plt.close(fig)
display(Markdown(f"""**Bacanya:** sumbu bawah adalah eps yang dicoba. Bintang oranye adalah pilihan model.
Grafik kiri mencari skor lebih tinggi; grafik kanan mencari skor lebih rendah.

Grafik ini memperlihatkan **{len(simple_candidates)} kandidat dengan min_samples {dbscan_metadata['min_samples']}**, diambil dari seluruh {len(search)} percobaan.
Pilihan akhirnya tetap memakai seluruh pencarian, bukan hanya titik-titik ini.
Tanda silang abu-abu berarti tidak memenuhi aturan penelitian walaupun skornya dapat terlihat baik.
Kandidat yang hanya menghasilkan satu kelompok tidak memiliki skor Silhouette/DBI, sehingga tidak digambar sebagai titik skor.

Di antara kandidat yang lolos, dipilih rata-rata Silhouette tertinggi; jika sama, noise lebih kecil didahulukan.
Seluruh percobaan dan alasan penyaringannya dapat dibuka pada bagian 10.
"""))
'''

RESULT = '''unique_labels, counts = np.unique(dbscan_model.labels_, return_counts=True)
ordered = sorted(zip(unique_labels, counts), key=lambda item: (item[0] < 0, item[0]))
names = ["Noise" if label < 0 else f"Kelompok {chr(65 + int(label))}" for label, _ in ordered]
values = [count for _, count in ordered]
fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.bar(names, values, color=["#777777" if label < 0 else "#0072B2" for label, _ in ordered])
ax.bar_label(bars, labels=[jumlah(value) for value in values], padding=4)
ax.set_ylim(0, max(values) * 1.18)
ax.set(title="3. Hasil kelompok DBSCAN", xlabel="Kelompok hasil algoritma", ylabel="Jumlah baris kotak-triwulan")
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/dbscan_simple_group_sizes.png", dpi=160)
plt.show()
plt.close(fig)
display(Markdown(f"""**Bacanya:** batang lebih tinggi berarti kelompok memiliki lebih banyak anggota.
Noise ditampilkan abu-abu dan bukan kelompok tambahan.

| Hasil utama | Nilai |
| --- | ---: |
| eps terpilih | {angka(dbscan_metadata['eps'], 3)} |
| min_samples | {dbscan_metadata['min_samples']} |
| Kelompok terbentuk | {jumlah(final_metrics['n_clusters'])} |
| Baris masuk kelompok | {jumlah(final_metrics['n_assigned'])} |
| Noise | {jumlah(noise_rows)} ({angka(final_metrics['noise_fraction'] * 100, 2)}%) |
| Silhouette seleksi, seperti grafik bagian 8 | {angka(selected_result['silhouette'])} |
| Davies-Bouldin | {angka(selected_result['davies_bouldin'])} |

**Kesimpulan:** pada parameter tersebut terbentuk {jumlah(final_metrics['n_clusters'])} kelompok.
Pemisahannya masih lemah menurut Silhouette yang dekat nol, sehingga hasil dipakai sebagai gambaran eksploratif.
Skor ini bukan akurasi dan warna kelompok bukan tingkat bahaya.
"""))
if dbscan_metadata["diagnostic_only"]:
    display(Markdown("**Status diagnostik:** tidak ada kandidat lolos seluruh aturan. Hasil belum memenuhi kriteria penelitian."))
'''

DETAILS = '''from IPython.display import HTML

def alasan_kandidat(row):
    reasons = []
    if row["n_clusters"] < 2:
        reasons.append("Kurang dari 2 kelompok")
    if row["n_clusters"] > cfg["dbscan_max_clusters"]:
        reasons.append("Terlalu banyak kelompok")
    if row["smallest_cluster_size"] < cfg["dbscan_min_cluster_size"]:
        reasons.append("Kelompok terkecil terlalu sedikit anggota")
    if row["noise_fraction"] > cfg["dbscan_max_noise_fraction"]:
        reasons.append("Noise melewati batas")
    if row["largest_cluster_fraction_assigned"] > cfg["dbscan_max_largest_cluster_fraction"]:
        reasons.append("Satu kelompok terlalu dominan")
    if not np.isfinite(row["silhouette"]):
        reasons.append("Skor tidak terdefinisi")
    return "; ".join(reasons) if reasons else "Lolos aturan"

audit = search[["eps", "min_samples", "n_clusters", "smallest_cluster_size", "noise_fraction", "silhouette", "davies_bouldin"]].copy()
audit["noise_fraction"] *= 100
audit["Status"] = np.where(selected_mask, "TERPILIH", np.where(search["accepted"], "Lolos", "Tidak lolos"))
audit["Alasan"] = search.apply(alasan_kandidat, axis=1)
audit = audit.rename(columns={"n_clusters": "Kelompok", "smallest_cluster_size": "Anggota terkecil", "noise_fraction": "Noise (%)", "silhouette": "Silhouette seleksi", "davies_bouldin": "DBI"})
display(HTML('<details><summary>Buka tabel seluruh percobaan dan alasannya</summary>' + audit.round(4).to_html(index=False, escape=True) + '</details>'))
display(Markdown(f"""**Catatan metode:** aturan penyaringan adalah 2 sampai {cfg['dbscan_max_clusters']} kelompok,
kelompok terkecil minimal {cfg['dbscan_min_cluster_size']} baris, noise maksimal {angka(cfg['dbscan_max_noise_fraction'] * 100, 0)}%,
dominasi kelompok terbesar maksimal {angka(cfg['dbscan_max_largest_cluster_fraction'] * 100, 0)}% dari anggota yang ditugaskan.
Ini pilihan rancangan penelitian; pengaruhnya diperiksa di notebook 08.

Silhouette seleksi **{angka(selected_result['silhouette'])}** adalah rata-rata tiga sampel seed {cfg['silhouette_seeds']}.
Silhouette evaluasi dengan satu sampel seed 42 adalah **{angka(final_metrics['silhouette'])}**, sama dengan notebook 06.
Setiap sampel maksimal {jumlah(cfg['silhouette_sample_size'])} baris; DBI memakai seluruh anggota kelompok.
Noise dikeluarkan dari kedua skor. Perbedaan skor sampel tidak mengubah model.
"""))
display(FileLink(str(ROOT / "models/dbscan/parameter_search.csv"), result_html_prefix="CSV seluruh percobaan: "))
display(FileLink(str(ROOT / "reports/figures/dbscan_silhouette_samples.png"), result_html_prefix="Grafik Silhouette per anggota (detail tambahan): "))
'''

CONCLUSION = '''display(Markdown(f"""### Alur yang bisa diikuti

**Lihat jarak antarpola → bandingkan kandidat eps → periksa kelompok yang terbentuk.**

K-Means memakai grafik elbow untuk membantu memilih jumlah kelompok k.
DBSCAN memakai eps dan min_samples; jumlah kelompok muncul setelah model dijalankan.
Kurva k-distance membantu memahami kandidat eps, sedangkan pemilihan dalam penelitian ini memakai skor dan aturan penyaringan.
Kita tidak menetapkan tujuh kelompok sejak awal.

Hasil pada data ini: **{jumlah(final_metrics['n_clusters'])} kelompok dan {angka(final_metrics['noise_fraction'] * 100, 2)}% noise**.
Ciri setiap kelompok dibaca pada tabel bagian 6.
Perbandingan dengan K-Means dilanjutkan di [notebook 06](../06_model_evaluation/06_model_evaluation.ipynb).
"""))
'''
