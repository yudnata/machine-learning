"""Ringkasan evaluasi DBSCAN dari model dan data training yang tersimpan."""

METRICS = '''from src.common import load_model

evaluation_cfg, evaluation_panel, evaluation_train, evaluation_X, _, _ = load_prepared(ROOT)
dbscan_model, dbscan_metadata = load_model(ROOT, "dbscan")
candidate_results = pd.read_csv(ROOT / "models/dbscan/parameter_search.csv")
selected_mask = (np.isclose(candidate_results["eps"], dbscan_metadata["eps"], rtol=1e-10, atol=1e-12)
                 & candidate_results["min_samples"].eq(dbscan_metadata["min_samples"]))
if selected_mask.sum() != 1:
    raise ValueError("Baris parameter terpilih tidak cocok dengan metadata model DBSCAN.")
selected_result = candidate_results.loc[selected_mask].iloc[0]
if len(dbscan_model.labels_) != len(evaluation_X):
    raise ValueError("Jumlah label model berbeda dari data training. Jalankan ulang tahap model.")
final_metrics = cluster_metrics(evaluation_X, dbscan_model.labels_, evaluation_cfg["silhouette_sample_size"], seed=42)
assert final_metrics["n_clusters"] == int(selected_result["n_clusters"])
assert final_metrics["n_assigned"] == int(selected_result["n_assigned"])
assert np.isclose(final_metrics["davies_bouldin"], selected_result["davies_bouldin"])
angka = lambda value, digits=4: f"{value:.{digits}f}".replace(".", ",")
jumlah = lambda value: f"{int(value):,}".replace(",", ".")
noise_rows = final_metrics["n_rows"] - final_metrics["n_assigned"]
display(Markdown(f"""### Skor DBSCAN yang terpilih

Model dibentuk dari **2021-2025**, dengan `eps = {angka(dbscan_metadata['eps'], 6)}`
dan `min_samples = {dbscan_metadata['min_samples']}`.
Satu baris berarti **satu kotak selama satu triwulan**.

| Yang diperiksa | Hasil | Cara membaca |
| --- | ---: | --- |
| Silhouette evaluasi | **{angka(final_metrics['silhouette'])}** | Rentang -1 sampai 1; makin tinggi menunjukkan pemisahan lebih baik menurut metrik ini |
| Davies-Bouldin Index (DBI) | **{angka(final_metrics['davies_bouldin'])}** | Makin kecil menunjukkan kelompok lebih kompak dan terpisah menurut metrik ini |
| Jumlah kelompok, tanpa noise | {jumlah(final_metrics['n_clusters'])} | Banyaknya kelompok yang terbentuk |
| Baris masuk kelompok | {jumlah(final_metrics['n_assigned'])} | Baris kotak-triwulan yang berhasil dikelompokkan |
| Noise | {jumlah(noise_rows)} ({angka(final_metrics['noise_fraction'] * 100, 2)}%) | Baris yang tidak tergabung ke kelompok |
| Seluruh baris training layak | {jumlah(final_metrics['n_rows'])} | Penyebut untuk menghitung persentase noise |
| Kelompok terkecil | {jumlah(final_metrics['smallest_cluster_size'])} baris | Dukungan data untuk kelompok paling kecil |

**Artinya:** Silhouette yang dekat nol menunjukkan pemisahan antarkelompok masih lemah menurut ukuran jarak ini.
Hasil perlu dibaca sebagai pengelompokan eksploratif, bersama profil, noise, dan uji sensitivitas.
DBI tidak memiliki batas lulus universal; nilainya terutama berguna saat membandingkan hasil pada data dan ruang fitur yang sebanding.
Kedua skor ini tidak menangkap seluruh sifat kelompok berbasis kepadatan DBSCAN.

**Ini bukan akurasi.** Skor {angka(final_metrics['silhouette'])} tidak berarti akurasi {angka(final_metrics['silhouette'] * 100, 2)}%.
Noise dikeluarkan dari perhitungan kedua skor dan tetap dilaporkan terpisah.
Kotak tanpa kejadian juga tidak termasuk data yang dinilai.
"""))
display(Markdown(f"""### Mengapa ada Silhouette {angka(selected_result['silhouette'])} di tabel pencarian?

| Penggunaan | Silhouette | Cara perhitungan |
| --- | ---: | --- |
| Memilih parameter, pada tabel pencarian | {angka(selected_result['silhouette'])} | Rata-rata tiga sampel acak; seed {evaluation_cfg['silhouette_seeds']} |
| Mengevaluasi model terpilih di bagian ini | {angka(final_metrics['silhouette'])} | Satu sampel tetap dengan seed 42, sama dengan notebook 06 |

Setiap sampel memakai maksimal {jumlah(evaluation_cfg['silhouette_sample_size'])} baris yang masuk kelompok.
Perbedaan kecil terjadi karena sampelnya berbeda, bukan karena modelnya berubah.
DBI dihitung pada seluruh {jumlah(final_metrics['n_assigned'])} baris yang masuk kelompok.
Simpangan baku tiga skor seleksi adalah {angka(selected_result['silhouette_std'])}; ini variasi antarsampel, bukan interval kepercayaan.

Perbandingan DBSCAN dengan K-Means pada baris yang sama tersedia di
[notebook 06](../06_model_evaluation/06_model_evaluation.ipynb).
"""))
if dbscan_metadata["diagnostic_only"]:
    display(Markdown("**Status diagnostik:** tidak ada kandidat yang memenuhi seluruh aturan penyaringan. Hasil ini tidak boleh dipresentasikan sebagai model yang lolos aturan penelitian."))
'''

CANDIDATES = '''def rejection_reasons(row):
    reasons = []
    if row["n_clusters"] < 2:
        reasons.append("Kurang dari 2 kelompok")
    if row["n_clusters"] > evaluation_cfg["dbscan_max_clusters"]:
        reasons.append("Terlalu banyak kelompok")
    if row["smallest_cluster_size"] < evaluation_cfg["dbscan_min_cluster_size"]:
        reasons.append("Kelompok terkecil terlalu sedikit anggotanya")
    if row["noise_fraction"] > evaluation_cfg["dbscan_max_noise_fraction"]:
        reasons.append("Noise melewati batas")
    if row["largest_cluster_fraction_assigned"] > evaluation_cfg["dbscan_max_largest_cluster_fraction"]:
        reasons.append("Satu kelompok terlalu dominan")
    if not np.isfinite(row["silhouette"]):
        reasons.append("Silhouette tidak dapat dihitung")
    return "; ".join(reasons) if reasons else "Memenuhi aturan penyaringan"

candidate_view = candidate_results[["eps", "min_samples", "n_clusters", "smallest_cluster_size", "noise_fraction", "silhouette", "davies_bouldin"]].copy()
candidate_view["noise_fraction"] *= 100
candidate_view["Status"] = np.where(selected_mask, "TERPILIH", np.where(candidate_results["accepted"], "Lolos, tidak terpilih", "Tidak lolos"))
candidate_view["Alasan"] = candidate_results.apply(rejection_reasons, axis=1)
candidate_view = candidate_view.rename(columns={"eps": "eps", "min_samples": "min_samples", "n_clusters": "Jumlah kelompok", "smallest_cluster_size": "Anggota kelompok terkecil", "noise_fraction": "Noise (%)", "silhouette": "Silhouette seleksi (lebih tinggi)", "davies_bouldin": "DBI (lebih rendah)"})
with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
    display(candidate_view.round(4))
display(Markdown(f"""**Cara membaca:** lihat baris **TERPILIH** untuk hasil model utama.
Silhouette pada tabel ini adalah rata-rata tiga sampel, bukan skor satu sampel evaluasi di bagian 8.

Aturan penyaringan penelitian: jumlah kelompok 2 sampai {evaluation_cfg['dbscan_max_clusters']},
anggota kelompok terkecil minimal {evaluation_cfg['dbscan_min_cluster_size']} baris,
noise maksimal {angka(evaluation_cfg['dbscan_max_noise_fraction'] * 100, 0)}%,
dan kelompok terbesar maksimal {angka(evaluation_cfg['dbscan_max_largest_cluster_fraction'] * 100, 0)}% dari baris yang masuk kelompok.
Skor Silhouette juga harus dapat dihitung.
Di antara kandidat yang lolos, dipilih rata-rata Silhouette tertinggi; jika sama, noise lebih kecil didahulukan.

Batas tersebut adalah pilihan rancangan penelitian, bukan ketentuan universal DBSCAN.
Pengaruh perubahan aturan diperiksa pada notebook 08.
"""))
best_score = candidate_results.loc[candidate_results["silhouette"].idxmax()]
if not bool(best_score["accepted"]):
    display(Markdown(f"""**Mengapa skor tertinggi tidak langsung dipilih?**
Kandidat dengan Silhouette {angka(best_score['silhouette'])} menghasilkan {jumlah(best_score['n_clusters'])} kelompok,
dengan kelompok terkecil hanya {jumlah(best_score['smallest_cluster_size'])} baris.
Kandidat itu tidak lolos karena: {rejection_reasons(best_score).lower()}.
Karena itu, skor tertinggi saja tidak cukup untuk menentukan hasil penelitian.
"""))
display(FileLink(str(ROOT / "models/dbscan/parameter_search.csv"), result_html_prefix="Seluruh angka pencarian: "))
'''

SEARCH_PLOTS = '''from matplotlib.patches import Patch

plot_candidates = candidate_results.reset_index(drop=True)
plot_selected = (np.isclose(plot_candidates["eps"], dbscan_metadata["eps"], rtol=1e-10, atol=1e-12)
                 & plot_candidates["min_samples"].eq(dbscan_metadata["min_samples"]))
positions = np.arange(len(plot_candidates))
candidate_names = [f"eps={angka(row.eps, 3)} | min={int(row.min_samples)}" + ("  TERPILIH" if chosen else "")
                   for row, chosen in zip(plot_candidates.itertuples(), plot_selected)]
bar_colors = ["#B44A00" if chosen else ("#0072B2" if accepted else "#b3b3b3")
              for chosen, accepted in zip(plot_selected, plot_candidates["accepted"])]
fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
for ax, column, title in zip(axes, ["silhouette", "davies_bouldin", "noise_fraction"],
                            ["Silhouette seleksi: lebih tinggi lebih baik", "Davies-Bouldin: lebih rendah lebih baik", "Noise: baca bersama cakupan model"]):
    values = plot_candidates[column].to_numpy().copy()
    if column == "noise_fraction":
        values *= 100
    finite = np.isfinite(values)
    errors = plot_candidates.loc[finite, "silhouette_std"].to_numpy() if column == "silhouette" else None
    ax.barh(positions[finite], values[finite], color=np.array(bar_colors)[finite], xerr=errors, capsize=2)
    ax.set_title(title, fontsize=11, pad=12)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    if column == "silhouette":
        ax.set_xlim(-1, 1)
        ax.axvline(0, color="#333333", linewidth=.8)
        ax.set_xlabel("Skor -1 sampai 1; rata-rata tiga sampel")
    elif column == "noise_fraction":
        ax.set_xlim(0, 100)
        ax.axvline(evaluation_cfg["dbscan_max_noise_fraction"] * 100, color="#333333", linestyle="--", linewidth=1)
        ax.set_xlabel("Persentase baris training menjadi noise (%)")
    else:
        ax.set_xlabel("Davies-Bouldin Index")
        ax.set_xlim(left=0)
    for position in positions[~finite]:
        ax.text(.03, position, "Tidak terdefinisi", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
axes[0].set_yticks(positions, candidate_names, fontsize=9)
axes[0].invert_yaxis()
fig.suptitle("Perbandingan 12 kandidat DBSCAN pada training 2021-2025", fontsize=15, y=.98)
fig.legend(handles=[Patch(color="#B44A00", label="Model terpilih"), Patch(color="#0072B2", label="Lolos aturan, tidak terpilih"),
                    Patch(color="#b3b3b3", label="Tidak lolos aturan")], loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=[0, .06, 1, .94])
fig.savefig(ROOT / "reports/figures/dbscan_selection_metrics.png", dpi=160)
plt.show()
plt.close(fig)
display(Markdown(f"""**Cara membaca:** satu baris adalah satu percobaan parameter.
Baris **oranye bertuliskan TERPILIH** adalah model utama: eps {angka(dbscan_metadata['eps'], 6)}, min_samples {dbscan_metadata['min_samples']}.

- **Kiri:** bandingkan Silhouette; makin ke kanan makin tinggi. Garis kecil di ujung batang menunjukkan simpangan baku tiga sampel, bukan interval kepercayaan.
- **Tengah:** bandingkan DBI; makin pendek batangnya makin kecil nilainya.
- **Kanan:** persentase noise. Garis putus-putus menunjukkan batas noise dalam aturan penelitian. Noise rendah saja belum menjamin kelompok yang berguna.

Batang abu-abu dapat memiliki skor tinggi tetapi tidak lolos aturan penelitian.
Alasannya tersedia pada tabel bagian 9. Kandidat satu kelompok tidak memiliki skor Silhouette/DBI yang terdefinisi.
"""))
'''

MODEL_PLOTS = '''from sklearn.metrics import silhouette_samples

assigned_mask = dbscan_model.labels_ >= 0
assigned_X = evaluation_X[assigned_mask]
assigned_labels = dbscan_model.labels_[assigned_mask]
rng = np.random.default_rng(42)
sample_indices = rng.choice(len(assigned_X), min(evaluation_cfg["silhouette_sample_size"], len(assigned_X)), replace=False)
sample_X, sample_labels = assigned_X[sample_indices], assigned_labels[sample_indices]
sample_values = silhouette_samples(sample_X, sample_labels)
assert np.isclose(sample_values.mean(), final_metrics["silhouette"])
group_colors = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(11, 8))
bottom, group_ticks, group_names = 10, [], []
for cluster in sorted(np.unique(sample_labels)):
    group_values = np.sort(sample_values[sample_labels == cluster])
    top = bottom + len(group_values)
    ax.fill_betweenx(np.arange(bottom, top), 0, group_values, color=group_colors(int(cluster) % 10), alpha=.85)
    group_ticks.append((bottom + top) / 2)
    group_names.append(f"Kelompok {chr(65 + int(cluster))} / P{int(cluster)} ({len(group_values)} sampel)")
    bottom = top + 80
ax.axvline(sample_values.mean(), color="#222222", linestyle="--", label=f"Rata-rata sampel: {angka(sample_values.mean())}")
ax.axvline(0, color="#777777", linewidth=.8)
ax.set(xlim=(-1, 1), xlabel="Silhouette setiap baris sampel", ylabel="Kelompok dan jumlah baris dalam sampel",
       title="Silhouette model DBSCAN terpilih: seberapa cocok anggota dengan kelompoknya?")
ax.set_yticks(group_ticks, group_names)
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/dbscan_silhouette_samples.png", dpi=160)
plt.show()
plt.close(fig)
display(Markdown(f"""**Cara membaca:** setiap irisan tipis adalah satu baris kotak-triwulan dalam sampel.
Nilai mendekati +1 menunjukkan baris lebih cocok dengan kelompoknya menurut ukuran ini;
dekat 0 berarti sulit dibedakan dari kelompok terdekat; nilai negatif berarti rata-rata jaraknya lebih dekat ke kelompok lain.
Garis putus-putus adalah rata-rata **{angka(sample_values.mean())}**.

Grafik memakai **{jumlah(len(sample_values))} baris sampel**, seed 42, dan **tidak memasukkan noise**.
Jumlah pada label kelompok adalah jumlah sampel, bukan seluruh anggota kelompok historis.
Grafik ini menilai jarak pada lima fitur standar, bukan jarak geografis atau tingkat risiko.
"""))
unique_labels, member_counts = np.unique(dbscan_model.labels_, return_counts=True)
ordered = sorted(zip(unique_labels, member_counts), key=lambda item: (item[0] < 0, item[0]))
size_names = ["Noise" if cluster < 0 else f"Kelompok {chr(65 + int(cluster))} / P{int(cluster)}" for cluster, _ in ordered]
size_counts = [count for _, count in ordered]
size_colors = ["#777777" if cluster < 0 else group_colors(int(cluster) % 10) for cluster, _ in ordered]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(size_names, size_counts, color=size_colors)
ax.bar_label(bars, labels=[jumlah(count) for count in size_counts], padding=5)
ax.invert_yaxis()
ax.set_xlim(0, max(size_counts) * 1.15)
ax.set(title="Jumlah anggota seluruh kelompok historis dan noise", xlabel="Jumlah baris kotak-triwulan training")
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/dbscan_group_sizes.png", dpi=160)
plt.show()
plt.close(fig)
display(Markdown("**Cara membaca:** batang panjang berarti lebih banyak baris kotak-triwulan dalam kelompok tersebut. Grafik ini memakai seluruh training layak, sehingga berbeda dari jumlah sampel pada grafik Silhouette. Noise ditampilkan terpisah; warna tidak menunjukkan urutan bahaya."))
'''
