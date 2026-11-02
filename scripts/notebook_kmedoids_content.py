"""Sel notebook K-Medoids (PAM): kode algoritma, verifikasi medoid, dan grafik."""

THEORY = """K-Medoids memilih **k observasi nyata sebagai pusat kelompok (medoid)** dan meminimalkan jumlah jarak setiap observasi ke medoid kelompoknya:

$$\\min_{M \\subset X,\\ |M|=k} \\; \\sum_{i=1}^{n} \\min_{m \\in M} d(x_i, m)$$

K-Means meminimalkan jumlah **kuadrat** jarak ke rata-rata (centroid) yang berupa titik hitung, bukan observasi. Dua perbedaan ini menentukan cara membaca hasil:

| Aspek | K-Means | K-Medoids (PAM) |
| --- | --- | --- |
| Pusat kelompok | Rata-rata anggota (titik hitung) | Satu observasi nyata, yaitu satu kotak-triwulan |
| Fungsi tujuan | Jumlah jarak kuadrat | Jumlah jarak (tanpa kuadrat) |
| Pengaruh pencilan | Rata-rata ikut tertarik pencilan | Medoid tetap berupa titik data; pencilan jauh lebih kecil pengaruhnya |
| Inisialisasi | Acak (n_init=20) | BUILD deterministik, sehingga hasil tidak bergantung seed awal |
| Biaya hitung | O(n k) per iterasi | O(n^2) per sapuan SWAP |
| Interpretasi | Profil rata-rata | Ada kotak-triwulan contoh yang dapat dibuka di peta dan dicek ke katalog |

**Algoritma PAM** (Kaufman & Rousseeuw, 1990). Tahap BUILD memilih medoid pertama yang meminimalkan jumlah jarak ke seluruh titik, lalu menambah medoid satu per satu yang paling menurunkan biaya. Tahap SWAP menukar medoid dengan non-medoid selama biaya total turun. Evaluasi SWAP memakai rumus delta-biaya FastPAM (Schubert & Rousseeuw, 2021) sehingga setiap kandidat dinilai terhadap semua medoid dalam satu lintasan O(n).

**Pengaturan agar sebanding dengan K-Means dan DBSCAN:**

- Data: lima fitur standar dari training 2021-2025, sama persis dengan model lain.
- Jarak: Euclidean pada ruang fitur standar, sama dengan jarak yang dipakai Silhouette dan Davies-Bouldin.
- Kandidat k: 2 sampai 8. Aturan pemilihan identik dengan K-Means: rata-rata Silhouette tiga sampel tertinggi, lalu DBI terendah, lalu k terkecil.
- Matriks jarak penuh tidak disimpan. Baris jarak dihitung per potongan sehingga memori tetap kecil untuk 13 ribu lebih baris.
- Implementasi ditulis sendiri di `src/kmedoids.py` karena `scikit-learn-extra` tidak tersedia untuk scikit-learn versi ini. Kebenarannya diuji pada `tests/test_kmedoids.py`, termasuk pembandingan dengan solusi optimum hasil pencarian menyeluruh pada data kecil.
"""

ALGORITHM = '''import inspect

display(Markdown("```python\\n" + inspect.getsource(KMedoids) + "\\n```"))
'''

RUN = '''from scipy.spatial.distance import cdist
from src.common import load_model

search, labels = fit_kmedoids(ROOT)
kmedoids_cfg, kmedoids_panel, kmedoids_train, kmedoids_X, _, _ = load_prepared(ROOT)
kmedoids_model, kmedoids_metadata = load_model(ROOT, "kmedoids")
display(Markdown("### Seluruh kandidat k"))
display(search)
display(Markdown("### Profil rata-rata setiap kelompok (satuan asli)"))
display(pd.read_csv(ROOT / "models/kmedoids/profiles.csv"))
if kmedoids_metadata.get("matched_k_with_kmeans") is not None:
    display(Markdown(f"### Profil kontrol k={kmedoids_metadata['matched_k_with_kmeans']} (bukan k terpilih)"))
    display(pd.read_csv(ROOT / "models/kmedoids/profiles_matched_k.csv"))
    display(Markdown("### Medoid kontrol: kotak-triwulan nyata"))
    display(pd.read_csv(ROOT / "models/kmedoids/medoids_matched_k.csv"))
'''

VERIFY = '''angka = lambda value, digits=4: f"{value:.{digits}f}".replace(".", ",")
jumlah = lambda value: f"{int(value):,}".replace(",", ".")
centers = kmedoids_model.cluster_centers_
assert kmedoids_model.n_clusters == kmedoids_metadata["selected_k"]
assert len(kmedoids_model.labels_) == len(kmedoids_X) == len(kmedoids_train)
assert len(set(kmedoids_model.medoid_indices_.tolist())) == kmedoids_model.n_clusters
assert np.allclose(centers, kmedoids_X[kmedoids_model.medoid_indices_]), "Pusat kelompok harus observasi nyata."
assert (cdist(kmedoids_X, centers).argmin(axis=1) == kmedoids_model.labels_).all(), "Setiap baris harus berlabel medoid terdekat."
assert np.isclose(kmedoids_model.inertia_, cdist(kmedoids_X, centers).min(axis=1).sum())
selected_row = search.loc[search["k"].eq(kmedoids_metadata["selected_k"])].iloc[0]
display(Markdown(f"""### Pemeriksaan hasil

- Jumlah kelompok terpilih: **{kmedoids_model.n_clusters}** (aturan seleksi sama dengan K-Means).
- Biaya total (jumlah jarak ke medoid): **{angka(kmedoids_model.inertia_, 1)}**.
- Pertukaran medoid sampai konvergen: **{kmedoids_model.n_swaps_}** pada **{kmedoids_model.n_iter_}** sapuan.
- Silhouette seleksi (rata-rata 3 sampel): **{angka(selected_row['silhouette'])}**; DBI: **{angka(selected_row['davies_bouldin'])}**.
- Pemeriksaan lulus: setiap pusat kelompok adalah satu observasi nyata, dan setiap baris berlabel medoid terdekat.
"""))
medoids = pd.read_csv(ROOT / "models/kmedoids/medoids.csv")
medoids["profile"] = medoids["cluster"].map(lambda number: f"KMEDOIDS_P{number}")
medoids["frekuensi_per_bulan"] = medoids["frequency_per_month"]
display(Markdown("### Medoid: kotak-triwulan nyata yang mewakili tiap kelompok"))
display(medoids[["profile", "grid_id", "period", "earthquake_count", "avg_magnitude", "max_magnitude", "avg_depth", "shallow_fraction"]].rename(columns={
    "profile": "Kode algoritma", "grid_id": "Kotak", "period": "Triwulan", "earthquake_count": "Jumlah gempa",
    "avg_magnitude": "Rata-rata M", "max_magnitude": "M terbesar", "avg_depth": "Rata-rata kedalaman (km)", "shallow_fraction": "Proporsi dangkal"}).round(3))
display(Markdown("Berbeda dari centroid K-Means, setiap baris di atas adalah kotak dan triwulan yang benar-benar ada di katalog. Medoid dapat dicari di peta notebook 11 menurut kode kotak dan triwulannya."))
'''

FIGURES = '''from sklearn.decomposition import PCA

display(Image(filename=str(ROOT / "reports/figures/kmedoids_selection.png")))
display(Markdown("**Cara membaca:** kiri adalah jumlah jarak ke medoid (turun saat k naik, tidak ada titik siku tegas); tengah Silhouette, lebih tinggi lebih baik; kanan Davies-Bouldin, lebih rendah lebih baik. Garis putus-putus menandai k terpilih."))
palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#882255"]
pca = PCA(n_components=2, random_state=42).fit(kmedoids_X)
projection = pca.transform(kmedoids_X)
medoid_projection = pca.transform(kmedoids_model.cluster_centers_)
fig, ax = plt.subplots(figsize=(9, 6.2))
for number in range(kmedoids_model.n_clusters):
    mask = kmedoids_model.labels_ == number
    ax.scatter(projection[mask, 0], projection[mask, 1], s=7, alpha=.4, color=palette[number % len(palette)],
               label=f"Kelompok {chr(65 + number)} (n={jumlah(mask.sum())})")
ax.scatter(medoid_projection[:, 0], medoid_projection[:, 1], marker="*", s=330, color="white", edgecolor="black", linewidth=1.4, zorder=5, label="Medoid")
ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variansi)", ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variansi)",
       title="K-Medoids: proyeksi PCA lima fitur training; bintang adalah medoid")
legend = ax.legend(fontsize=9)
for handle in legend.legend_handles:
    handle.set_sizes([40])
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/kmedoids_pca_medoids.png", dpi=160)
plt.show()
plt.close(fig)
display(Markdown("**Cara membaca:** setiap titik adalah satu kotak-triwulan training pada dua sumbu PCA. PCA hanya alat visual, model dilatih pada lima fitur standar penuh. Bintang putih adalah medoid, yaitu titik data nyata, bukan titik rata-rata."))
display(labels["profile"].value_counts().rename("n_grid_quarters"))
'''
