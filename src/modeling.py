from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from .common import ROOT, load_prepared, save_model_metadata


def cluster_metrics(X, labels, sample_size=2000, seed=42):
    X, labels = np.asarray(X), np.asarray(labels)
    assigned = labels >= 0
    values, counts = np.unique(labels[assigned], return_counts=True)
    result = {"n_rows": len(labels), "n_clusters": len(values), "n_assigned": int(assigned.sum()),
              "noise_fraction": float((~assigned).mean()),
              "largest_cluster_fraction_assigned": float(counts.max()/counts.sum()) if len(counts) else np.nan,
              "smallest_cluster_size": int(counts.min()) if len(counts) else 0,
              "silhouette": np.nan, "davies_bouldin": np.nan, "silhouette_n": 0}
    if len(values) < 2 or assigned.sum() <= len(values):
        return result
    xx, yy = X[assigned], labels[assigned]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(xx), min(sample_size, len(xx)), replace=False)
    observed = len(np.unique(yy[indices]))
    if 1 < observed < len(indices):
        result["silhouette"] = float(silhouette_score(xx[indices], yy[indices]))
        result["silhouette_n"] = len(indices)
    result["davies_bouldin"] = float(davies_bouldin_score(xx, yy))
    return result


def full_labels(panel, train, labels, method):
    values = train[["row_id"]].copy()
    values["cluster"] = labels
    result = panel.loc[panel["split"].eq("train")].merge(values, on="row_id", how="left", validate="one_to_one")
    result["profile"] = result["state"]
    valid = result["cluster"].notna()
    result.loc[valid, "profile"] = result.loc[valid, "cluster"].astype(int).map(
        lambda n: "NOISE" if n < 0 else f"{method.upper()}_P{n}")
    return result


def profile_table(result, features):
    eligible = result.loc[result["cluster"].notna()]
    table = eligible.groupby("profile")[features + ["earthquake_count", "frequency_per_month", "depth_coverage"]].mean()
    table["n_grid_quarters"] = eligible.groupby("profile").size()
    table["low_support_fraction"] = eligible.groupby("profile")["low_support"].mean()
    return table.reset_index()


def save_cluster_plot(root, method, X, labels):
    projection = PCA(n_components=2, random_state=42).fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 6))
    for label in sorted(np.unique(labels)):
        mask = labels == label
        ax.scatter(projection[mask, 0], projection[mask, 1], s=7, alpha=.45,
                   label=("Noise" if label < 0 else f"Profil {label}") if len(np.unique(labels)) <= 20 else None)
    ax.set(xlabel="PC1", ylabel="PC2", title=f"{method.upper()}: proyeksi PCA fitur training (bukan koordinat)")
    if len(np.unique(labels)) <= 20:
        ax.legend()
    fig.tight_layout()
    fig.savefig(Path(root) / f"reports/figures/{method}_pca.png", dpi=160)
    plt.close(fig)


def fit_kmeans(root=ROOT):
    root = Path(root)
    cfg, panel, train, X, _, meta = load_prepared(root)
    rows = []
    unique_count = len(np.unique(X, axis=0))
    with threadpool_limits(limits=2):
        for k in cfg["k_candidates"]:
            if not 2 <= k < min(len(X), unique_count + 1):
                continue
            model = KMeans(n_clusters=k, n_init=20, random_state=cfg["random_state"])
            labels = model.fit_predict(X)
            metrics = cluster_metrics(X, labels, cfg["silhouette_sample_size"])
            repeat_scores = [cluster_metrics(X, labels, cfg["silhouette_sample_size"], seed)["silhouette"] for seed in cfg["silhouette_seeds"]]
            metrics["silhouette"] = float(np.nanmean(repeat_scores))
            metrics["silhouette_std"] = float(np.nanstd(repeat_scores))
            rows.append({"k": k, "inertia": model.inertia_, **metrics})
        scores = pd.DataFrame(rows)
        valid = scores.loc[scores["silhouette"].notna()]
        if valid.empty:
            raise ValueError("K-Means tidak menghasilkan kandidat evaluasi yang valid.")
        selected = valid.sort_values(["silhouette", "davies_bouldin", "k"], ascending=[False, True, True]).iloc[0]
        model = KMeans(n_clusters=int(selected["k"]), n_init=20, random_state=cfg["random_state"]).fit(X)
    result = full_labels(panel, train, model.labels_, "kmeans")
    directory = root / "models/kmeans"
    scores.to_csv(directory / "parameter_search.csv", index=False)
    result.to_csv(directory / "training_labels.csv", index=False)
    profile_table(result, cfg["features"]).to_csv(directory / "profiles.csv", index=False)
    joblib.dump(model, directory / "model.joblib")
    save_model_metadata(root, "kmeans", meta, {"selected_k": int(selected["k"]), "selection": "rata-rata silhouette 3 sampel tertinggi; tie-break DBI terendah dan k terkecil", "fit_split": "train"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ["inertia", "silhouette", "davies_bouldin"]):
        ax.plot(scores["k"], scores[metric], marker="o")
        ax.axvline(int(selected["k"]), color="gray", ls="--")
        ax.set(xlabel="k", ylabel=metric, title=metric)
    fig.tight_layout()
    fig.savefig(root / "reports/figures/kmeans_selection.png", dpi=160)
    plt.close(fig)
    save_cluster_plot(root, "kmeans", X, model.labels_)
    return scores, result


def fit_dbscan(root=ROOT):
    root = Path(root)
    cfg, panel, train, X, _, meta = load_prepared(root)
    rows = []
    fig, ax = plt.subplots(figsize=(9, 5))
    with threadpool_limits(limits=2):
        for min_samples in cfg["dbscan_min_samples"]:
            if min_samples > len(X):
                continue
            # Query X secara eksplisit: tetangga diri berjarak nol ikut dihitung seperti min_samples DBSCAN.
            distances = NearestNeighbors(n_neighbors=min_samples, n_jobs=1).fit(X).kneighbors(X)[0][:, -1]
            ax.plot(np.sort(distances), label=f"min_samples={min_samples}")
            candidates = sorted(set(float(v) for v in np.quantile(distances, cfg["dbscan_distance_quantiles"]) if v > 0))
            if not candidates:
                candidates = [0.05, 0.1, 0.2]
            for eps in candidates:
                labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit_predict(X)
                metrics = cluster_metrics(X, labels, cfg["silhouette_sample_size"])
                repeat_scores = [cluster_metrics(X, labels, cfg["silhouette_sample_size"], seed)["silhouette"] for seed in cfg["silhouette_seeds"]]
                metrics["silhouette"] = float(np.nanmean(repeat_scores)) if np.isfinite(repeat_scores).any() else np.nan
                metrics["silhouette_std"] = float(np.nanstd(repeat_scores)) if np.isfinite(repeat_scores).any() else np.nan
                accepted = (metrics["n_clusters"] >= 2 and np.isfinite(metrics["silhouette"])
                            and metrics["n_clusters"] <= cfg["dbscan_max_clusters"]
                            and metrics["smallest_cluster_size"] >= cfg["dbscan_min_cluster_size"]
                            and metrics["noise_fraction"] <= cfg["dbscan_max_noise_fraction"]
                            and metrics["largest_cluster_fraction_assigned"] <= cfg["dbscan_max_largest_cluster_fraction"])
                rows.append({"eps": eps, "min_samples": min_samples, "accepted": accepted, **metrics})
        scores = pd.DataFrame(rows)
        if scores.empty:
            raise ValueError("Tidak ada kandidat DBSCAN.")
        pool = scores.loc[scores["accepted"]]
        diagnostic_only = pool.empty
        if diagnostic_only:
            pool = scores
        selected = pool.sort_values(["silhouette", "noise_fraction"], ascending=[False, True], na_position="last").iloc[0]
        model = DBSCAN(eps=float(selected["eps"]), min_samples=int(selected["min_samples"]), n_jobs=1).fit(X)
    result = full_labels(panel, train, model.labels_, "dbscan")
    directory = root / "models/dbscan"
    scores.to_csv(directory / "parameter_search.csv", index=False)
    result.to_csv(directory / "training_labels.csv", index=False)
    profile_table(result, cfg["features"]).to_csv(directory / "profiles.csv", index=False)
    joblib.dump(model, directory / "model.joblib")
    save_model_metadata(root, "dbscan", meta, {"eps": float(selected["eps"]), "min_samples": int(selected["min_samples"]),
                        "diagnostic_only": diagnostic_only, "fit_split": "train",
                        "selection": "rata-rata silhouette tertinggi di kandidat lolos batas noise, dominasi, fragmentasi dan dukungan; jika tidak ada, hanya diagnostik"})
    ax.set(title="k-distance training: dasar kandidat eps", xlabel="Observasi diurutkan", ylabel="Jarak ke tetangga ke-k")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "reports/figures/dbscan_k_distance.png", dpi=160)
    plt.close(fig)
    save_cluster_plot(root, "dbscan", X, model.labels_)
    if diagnostic_only:
        print("Tidak ada kandidat DBSCAN lolos kriteria. Hasil disimpan untuk diagnosis, bukan model yang telah tervalidasi.")
    return scores, result
