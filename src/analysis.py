from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import ROOT, load_model, load_prepared, write_json
from .modeling import cluster_metrics
from .preprocessing import grid_panel


def eda(root=ROOT):
    root = Path(root)
    cfg, panel, train, _, _, _ = load_prepared(root)
    events = pd.read_csv(root / "data/processed/events_clean.csv")
    historic = events.loc[events["year"].between(2021, 2025)]
    report = root / "reports/evaluation"
    annual = historic.groupby("year").agg(n_events=("event_key", "size"), min_magnitude=("magnitude", "min"),
                 median_magnitude=("magnitude", "median"), invalid_depth_fraction=("depth_invalid", "mean"))
    annual.to_csv(report / "catalogue_by_year.csv")
    thresholds = []
    for threshold in cfg["magnitude_sensitivity"]:
        counts = historic.loc[historic["magnitude"].ge(threshold)].groupby("year").size().reindex(range(2021, 2026), fill_value=0)
        thresholds.extend({"year": int(y), "min_magnitude": threshold, "event_count": int(n)} for y,n in counts.items())
    pd.DataFrame(thresholds).to_csv(report / "magnitude_threshold_audit.csv", index=False)
    historic.groupby(["year", "magnitude_type"], dropna=False).size().rename("n").to_csv(report / "magnitude_types.csv")
    historic.groupby(["year", "author"], dropna=False).size().rename("n").to_csv(report / "catalogue_authors.csv")
    mc_rows = []
    for year, group in historic.groupby("year"):
        magnitude = group["magnitude"].dropna()
        rounded = (magnitude / .1).round().astype(int)
        mode = float(rounded.value_counts().idxmax() * .1) if len(rounded) else np.nan
        mc_rows.append({"year": year, "max_curvature_histogram_mode": mode,
                        "analysis_threshold": cfg["min_magnitude"],
                        "threshold_below_histogram_mode": cfg["min_magnitude"] < mode,
                        "note": "diagnostik histogram 0.1, bukan estimasi Mc tervalidasi atau bukti katalog lengkap"})
    pd.DataFrame(mc_rows).to_csv(report / "completeness_diagnostic.csv", index=False)
    train[cfg["features"]].corr().to_csv(report / "feature_correlation.csv")
    panel.groupby(["split", "state"], observed=True).size().rename("n").to_csv(report / "panel_states.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    historic["magnitude"].hist(ax=axes[0,0], bins=40)
    axes[0,0].axvline(cfg["min_magnitude"], color="red", ls="--")
    axes[0,0].set(title="Magnitudo katalog 2021-2025", xlabel="Magnitudo", ylabel="Event")
    historic["depth"].hist(ax=axes[0,1], bins=40)
    axes[0,1].set(title="Kedalaman properties.depth", xlabel="km", ylabel="Event")
    historic.groupby("period").size().plot(ax=axes[1,0], marker="o", rot=65)
    axes[1,0].set(title="Jumlah tercatat per triwulan (sebelum ambang M)", ylabel="Event")
    axes[1,1].scatter(historic["magnitude"], historic["depth"], s=3, alpha=.15)
    axes[1,1].set(xlabel="Magnitudo", ylabel="Kedalaman (km)", title="Magnitudo dan kedalaman")
    fig.tight_layout()
    fig.savefig(root / "reports/figures/eda_training.png", dpi=160)
    plt.close(fig)
    return annual, pd.DataFrame(thresholds)


def evaluate(root=ROOT):
    root = Path(root)
    cfg, _, train, X, _, _ = load_prepared(root)
    km, _ = load_model(root, "kmeans")
    db, dbmeta = load_model(root, "dbscan")
    rows = []
    for method, labels in [("kmeans", km.labels_), ("dbscan", db.labels_)]:
        rows.append({"model": method, "evaluation_scope": "seluruh baris training layak; noise dikeluarkan dari metrik jarak",
                     **cluster_metrics(X, labels, cfg["silhouette_sample_size"])})
    common = db.labels_ >= 0
    for method, labels in [("kmeans", km.labels_), ("dbscan", db.labels_)]:
        if common.any():
            rows.append({"model": method, "evaluation_scope": "subset identik: baris training yang ditugaskan DBSCAN",
                         **cluster_metrics(X[common], labels[common], cfg["silhouette_sample_size"])})
    scores = pd.DataFrame(rows)
    scores["dbscan_diagnostic_only"] = dbmeta["diagnostic_only"]
    scores.to_csv(root / "reports/evaluation/model_comparison.csv", index=False)
    ari = adjusted_rand_score(km.labels_[common], db.labels_[common]) if common.sum() > 1 else None
    write_json(root / "reports/evaluation/agreement.json", {"ari_common_assigned": ari,
               "n_common_assigned": int(common.sum()), "fraction_common_assigned": float(common.mean()),
               "meaning": "kesepakatan partisi, bukan akurasi terhadap ground truth"})
    return scores


def temporal_tables(labels):
    data = labels.sort_values(["grid_id", "period"]).copy()
    data["previous_profile"] = data.groupby("grid_id")["profile"].shift()
    data["previous_period"] = data.groupby("grid_id")["period"].shift()
    data["is_profile"] = data["profile"].str.contains("_P", regex=False)
    previous_is_profile = data["previous_profile"].fillna("").str.contains("_P", regex=False)
    ordinal = pd.PeriodIndex(data["period"], freq="Q").asi8
    previous_ordinal = pd.Series(ordinal, index=data.index).groupby(data["grid_id"]).shift()
    adjacent = ordinal - previous_ordinal == 1
    data["comparable"] = data["is_profile"] & previous_is_profile & adjacent
    data["changed"] = data["comparable"] & data["profile"].ne(data["previous_profile"])
    transitions = pd.crosstab(data.loc[data["comparable"], "previous_profile"], data.loc[data["comparable"], "profile"])
    state_transitions = pd.crosstab(data.loc[adjacent, "previous_profile"], data.loc[adjacent, "profile"])
    summaries, spells = [], []
    for grid_id, group in data.groupby("grid_id", sort=True):
        group = group.copy()
        group["run"] = (group["profile"].ne(group["profile"].shift()) | ~adjacent.loc[group.index]).cumsum()
        valid_spells = []
        for _, run in group.groupby("run"):
            if run["is_profile"].all():
                duration = len(run)
                valid_spells.append(duration)
                spells.append({"grid_id": grid_id, "profile": run.iloc[0]["profile"], "start": run.iloc[0]["period"],
                               "end": run.iloc[-1]["period"], "duration_quarters": duration})
        n = int(group["comparable"].sum())
        changes = int(group["changed"].sum())
        summaries.append({"grid_id": grid_id, "n_profile_quarters": int(group["is_profile"].sum()),
                          "n_comparable_pairs": n, "n_profile_changes": changes,
                          "stability_rate": 1-changes/n if n else np.nan,
                          "max_profile_run_quarters": max(valid_spells, default=0),
                          "n_no_event_quarters": int(group["profile"].eq("NO_EVENTS").sum()),
                          "n_noise_quarters": int(group["profile"].eq("NOISE").sum())})
    return data, pd.DataFrame(summaries), transitions, state_transitions, pd.DataFrame(spells)


def temporal(root=ROOT):
    root = Path(root)
    load_prepared(root)
    outputs = {}
    for method in ["kmeans", "dbscan"]:
        load_model(root, method)
        labels = pd.read_csv(root / f"models/{method}/training_labels.csv")
        tables = temporal_tables(labels)
        output = root / f"reports/temporal/{method}"
        output.mkdir(exist_ok=True)
        for name, table in zip(["history", "grid_stability", "profile_transitions", "state_transitions", "profile_spells"], tables):
            table.to_csv(output / f"{name}.csv", index=name in ["profile_transitions", "state_transitions"])
        matrix = tables[2]
        matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).to_csv(output / "transition_probabilities.csv")
        if not matrix.empty:
            fig, ax = plt.subplots(figsize=(7,6))
            im = ax.imshow(matrix.to_numpy(), cmap="Blues")
            ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=60, ha="right")
            ax.set_yticks(range(len(matrix.index)), matrix.index)
            ax.set(xlabel="Profil berikutnya", ylabel="Profil sebelumnya", title=f"{method.upper()}: transisi berurutan 2021-2025")
            fig.colorbar(im, ax=ax, label="Jumlah pasangan grid-triwulan")
            fig.tight_layout()
            fig.savefig(root / f"reports/figures/{method}_transitions.png", dpi=160)
            plt.close(fig)
        outputs[method] = tables[1]
    return outputs


def extend_dbscan(model, X):
    labels = np.full(len(X), -1, dtype=int)
    matching_clusters = np.zeros(len(X), dtype=int)
    distances = np.full(len(X), np.nan)
    if len(X) == 0 or len(model.components_) == 0:
        return labels, matching_clusters, distances
    core_labels = model.labels_[model.core_sample_indices_]
    nn = NearestNeighbors(radius=model.eps, n_jobs=1).fit(model.components_)
    nearest_distances, _ = nn.kneighbors(X, n_neighbors=1)
    distances = nearest_distances[:,0]
    # Per batch agar radius-neighbor tidak menampung seluruh daftar sekaligus.
    for start in range(0, len(X), 500):
        neighbors = nn.radius_neighbors(X[start:start+500], return_distance=False)
        for offset, indices in enumerate(neighbors):
            clusters = np.unique(core_labels[indices])
            row = start + offset
            matching_clusters[row] = len(clusters)
            if len(clusters) == 1:
                labels[row] = clusters[0]
            elif len(clusters) > 1:
                labels[row] = -2
    return labels, matching_clusters, distances


def testing_2026(root=ROOT):
    root = Path(root)
    cfg, panel, _, Xtrain, bundle, _ = load_prepared(root)
    km, _ = load_model(root, "kmeans")
    db, dbmeta = load_model(root, "dbscan")
    test = panel.loc[panel["split"].ne("train")].copy()
    eligible = test.loc[test["state"].eq("ELIGIBLE")].copy()
    if eligible.empty:
        raise ValueError("Tidak ada observasi 2026 yang layak.")
    X = bundle["scaler"].transform(eligible[cfg["features"]])
    with threadpool_limits(limits=2):
        km_labels = km.predict(X)
        distances = km.transform(X).min(axis=1)
        training_distances = km.transform(Xtrain).min(axis=1)
        threshold = float(np.quantile(training_distances, .99))
        db_labels, matches, core_distances = extend_dbscan(db, X)
    predicted = eligible[["row_id"]].copy()
    predicted["kmeans_cluster"] = km_labels
    predicted["kmeans_profile"] = [f"KMEANS_P{label}" for label in km_labels]
    predicted["centroid_distance"] = distances
    predicted["outside_training_distance_q99"] = distances > threshold
    predicted["dbscan_cluster"] = db_labels
    predicted["dbscan_profile"] = ["AMBIGUOUS_CORE_MATCH" if x == -2 else "NO_CORE_MATCH" if x == -1 else f"DBSCAN_P{x}" for x in db_labels]
    predicted["matching_core_clusters"] = matches
    predicted["nearest_core_distance"] = core_distances
    test = test.merge(predicted, on="row_id", how="left", validate="one_to_one")
    for method in ["kmeans", "dbscan"]:
        test[f"{method}_profile"] = test[f"{method}_profile"].fillna(test["state"])
    out = root / "reports/testing_2026"
    test.to_csv(out / "labels_all_2026.csv", index=False)
    test.loc[test["split"].eq("test")].to_csv(out / "labels_complete_quarters.csv", index=False)
    test.loc[test["split"].eq("test_partial")].to_csv(out / "labels_partial_quarter.csv", index=False)
    rows = []
    for split in ["test", "test_partial"]:
        mask = eligible["split"].eq(split).to_numpy()
        if not mask.any():
            continue
        for method, labels in [("kmeans", km_labels), ("dbscan_core_extension", db_labels)]:
            rows.append({"split": split, "method": method, "ambiguous_fraction": float((labels[mask] == -2).mean()),
                         **cluster_metrics(X[mask], labels[mask], cfg["silhouette_sample_size"])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "holdout_metrics.csv", index=False)
    drift = []
    train = panel.loc[panel["split"].eq("train") & panel["state"].eq("ELIGIBLE")]
    complete = eligible.loc[eligible["split"].eq("test")]
    for feature in cfg["features"]:
        sd = train[feature].std(ddof=0)
        drift.append({"feature": feature, "train_mean": train[feature].mean(), "test_complete_mean": complete[feature].mean(),
                      "standardized_mean_shift": (complete[feature].mean()-train[feature].mean())/sd if sd else np.nan})
    pd.DataFrame(drift).to_csv(out / "feature_drift.csv", index=False)
    changes = []
    for method in ["kmeans", "dbscan"]:
        last = pd.read_csv(root / f"models/{method}/training_labels.csv")
        last = last.loc[last["period"].eq("2025Q4"), ["grid_id", "profile"]].rename(columns={"profile": "previous_profile"})
        combined = test.merge(last, on="grid_id", how="left")
        combined["model"] = method
        combined["profile"] = combined[f"{method}_profile"]
        combined["baseline_period"] = "2025Q4"
        combined["comparable_to_baseline"] = combined["profile"].str.contains("_P", regex=False) & combined["previous_profile"].fillna("").str.contains("_P", regex=False)
        combined["changed_from_baseline"] = combined["comparable_to_baseline"] & combined["profile"].ne(combined["previous_profile"])
        changes.append(combined[["grid_id", "period", "split", "model", "baseline_period", "previous_profile", "profile", "comparable_to_baseline", "changed_from_baseline"]])
    pd.concat(changes).to_csv(out / "changes_from_2025Q4.csv", index=False)
    write_json(out / "methodology.json", {"train_period": "2021Q1-2025Q4", "test_end_exclusive_utc": cfg["test_end_exclusive"],
               "complete_quarters": sorted(test.loc[test["split"].eq("test"), "period"].unique().tolist()),
               "partial_quarters": sorted(test.loc[test["split"].eq("test_partial"), "period"].unique().tolist()),
               "kmeans_distance_q99_training": threshold, "dbscan_method": "core-radius extension, bukan predict bawaan; beberapa cluster core dalam eps = ambigu",
               "dbscan_diagnostic_only": dbmeta["diagnostic_only"], "accuracy": "tidak dihitung karena tidak ada label ground truth"})
    return metrics, test


def stability(root=ROOT):
    root = Path(root)
    cfg, _, train, X, _, _ = load_prepared(root)
    km, _ = load_model(root, "kmeans")
    db, _ = load_model(root, "dbscan")
    rows = []
    with threadpool_limits(limits=2):
        for k in sorted(set([max(2, km.n_clusters-1), km.n_clusters, km.n_clusters+1])):
            labels = KMeans(n_clusters=k, n_init=20, random_state=cfg["random_state"]).fit_predict(X)
            rows.append({"method": "kmeans", "perturbation": "n_clusters", "value": k, "n_same_rows":len(X),
                         "ari_all":adjusted_rand_score(km.labels_, labels), "ari_common_assigned":adjusted_rand_score(km.labels_, labels), "common_assigned_fraction":1.0})
        for seed in cfg["stability_seeds"]:
            labels = KMeans(n_clusters=km.n_clusters, n_init=20, random_state=seed).fit_predict(X)
            rows.append({"method": "kmeans", "perturbation": "random_seed", "value": seed, "n_same_rows": len(X),
                         "ari_all": adjusted_rand_score(km.labels_, labels), "ari_common_assigned": adjusted_rand_score(km.labels_, labels), "common_assigned_fraction": 1.0})
        for factor in [.8, 1.0, 1.2]:
            for min_samples in sorted(set([max(2, db.min_samples//2), db.min_samples, db.min_samples*2])):
                labels = DBSCAN(eps=db.eps*factor, min_samples=min_samples, n_jobs=1).fit_predict(X)
                common = (db.labels_ >= 0) & (labels >= 0)
                rows.append({"method": "dbscan", "perturbation": "eps_and_min_samples", "value": f"eps_x{factor};min_samples={min_samples}",
                             "n_same_rows": len(X), "ari_all": adjusted_rand_score(db.labels_, labels),
                             "ari_common_assigned": adjusted_rand_score(db.labels_[common], labels[common]) if common.sum()>1 else np.nan,
                             "common_assigned_fraction": common.mean()})
    params = pd.DataFrame(rows)
    params.to_csv(root / "reports/stability/parameter_ari.csv", index=False)
    support_rows=[]
    with threadpool_limits(limits=2):
        for minimum_count in [1,3,5,10]:
            mask=train["earthquake_count"].ge(minimum_count).to_numpy()
            subset=train.loc[mask]
            if len(subset)<=km.n_clusters:
                continue
            xx=StandardScaler().fit_transform(subset[cfg["features"]])
            for method,reference in [("kmeans",km.labels_),("dbscan",db.labels_)]:
                model=KMeans(n_clusters=km.n_clusters,n_init=20,random_state=cfg["random_state"]) if method=="kmeans" else DBSCAN(eps=db.eps,min_samples=db.min_samples,n_jobs=1)
                alternate=model.fit_predict(xx)
                common=(reference[mask]>=0)&(alternate>=0)
                support_rows.append({"method":method,"min_events_per_quarter":minimum_count,"n_same_grid_quarters":int(mask.sum()),
                                     "fraction_reference_rows":mask.mean(),"n_clusters":len(set(alternate)-{-1}),"noise_fraction":float((alternate<0).mean()),
                                     "ari_same_grid_quarters":adjusted_rand_score(reference[mask],alternate),
                                     "ari_common_assigned":adjusted_rand_score(reference[mask][common],alternate[common]) if common.sum()>1 else np.nan,
                                     "common_assigned_fraction":common.mean()})
    pd.DataFrame(support_rows).to_csv(root / "reports/stability/minimum_event_support.csv",index=False)
    search = pd.read_csv(root / "models/dbscan/parameter_search.csv")
    policies = []
    for max_clusters in [10, 20, 50, 300]:
        for min_cluster_size in [5, 20, 50]:
            valid = search.loc[search["n_clusters"].between(2,max_clusters)
                               & search["smallest_cluster_size"].ge(min_cluster_size)
                               & search["noise_fraction"].le(cfg["dbscan_max_noise_fraction"])
                               & search["largest_cluster_fraction_assigned"].le(cfg["dbscan_max_largest_cluster_fraction"])
                               & search["silhouette"].notna()]
            if valid.empty:
                policies.append({"max_clusters":max_clusters,"min_cluster_size":min_cluster_size,"status":"no_candidate"})
            else:
                selected=valid.sort_values(["silhouette","noise_fraction"],ascending=[False,True]).iloc[0]
                policies.append({"max_clusters":max_clusters,"min_cluster_size":min_cluster_size,"status":"candidate", **selected[["eps","min_samples","n_clusters","noise_fraction","silhouette"]].to_dict()})
    pd.DataFrame(policies).to_csv(root / "reports/stability/dbscan_selection_policy.csv", index=False)
    events = pd.read_csv(root / "data/processed/events_clean.csv")
    events["time"] = pd.to_datetime(events["time"], utc=True, format="mixed")
    reference_events = pd.read_csv(root / "data/processed/events_with_grid.csv")
    reference_events = reference_events.loc[reference_events["year"].between(2021,2025)]
    reference_events["row_id"] = reference_events["grid_id"] + "|" + reference_events["period"]
    base = reference_events[["event_key", "row_id"]].merge(train[["row_id"]].assign(kmeans=km.labels_, dbscan=db.labels_), on="row_id", how="inner")
    tests = [("grid_km", size) for size in cfg["grid_sensitivity_km"]]
    tests += [("min_magnitude", value) for value in cfg["magnitude_sensitivity"]]
    resolution_rows = []
    with threadpool_limits(limits=2):
        for kind, value in tests:
            alternate_cfg = {**cfg, kind: value}
            alt_panel, alt_events = grid_panel(events, alternate_cfg)
            alt_train = alt_panel.loc[alt_panel["split"].eq("train") & alt_panel["state"].eq("ELIGIBLE")].sort_values("row_id").reset_index(drop=True)
            if len(alt_train) <= km.n_clusters:
                continue
            xx = StandardScaler().fit_transform(alt_train[cfg["features"]])
            alt_events = alt_events.loc[alt_events["year"].between(2021,2025)].copy()
            alt_events["row_id"] = alt_events["grid_id"] + "|" + alt_events["period"]
            for method, ref_model in [("kmeans",km),("dbscan",db)]:
                model = KMeans(n_clusters=km.n_clusters, n_init=20, random_state=cfg["random_state"]) if method=="kmeans" else DBSCAN(eps=db.eps, min_samples=db.min_samples, n_jobs=1)
                labels = model.fit_predict(xx)
                membership = alt_events[["event_key", "row_id"]].merge(alt_train[["row_id"]].assign(alternate_label=labels), on="row_id", how="inner")
                common = base[["event_key",method]].merge(membership[["event_key","alternate_label"]], on="event_key", validate="one_to_one")
                assigned = common[method].ge(0) & common["alternate_label"].ge(0)
                resolution_rows.append({"method":method,"perturbation":kind,"value":value,"n_common_events":len(common),
                    "fraction_reference_events":len(common)/len(base) if len(base) else np.nan,
                    "ari_same_events_including_noise":adjusted_rand_score(common[method],common["alternate_label"]) if len(common)>1 else np.nan,
                    "ari_common_assigned":adjusted_rand_score(common.loc[assigned,method],common.loc[assigned,"alternate_label"]) if assigned.sum()>1 else np.nan,
                    "common_assigned_fraction":assigned.mean(), "n_grid_quarters":len(alt_train)})
    resolution = pd.DataFrame(resolution_rows)
    resolution.to_csv(root / "reports/stability/grid_and_magnitude_ari.csv", index=False)
    return params, resolution
