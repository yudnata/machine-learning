from __future__ import annotations

import itertools
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kmedoids import KMedoids


def blobs(seed=0, per_cluster=25):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [6, 0], [0, 6]], dtype=float)
    return np.vstack([rng.normal(c, .5, size=(per_cluster, 2)) for c in centers])


def total_cost(X, medoids):
    return cdist(X, X[list(medoids)]).min(axis=1).sum()


def test_medoids_are_real_observations_and_labels_are_consistent():
    X = blobs()
    model = KMedoids(n_clusters=3).fit(X)
    assert len(set(model.medoid_indices_.tolist())) == 3
    np.testing.assert_allclose(model.cluster_centers_, X[model.medoid_indices_])
    np.testing.assert_array_equal(model.labels_, cdist(X, model.cluster_centers_).argmin(axis=1))
    np.testing.assert_array_equal(model.predict(X), model.labels_)
    assert np.isclose(model.inertia_, total_cost(X, model.medoid_indices_))
    assert len(np.unique(model.labels_)) == 3


def test_pam_matches_exhaustive_optimum_on_well_separated_data():
    X = blobs(seed=1, per_cluster=8)
    model = KMedoids(n_clusters=3).fit(X)
    optimum = min(total_cost(X, combo) for combo in itertools.combinations(range(len(X)), 3))
    assert np.isclose(model.inertia_, optimum)


def test_result_is_deterministic_and_survives_joblib(tmp_path):
    X = blobs(seed=2)
    first = KMedoids(n_clusters=3, random_state=1).fit(X)
    second = KMedoids(n_clusters=3, random_state=1).fit(X)
    np.testing.assert_array_equal(first.labels_, second.labels_)
    joblib.dump(first, tmp_path / "model.joblib")
    loaded = joblib.load(tmp_path / "model.joblib")
    np.testing.assert_array_equal(loaded.predict(X), first.labels_)


def test_rejects_invalid_number_of_clusters():
    X = blobs(per_cluster=2)
    for k in (1, len(X) + 1):
        with pytest.raises(ValueError, match="n_clusters"):
            KMedoids(n_clusters=k).fit(X)
