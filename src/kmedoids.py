"""K-Medoids (Partitioning Around Medoids, PAM) tanpa dependensi tambahan.

Implementasi mengikuti Kaufman & Rousseeuw (1990): inisialisasi BUILD yang
deterministik, lalu SWAP. Evaluasi SWAP memakai rumus delta-TD dari FastPAM
(Schubert & Rousseeuw, 2021) dengan penerapan swap segera (eager) sehingga
setiap kandidat dinilai terhadap seluruh medoid dalam O(n).

Matriks jarak penuh tidak disimpan. Baris jarak dihitung per potongan, jadi
memori tetap kecil walaupun n belasan ribu. Jarak yang dipakai adalah Euclidean
tanpa dikuadratkan, sama dengan ruang fitur K-Means dan Silhouette proyek ini.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin


class KMedoids(ClusterMixin, BaseEstimator):
    """Pengelompokan K-Medoids PAM; pusat kelompok selalu berupa observasi nyata."""

    def __init__(self, n_clusters=3, max_iter=100, random_state=42, chunk_size=256, tol=1e-10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.chunk_size = chunk_size
        self.tol = tol

    def _build(self, X):
        n, k = len(X), self.n_clusters
        totals = np.concatenate([cdist(X[a:a + self.chunk_size], X).sum(axis=1) for a in range(0, n, self.chunk_size)])
        medoids = [int(np.argmin(totals))]
        nearest = cdist(X, X[medoids]).ravel()
        while len(medoids) < k:
            gains = np.concatenate([np.maximum(nearest[None, :] - cdist(X[a:a + self.chunk_size], X), 0).sum(axis=1)
                                    for a in range(0, n, self.chunk_size)])
            gains[medoids] = -np.inf
            chosen = int(np.argmax(gains))
            medoids.append(chosen)
            nearest = np.minimum(nearest, cdist(X, X[[chosen]]).ravel())
        return np.array(medoids, dtype=int)

    @staticmethod
    def _assign(X, medoids):
        distances = cdist(X, X[medoids])
        order = np.argsort(distances, axis=1)[:, :2]
        rows = np.arange(len(X))
        return order[:, 0], distances[rows, order[:, 0]], distances[rows, order[:, 1]]

    def fit(self, X, y=None):
        X = np.ascontiguousarray(X, dtype=float)
        n, k = X.shape[0], self.n_clusters
        if not 2 <= k <= n:
            raise ValueError("n_clusters harus berada pada rentang 2..jumlah observasi.")
        medoids = self._build(X)
        near, nearest, second = self._assign(X, medoids)
        rng = np.random.default_rng(self.random_state)
        swaps, passes = 0, 0
        for passes in range(1, self.max_iter + 1):
            improved = False
            order = rng.permutation(n)
            for start in range(0, n, self.chunk_size):
                candidates = order[start:start + self.chunk_size]
                block = cdist(X[candidates], X)
                for row, candidate in enumerate(candidates):
                    if candidate in medoids:
                        continue
                    d = block[row]
                    gain_others = np.minimum(d - nearest, 0.0)
                    extra = np.minimum(d, second) - nearest - gain_others
                    delta = gain_others.sum() + np.bincount(near, weights=extra, minlength=k)
                    best = int(np.argmin(delta))
                    if delta[best] < -self.tol:
                        medoids[best] = candidate
                        near, nearest, second = self._assign(X, medoids)
                        swaps += 1
                        improved = True
            if not improved:
                break
        self.medoid_indices_ = medoids
        self.cluster_centers_ = X[medoids].copy()
        self.labels_ = near.astype(int)
        self.inertia_ = float(nearest.sum())
        self.n_iter_ = passes
        self.n_swaps_ = swaps
        return self

    def transform(self, X):
        return cdist(np.asarray(X, dtype=float), self.cluster_centers_)

    def predict(self, X):
        return self.transform(X).argmin(axis=1)
