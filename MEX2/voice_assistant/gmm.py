"""From-scratch Gaussian Mixture Model (NumPy only).

Why this exists
---------------
The target runtime (and this dev box) has a broken ``sklearn`` install, and a
Pi deployment should not depend on it either. This is a small, dependency-free
GMM with **diagonal covariances** (stable for high-dimensional MFCC features,
tiny memory footprint, fast on a Pi). It exposes ``.fit()`` / ``.score()``
(mean log-likelihood) so it is a drop-in for ``sklearn.mixture.GaussianMixture``
for our use case.

Because the class lives in a real, importable module (``voice_assistant.gmm``),
models pickled with it load cleanly anywhere the package is importable --
unlike a class defined in ``__main__``.
"""
from __future__ import annotations

import numpy as np


class GMM:
    """Gaussian Mixture Model with diagonal covariances, trained by EM."""

    def __init__(self, n_components: int = 8, reg_covar: float = 1e-3,
                 n_init: int = 2, max_iter: int = 80, tol: float = 1e-3,
                 random_state: int = 0):
        self.k = int(n_components)
        self.reg = float(reg_covar)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rng = np.random.default_rng(random_state)
        self.means = None   # (k, d)
        self.vars = None    # (k, d)  diagonal variances
        self.weights = None  # (k,)

    # -- initialisation (KMeans-style) ------------------------------------ #
    def _init(self, X):
        n, d = X.shape
        k = min(self.k, n)
        idx = self.rng.choice(n, size=k, replace=False)
        M = X[idx].copy()
        for _ in range(10):
            dist = ((X[:, None, :] - M[None, :, :]) ** 2).sum(axis=2)
            assign = dist.argmin(axis=1)
            for j in range(k):
                sel = X[assign == j]
                if len(sel):
                    M[j] = sel.mean(axis=0)
        return M, k

    # -- one EM optimisation ---------------------------------------------- #
    def _em(self, X, M):
        n, d = X.shape
        k = M.shape[0]
        W = np.ones(k) / k
        V = np.maximum(X.var(axis=0), self.reg)[None, :].repeat(k, axis=0)
        prev = -np.inf
        for _ in range(self.max_iter):
            # E-step (diagonal Gaussian)
            comp = np.empty((n, k))
            for j in range(k):
                diff = X - M[j]
                logdet = np.log(V[j]).sum()
                maha = (diff ** 2 / V[j]).sum(axis=1)
                comp[:, j] = (np.log(W[j]) - 0.5 * (d * np.log(2 * np.pi)
                             + logdet + maha))
            m = comp.max(axis=1)
            resp = np.exp(comp - m[:, None])
            resp /= resp.sum(axis=1, keepdims=True)
            ll = (m + np.log(resp.sum(axis=1))).sum()
            # M-step
            nk = resp.sum(axis=0) + 1e-9
            W = nk / n
            M = (resp.T @ X) / nk[:, None]
            V = ((resp ** 2).T @ (X ** 2) / nk[:, None]
                 - 2.0 * M * (resp.T @ X) / nk[:, None]
                 + M ** 2)
            V = np.maximum(V, self.reg)
            if abs(ll - prev) < self.tol * n:
                break
            prev = ll
        return M, V, W

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        best = None
        for _ in range(self.n_init):
            M, k = self._init(X)
            M, V, W = self._em(X, M)
            ll = float(self._ll(X, M, V, W).mean())
            if best is None or ll > best[0]:
                best = (ll, M, V, W)
        _, self.means, self.vars, self.weights = best
        return self

    def _ll(self, X, M, V, W):
        """Per-sample log-likelihood via a stable log-sum-exp."""
        n, d = X.shape
        k = M.shape[0]
        comp = np.full((n, k), -np.inf)
        for j in range(k):
            diff = X - M[j]
            logdet = np.log(V[j]).sum()
            maha = (diff ** 2 / V[j]).sum(axis=1)
            comp[:, j] = (np.log(W[j]) - 0.5 * (d * np.log(2 * np.pi)
                         + logdet + maha))
        m = comp.max(axis=1)
        return m + np.log(np.exp(comp - m[:, None]).sum(axis=1))

    def score(self, X):
        """Mean per-sample log-likelihood of X (matches sklearn's .score)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return float(self._ll(X, self.means, self.vars, self.weights).mean())
