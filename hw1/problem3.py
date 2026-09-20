"""
EE 511 Homework 1, Problem 3
PCA / SVD on a planted-rank activation matrix.

Seed 511. NumPy only for the linear algebra (no sklearn.decomposition).
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEED = 511
M, N, R = 500, 64, 4
D_DIAG = np.array([10.0, 8.0, 6.0, 4.0])
NOISE_STD = 0.5
K_LIST = [1, 2, 4, 8, 16, 32, 64]
BYTES_PER_FP32 = 4

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)


def set_seed(seed: int = SEED) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def generate_data(rng: np.random.Generator, noise_std: float) -> tuple[np.ndarray, np.ndarray]:
    Z = rng.standard_normal((M, R))
    G = rng.standard_normal((N, R))
    W, _ = np.linalg.qr(G)
    E = rng.normal(0.0, noise_std, size=(M, N))
    offset = rng.uniform(1.5, 4.0, size=(N,))
    X = Z @ np.diag(D_DIAG) @ W.T + E + offset
    return X, Z


def pca_from_scratch(Xc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cov = (Xc.T @ Xc) / M
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def smallest_k_for_thresholds(evals: np.ndarray) -> dict[int, int]:
    ratios = np.cumsum(evals) / evals.sum()
    out = {}
    for pct in (90, 95, 99):
        out[pct] = int(np.searchsorted(ratios, pct / 100.0) + 1)
    return out


def dense_kb() -> float:
    return M * N * BYTES_PER_FP32 / 1024.0


def compressed_kb(k: int) -> float:
    return (M * k + N * k) * BYTES_PER_FP32 / 1024.0


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGDIR / f"{name}.pdf")
    plt.savefig(FIGDIR / f"{name}.png", dpi=160)
    plt.close()


def main() -> None:
    print(f"seed={SEED}")
    print(f"NumPy {np.__version__}, device=CPU")

    rng = set_seed(SEED)
    X, Z = generate_data(rng, NOISE_STD)
    mu = X.mean(axis=0)
    Xc = X - mu
    print(f"per-feature offset (first 5): {X.mean(axis=0)[:5]}")
    print(f"centered means max-abs: {np.max(np.abs(Xc.mean(axis=0))):.3e}")

    evals, evecs = pca_from_scratch(Xc)
    print("top 6 eigenvalues:")
    for i, val in enumerate(evals[:6], start=1):
        print(f"  lambda_{i} = {val:.6f}")

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    gap = np.max(np.abs(evals - (S**2) / M))
    print(f"max_i |lambda_i - sigma_i^2 / M| = {gap:.6e}")

    signs = np.sign(np.sum(evecs * V, axis=0))
    signs[signs == 0] = 1.0
    aligned = evecs * signs
    ev_gap = np.max(np.linalg.norm(aligned - V, axis=0))
    print(f"max eigenvector vs right-singular-vector gap (after sign flip) = {ev_gap:.6e}")

    k_thresh = smallest_k_for_thresholds(evals)
    print("smallest k for variance thresholds (noise std=0.5):", k_thresh)

    # Spectral gap around planted rank 4.
    print(f"lambda_4 = {evals[3]:.6f}")
    print(f"lambda_5 = {evals[4]:.6f}")
    print(f"lambda_4 / lambda_5 = {evals[3] / evals[4]:.4f}")

    # --- plots: scree + cumulative variance ---
    idx = np.arange(1, N + 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.semilogy(idx, evals, "o-", markersize=3.5, label="eigenvalue")
    ax.axvline(4.5, color="C1", linestyle="--", label="planted-rank gap")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue (variance units)")
    ax.set_title("Scree plot of the sample covariance")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    savefig("scree")

    evr = np.cumsum(evals) / evals.sum()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(idx, evr, "o-", markersize=3.5, label="cumulative variance ratio")
    for pct, color in zip((0.90, 0.95, 0.99), ("C1", "C2", "C3")):
        ax.axhline(pct, color=color, linestyle="--", label=f"{int(pct*100)}%")
    ax.set_xlabel("number of components $k$")
    ax.set_ylabel("fraction of total variance")
    ax.set_title("Cumulative explained-variance ratio")
    ax.set_ylim(0.6, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig("cumulative_variance")

    # --- (c)(iv) lower noise ---
    rng2 = set_seed(SEED)
    X_lo, _ = generate_data(rng2, 0.05)
    Xc_lo = X_lo - X_lo.mean(axis=0)
    evals_lo, _ = pca_from_scratch(Xc_lo)
    k_lo = smallest_k_for_thresholds(evals_lo)
    print("smallest k for variance thresholds (noise std=0.05):", k_lo)
    print(f"low-noise lambda_4 / lambda_5 = {evals_lo[3] / evals_lo[4]:.4f}")

    # --- (d) 2D projection ---
    scores = Xc @ evecs[:, :2]
    colors = np.where(Z[:, 0] >= 0, "C0", "C3")
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(scores[:, 0], scores[:, 1], c=colors, s=14, alpha=0.75, edgecolors="none")
    ax.scatter([], [], c="C0", s=18, label="sign of first latent coord. $+$")
    ax.scatter([], [], c="C3", s=18, label="sign of first latent coord. $-$")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Centered data on the top two principal components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    savefig("pca_scatter")

    # --- (e) reconstruction ---
    rel_emp = []
    rel_thy = []
    denom = np.linalg.norm(Xc, ord="fro")
    energy = S**2
    for k in K_LIST:
        Xhat = (U[:, :k] * S[:k]) @ Vt[:k, :]
        rel_emp.append(np.linalg.norm(Xc - Xhat, ord="fro") / denom)
        rel_thy.append(np.sqrt(energy[k:].sum() / energy.sum()) if k < N else 0.0)
    rel_emp = np.array(rel_emp)
    rel_thy = np.array(rel_thy)
    max_gap = np.max(np.abs(rel_emp - rel_thy))
    print("reconstruction relative errors:")
    for k, e, t in zip(K_LIST, rel_emp, rel_thy):
        print(f"  k={k:2d}  emp={e:.6e}  thy={t:.6e}")
    print(f"max |emp - theory| = {max_gap:.6e}")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(K_LIST, rel_emp, "o-", label=r"$\|X_c-\hat X_k\|_F/\|X_c\|_F$")
    ax.plot(K_LIST, rel_thy, "s--", label=r"$\sqrt{\sum_{i>k}\sigma_i^2/\sum_i\sigma_i^2}$")
    ax.set_xlabel("rank $k$")
    ax.set_ylabel("relative Frobenius error")
    ax.set_title("Reconstruction error vs. truncated-SVD bound")
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_LIST)
    ax.set_xticklabels([str(k) for k in K_LIST])
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig("reconstruction_error")

    # --- (f) footprints and timing ---
    print(f"dense Xc FP32 footprint = {dense_kb():.3f} KB")
    print("k  var_retained  footprint_KB  ratio  proj_ms")
    break_even_k = M * N / (M + N)
    print(f"break-even: k < MN/(M+N) = {break_even_k:.4f}")

    rng_t = np.random.default_rng(0)
    # warmup
    _ = Xc @ rng_t.standard_normal((N, 4))

    for k in K_LIST:
        var_ret = evr[k - 1]
        foot = compressed_kb(k)
        ratio = dense_kb() / foot
        Vk = evecs[:, :k]
        t0 = time.perf_counter()
        nrep = 200
        for _ in range(nrep):
            _ = Xc @ Vk
        dt = (time.perf_counter() - t0) / nrep * 1e3
        print(f"{k:2d}  {var_ret:.6f}  {foot:10.3f}  {ratio:6.3f}  {dt:.4f}")

    print("done. figures in", FIGDIR)


if __name__ == "__main__":
    main()
