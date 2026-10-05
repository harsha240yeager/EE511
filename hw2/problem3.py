"""
EE 511 Homework 2, Problem 3
Linear SVM and softmax classifiers on CIFAR-10, from scratch in NumPy.

Seed 511. Autograd is not used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

SEED = 511
N_VAL = 5000
BATCH = 256
EPOCHS = 20
INIT_STD = 1e-3
DELTA = 1.0
H_FD = 1e-5
CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
RESDIR = HERE / "results"
DATADIR = HERE / "data"
for d in (FIGDIR, RESDIR, DATADIR):
    d.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 140, "font.size": 10})


def set_seed(seed: int = SEED) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def dump(name: str, obj) -> None:
    path = RESDIR / name
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path.name}", flush=True)


def load_cifar():
    train = datasets.CIFAR10(root=str(DATADIR), train=True, download=True)
    test = datasets.CIFAR10(root=str(DATADIR), train=False, download=True)
    x_tr = train.data.astype(np.float64)  # (50000, 32, 32, 3)
    y_tr = np.array(train.targets, dtype=np.int64)
    x_te = test.data.astype(np.float64)
    y_te = np.array(test.targets, dtype=np.int64)
    return x_tr, y_tr, x_te, y_te


def split_and_standardize(x_tr, y_tr, x_te, y_te, rng):
    n = x_tr.shape[0]
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:N_VAL], perm[N_VAL:]
    x_val, y_val = x_tr[val_idx], y_tr[val_idx]
    x_tr, y_tr = x_tr[tr_idx], y_tr[tr_idx]

    x_tr = x_tr.reshape(len(x_tr), -1)
    x_val = x_val.reshape(len(x_val), -1)
    x_te = x_te.reshape(len(x_te), -1)

    mean = x_tr.mean(axis=0)
    std = x_tr.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)

    def pack(x):
        z = (x - mean) / std
        ones = np.ones((z.shape[0], 1), dtype=z.dtype)
        return np.hstack([z, ones])

    stats = {
        "mean": mean,
        "std": std,
        "n_train": int(len(x_tr)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_te)),
    }
    return pack(x_tr), y_tr, pack(x_val), y_val, pack(x_te), y_te, stats


def init_W(rng, dtype=np.float64):
    return (rng.normal(0.0, INIT_STD, size=(10, 3073))).astype(dtype)


def svm_loss_grad(W, X, y, lam, delta=DELTA):
    B = X.shape[0]
    scores = X @ W.T
    correct = scores[np.arange(B), y]
    margins = scores - correct[:, None] + delta
    margins[np.arange(B), y] = 0.0
    viol = margins > 0.0
    data = np.maximum(margins, 0.0).sum(axis=1).mean()
    reg = 0.5 * lam * np.sum(W[:, :-1] ** 2)
    dS = viol.astype(W.dtype)
    dS[np.arange(B), y] -= viol.sum(axis=1)
    dS /= B
    dW = dS.T @ X
    dW[:, :-1] += lam * W[:, :-1]
    return float(data + reg), dW, dS


def softmax_loss_grad(W, X, y, lam):
    B = X.shape[0]
    scores = X @ W.T
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    P = exp / exp.sum(axis=1, keepdims=True)
    data = -np.log(np.clip(P[np.arange(B), y], 1e-12, 1.0)).mean()
    reg = 0.5 * lam * np.sum(W[:, :-1] ** 2)
    dS = P.copy()
    dS[np.arange(B), y] -= 1.0
    dS /= B
    dW = dS.T @ X
    dW[:, :-1] += lam * W[:, :-1]
    return float(data + reg), dW, dS


def accuracy(W, X, y, batch=2048):
    correct = 0
    for i in range(0, len(X), batch):
        s = X[i : i + batch] @ W.T
        correct += int((s.argmax(axis=1) == y[i : i + batch]).sum())
    return correct / len(X)


def grad_check(loss_grad, W, X, y, lam, rng, n_coords=10):
    _, analytic, _ = loss_grad(W, X, y, lam)
    rels = []
    coords = []
    for _ in range(n_coords):
        i = int(rng.integers(0, W.shape[0]))
        j = int(rng.integers(0, W.shape[1]))
        E = np.zeros_like(W)
        E[i, j] = 1.0
        lp, _, _ = loss_grad(W + H_FD * E, X, y, lam)
        lm, _, _ = loss_grad(W - H_FD * E, X, y, lam)
        numeric = (lp - lm) / (2.0 * H_FD)
        ga = float(analytic[i, j])
        rel = abs(ga - numeric) / max(abs(ga) + abs(numeric), 1e-12)
        rels.append(rel)
        coords.append({"i": i, "j": j, "analytic": ga, "numeric": numeric, "rel": rel})
    return float(max(rels)), coords


def iterate_minibatches(X, y, batch, rng):
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch):
        sl = idx[start : start + batch]
        if len(sl) == 0:
            continue
        yield X[sl], y[sl]


def train(
    loss_grad,
    Xtr,
    ytr,
    Xval,
    yval,
    eta,
    lam,
    rng,
    epochs=EPOCHS,
    record_sparsity=False,
    l1=0.0,
):
    W = init_W(rng, dtype=np.float64)
    hist = {"train_loss": [], "train_acc": [], "val_acc": [], "grad_zero_frac": []}
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        losses = []
        zero_fracs = []
        for xb, yb in iterate_minibatches(Xtr, ytr, BATCH, rng):
            loss, dW, dS = loss_grad(W, xb, yb, lam)
            W -= eta * dW
            if l1 > 0.0:
                thr = eta * l1
                w = W[:, :-1]
                W[:, :-1] = np.sign(w) * np.maximum(np.abs(w) - thr, 0.0)
            losses.append(loss)
            if record_sparsity:
                zero_fracs.append(float((np.abs(dS) == 0).mean()))
        hist["train_loss"].append(float(np.mean(losses)))
        hist["train_acc"].append(accuracy(W, Xtr, ytr))
        hist["val_acc"].append(accuracy(W, Xval, yval))
        if record_sparsity:
            hist["grad_zero_frac"].append(float(np.mean(zero_fracs)))
        print(
            f"  ep {ep:02d} loss={hist['train_loss'][-1]:.4f} "
            f"tr={hist['train_acc'][-1]:.4f} va={hist['val_acc'][-1]:.4f}",
            flush=True,
        )
    hist["seconds"] = time.perf_counter() - t0
    hist["best_epoch"] = int(np.argmax(hist["val_acc"]) + 1)
    hist["best_val_acc"] = float(max(hist["val_acc"]))
    return W, hist


def templates_image(W, stats):
    """Undo flatten of torchvision 32x32x3 row-major layout; min-max to [0,255]."""
    weights = W[:, :-1].reshape(10, 32, 32, 3)
    imgs = []
    for k in range(10):
        t = weights[k]
        lo, hi = t.min(), t.max()
        scaled = np.zeros_like(t) if hi <= lo else (t - lo) / (hi - lo)
        imgs.append((scaled * 255.0).astype(np.uint8))
    return np.stack(imgs, axis=0)


def save_template_grid(imgs, title, path):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(imgs[k])
        ax.set_title(CIFAR_CLASSES[k], fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def plot_loss(hist, title, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    xs = np.arange(1, len(hist["train_loss"]) + 1)
    ax.plot(xs, hist["train_loss"], label="train loss (data + L2)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (nats / sample)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def plot_sparsity(svm_frac, sm_frac, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    xs = np.arange(1, len(svm_frac) + 1)
    ax.plot(xs, svm_frac, label="SVM  (fraction of zeros in dL/dS)")
    ax.plot(xs, sm_frac, label="softmax (fraction of zeros in dL/dS)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("fraction of exactly-zero gradient entries")
    ax.set_title("Score-gradient sparsity vs. epoch")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def plot_l1(rows, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    spars = [r["sparsity_pct"] for r in rows]
    accs = [r["test_acc"] for r in rows]
    ax.plot(spars, accs, "o-", label="softmax + L1 (proximal)")
    ax.set_xlabel("weight sparsity (exactly-zero %, weights only)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs. L1 sparsity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def float16_softmax_study():
    def naive(z):
        e = np.exp(z)
        return e / e.sum()

    def shifted(z):
        m = z.max()
        e = np.exp(z - m)
        return e / e.sum()

    out = {}
    for vec in ([8, 9, 10], [10, 11, 12]):
        z16 = np.array(vec, dtype=np.float16)
        z64 = np.array(vec, dtype=np.float64)
        out[str(vec)] = {
            "naive_f16": [None if not np.isfinite(v) else float(v) for v in naive(z16)],
            "shift_f16": [None if not np.isfinite(v) else float(v) for v in shifted(z16)],
            "naive_f64": [float(v) for v in naive(z64)],
            "shift_f64": [float(v) for v in shifted(z64)],
        }

    # smallest input where np.exp overflows in float16
    xs = np.linspace(8.0, 12.0, 4001)
    overflow_at = None
    for x in xs:
        y = np.exp(np.float16(x))
        if not np.isfinite(y):
            overflow_at = float(x)
            break
    out["float16_exp_overflow"] = overflow_at
    out["ln_f16_max"] = float(np.log(np.finfo(np.float16).max))
    return out


def main():
    rng = set_seed(SEED)
    print("loading CIFAR-10", flush=True)
    x_tr, y_tr, x_te, y_te = load_cifar()
    Xtr, ytr, Xval, yval, Xte, yte, stats = split_and_standardize(
        x_tr, y_tr, x_te, y_te, rng
    )
    print(
        f"split train={stats['n_train']} val={stats['n_val']} test={stats['n_test']}",
        flush=True,
    )

    # (a) gradient checks
    W0 = init_W(rng)
    Xb, yb = Xtr[:32], ytr[:32]
    svm_rel, svm_coords = grad_check(svm_loss_grad, W0, Xb, yb, 0.1, rng)
    sm_rel, sm_coords = grad_check(softmax_loss_grad, W0, Xb, yb, 0.1, rng)
    n_params = 10 * 3073
    grad_check_report = {
        "svm_max_rel": svm_rel,
        "softmax_max_rel": sm_rel,
        "svm_coords": svm_coords,
        "softmax_coords": sm_coords,
        "n_params": n_params,
        "centered_fd_forwards": 2 * n_params,
        "backprop_forwards": 1,
    }
    dump("problem3_gradcheck.json", grad_check_report)
    print(f"gradcheck SVM max rel={svm_rel:.3e} softmax={sm_rel:.3e}", flush=True)

    # (b) grids
    softmax_etas = [1e-2, 1e-3, 1e-4]
    svm_etas = [1e-3, 3e-4, 1e-4]
    lams = [1e-2, 1e-1, 1.0]

    def run_grid(name, loss_grad, etas):
        grid = []
        best = None
        for eta in etas:
            for lam in lams:
                print(f"{name} eta={eta:g} lam={lam:g}", flush=True)
                run_rng = np.random.default_rng(SEED)
                W, hist = train(loss_grad, Xtr, ytr, Xval, yval, eta, lam, run_rng)
                row = {
                    "eta": eta,
                    "lam": lam,
                    "best_val_acc": hist["best_val_acc"],
                    "last_val_acc": hist["val_acc"][-1],
                    "last_train_acc": hist["train_acc"][-1],
                    "seconds": hist["seconds"],
                }
                grid.append(row)
                if best is None or row["best_val_acc"] > best["row"]["best_val_acc"]:
                    best = {"row": row, "W": W, "hist": hist}
        return grid, best

    svm_grid, svm_best = run_grid("svm", svm_loss_grad, svm_etas)
    sm_grid, sm_best = run_grid("softmax", softmax_loss_grad, softmax_etas)

    # retrain bests with sparsity logging from a fresh identical init
    print("retrain best SVM with sparsity log", flush=True)
    W_svm, hist_svm = train(
        svm_loss_grad,
        Xtr, ytr, Xval, yval,
        svm_best["row"]["eta"], svm_best["row"]["lam"],
        np.random.default_rng(SEED),
        record_sparsity=True,
    )
    print("retrain best softmax with sparsity log", flush=True)
    W_sm, hist_sm = train(
        softmax_loss_grad,
        Xtr, ytr, Xval, yval,
        sm_best["row"]["eta"], sm_best["row"]["lam"],
        np.random.default_rng(SEED),
        record_sparsity=True,
    )

    svm_test = accuracy(W_svm, Xte, yte)
    sm_test = accuracy(W_sm, Xte, yte)
    svm_tr, svm_va = hist_svm["train_acc"][-1], hist_svm["val_acc"][-1]
    sm_tr, sm_va = hist_sm["train_acc"][-1], hist_sm["val_acc"][-1]

    dump(
        "problem3_grids.json",
        {
            "svm": svm_grid,
            "softmax": sm_grid,
            "svm_best": {**svm_best["row"], "test_acc": svm_test, "train_acc": svm_tr, "val_acc": svm_va},
            "softmax_best": {**sm_best["row"], "test_acc": sm_test, "train_acc": sm_tr, "val_acc": sm_va},
        },
    )

    plot_loss(hist_svm, "Best multiclass SVM: training loss", FIGDIR / "p3_svm_loss.pdf")
    plot_loss(hist_sm, "Best softmax: training loss", FIGDIR / "p3_softmax_loss.pdf")

    # (c) templates
    save_template_grid(
        templates_image(W_sm, stats),
        "Softmax class templates (min-max to [0, 255])",
        FIGDIR / "p3_softmax_templates.pdf",
    )
    save_template_grid(
        templates_image(W_svm, stats),
        "SVM class templates (min-max to [0, 255])",
        FIGDIR / "p3_svm_templates.pdf",
    )

    # (d) L1 vs L2
    l1_strengths = [1e-3, 1e-2, 3e-2]
    l1_rows = []
    for lam1 in l1_strengths:
        print(f"softmax L1 lam={lam1:g}", flush=True)
        Wl, histl = train(
            softmax_loss_grad,
            Xtr, ytr, Xval, yval,
            eta=1e-3, lam=0.0,
            rng=np.random.default_rng(SEED),
            l1=lam1,
        )
        w = Wl[:, :-1]
        sparsity = float((w == 0).mean() * 100.0)
        row = {
            "l1": lam1,
            "test_acc": accuracy(Wl, Xte, yte),
            "val_acc": histl["best_val_acc"],
            "sparsity_pct": sparsity,
        }
        l1_rows.append(row)
        print(f"  test={row['test_acc']:.4f} sparsity={sparsity:.2f}%", flush=True)

    l2_sparsity = float((W_sm[:, :-1] == 0).mean() * 100.0)
    dump(
        "problem3_l1.json",
        {"l1": l1_rows, "best_l2_sparsity_pct": l2_sparsity, "best_l2_test": sm_test},
    )
    plot_l1(
        l1_rows + [{"sparsity_pct": l2_sparsity, "test_acc": sm_test, "l1": "L2"}],
        FIGDIR / "p3_l1_sparsity.pdf",
    )

    # (e)
    f16 = float16_softmax_study()
    dump("problem3_float16.json", f16)
    print("float16 overflow at", f16["float16_exp_overflow"], flush=True)

    plot_sparsity(
        hist_svm["grad_zero_frac"],
        hist_sm["grad_zero_frac"],
        FIGDIR / "p3_grad_sparsity.pdf",
    )
    dump(
        "problem3_sparsity_curves.json",
        {"svm": hist_svm["grad_zero_frac"], "softmax": hist_sm["grad_zero_frac"]},
    )

    n_params = 10 * 3073
    footprint_kib = n_params * 4 / 1024.0
    dump(
        "problem3_summary.json",
        {
            "seed": SEED,
            "numpy": np.__version__,
            "device": "local CPU, 32 cores",
            "n_params": n_params,
            "fp32_kib": footprint_kib,
            "fits_128kib": footprint_kib <= 128.0,
            "gradcheck": grad_check_report,
            "svm_best": {**svm_best["row"], "test_acc": svm_test},
            "softmax_best": {**sm_best["row"], "test_acc": sm_test},
            "l1": l1_rows,
            "l2_sparsity_pct": l2_sparsity,
            "float16": f16,
        },
    )
    np.savez(RESDIR / "problem3_weights.npz", svm=W_svm, softmax=W_sm)
    print("problem 3 done", flush=True)


if __name__ == "__main__":
    main()
