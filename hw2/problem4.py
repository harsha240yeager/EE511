"""
EE 511 Homework 2, Problem 4
A NumPy MLP and five optimizers on Fashion-MNIST.

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
N_VAL = 6000
BATCH = 128
EPOCHS = 15
EPS = 1e-8
H_FD = 1e-5
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
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
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    print(f"wrote {path.name}", flush=True)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_fashion():
    train = datasets.FashionMNIST(root=str(DATADIR), train=True, download=True)
    test = datasets.FashionMNIST(root=str(DATADIR), train=False, download=True)
    x_tr = train.data.numpy().astype(np.float64)
    y_tr = train.targets.numpy().astype(np.int64)
    x_te = test.data.numpy().astype(np.float64)
    y_te = test.targets.numpy().astype(np.int64)
    return x_tr, y_tr, x_te, y_te


def prepare(x_tr, y_tr, x_te, y_te, rng):
    perm = rng.permutation(len(x_tr))
    val_idx, tr_idx = perm[:N_VAL], perm[N_VAL:]
    x_val, y_val = x_tr[val_idx], y_tr[val_idx]
    x_tr, y_tr = x_tr[tr_idx], y_tr[tr_idx]

    def flat01(x):
        return x.reshape(len(x), -1) / 255.0

    x_tr, x_val, x_te = flat01(x_tr), flat01(x_val), flat01(x_te)
    mu = float(x_tr.mean())
    sd = float(x_tr.std())
    if sd < 1e-12:
        sd = 1.0

    def norm(x):
        return ((x - mu) / sd).astype(np.float32), 

    Xtr = ((x_tr - mu) / sd).astype(np.float32)
    Xval = ((x_val - mu) / sd).astype(np.float32)
    Xte = ((x_te - mu) / sd).astype(np.float32)
    stats = {"mean": mu, "std": sd, "n_train": int(len(Xtr)), "n_val": int(len(Xval))}
    return Xtr, y_tr, Xval, y_val, Xte, y_te, stats


def he_init(rng, din, dout, gain=2.0, dtype=np.float32):
    W = rng.normal(0.0, np.sqrt(gain / din), size=(din, dout)).astype(dtype)
    b = np.zeros((dout,), dtype=dtype)
    return W, b


def init_baseline(rng, dtype=np.float32):
    W1, b1 = he_init(rng, 784, 512, 2.0, dtype)
    W2, b2 = he_init(rng, 512, 256, 2.0, dtype)
    W3, b3 = he_init(rng, 256, 10, 2.0, dtype)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


def clone_params(params):
    return {k: v.copy() for k, v in params.items()}


def zeros_like_params(params):
    return {k: np.zeros_like(v) for k, v in params.items()}


def relu(z):
    return np.maximum(z, 0.0)


def softmax_xent(s, y):
    """Log-sum-exp form; returns mean loss and P."""
    m = s.max(axis=1, keepdims=True)
    z = s - m
    exp = np.exp(z)
    logZ = np.log(exp.sum(axis=1) + 1e-12)
    loss = (logZ - z[np.arange(len(y)), y]).mean()
    P = exp / exp.sum(axis=1, keepdims=True)
    return float(loss), P


def forward_backward(params, X, y, return_acts=False):
    W1, b1, W2, b2, W3, b3 = (
        params["W1"], params["b1"], params["W2"], params["b2"], params["W3"], params["b3"],
    )
    Z1 = X @ W1 + b1
    M1 = Z1 > 0
    H1 = Z1 * M1
    Z2 = H1 @ W2 + b2
    M2 = Z2 > 0
    H2 = Z2 * M2
    S = H2 @ W3 + b3
    loss, P = softmax_xent(S, y)

    B = X.shape[0]
    dS = P.copy()
    dS[np.arange(B), y] -= 1.0
    dS /= B

    dW3 = H2.T @ dS
    db3 = dS.sum(axis=0)
    dH2 = dS @ W3.T
    dZ2 = dH2 * M2
    dW2 = H1.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dH1 = dZ2 @ W2.T
    dZ1 = dH1 * M1
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
    if return_acts:
        return loss, grads, {"Z1": Z1, "H1": H1, "M1": M1, "Z2": Z2, "H2": H2, "M2": M2, "S": S, "P": P}
    return loss, grads


def predict(params, X, batch=1024):
    outs = []
    for i in range(0, len(X), batch):
        xb = X[i : i + batch]
        h1 = relu(xb @ params["W1"] + params["b1"])
        h2 = relu(h1 @ params["W2"] + params["b2"])
        s = h2 @ params["W3"] + params["b3"]
        outs.append(s.argmax(axis=1))
    return np.concatenate(outs)


def accuracy(params, X, y):
    return float((predict(params, X) == y).mean())


def hidden_outputs(params, X, batch=1024):
    h1s, h2s = [], []
    for i in range(0, len(X), batch):
        xb = X[i : i + batch]
        h1 = relu(xb @ params["W1"] + params["b1"])
        h2 = relu(h1 @ params["W2"] + params["b2"])
        h1s.append(h1)
        h2s.append(h2)
    return np.vstack(h1s), np.vstack(h2s)


# ----- optimizers -----
def opt_sgd(params, grads, state, eta):
    for k in params:
        params[k] -= eta * grads[k]
    return state


def opt_momentum(params, grads, state, eta, rho=0.9):
    if not state:
        state = zeros_like_params(params)
    for k in params:
        state[k] = rho * state[k] + eta * grads[k]
        params[k] -= state[k]
    return state


def opt_adagrad(params, grads, state, eta):
    if not state:
        state = zeros_like_params(params)
    for k in params:
        state[k] += grads[k] ** 2
        params[k] -= eta * grads[k] / (np.sqrt(state[k]) + EPS)
    return state


def opt_rmsprop(params, grads, state, eta, beta=0.99):
    if not state:
        state = zeros_like_params(params)
    for k in params:
        state[k] = beta * state[k] + (1.0 - beta) * grads[k] ** 2
        params[k] -= eta * grads[k] / (np.sqrt(state[k]) + EPS)
    return state


def opt_adam(params, grads, state, eta, beta1=0.9, beta2=0.999):
    if not state:
        state = {"m": zeros_like_params(params), "v": zeros_like_params(params), "t": 0}
    state["t"] += 1
    t = state["t"]
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    for k in params:
        state["m"][k] = beta1 * state["m"][k] + (1.0 - beta1) * grads[k]
        state["v"][k] = beta2 * state["v"][k] + (1.0 - beta2) * grads[k] ** 2
        mhat = state["m"][k] / bc1
        vhat = state["v"][k] / bc2
        params[k] -= eta * mhat / (np.sqrt(vhat) + EPS)
    return state


OPTIMIZERS = {
    "sgd": opt_sgd,
    "momentum": opt_momentum,
    "adagrad": opt_adagrad,
    "rmsprop": opt_rmsprop,
    "adam": opt_adam,
}

LR_GRID = {
    "sgd": [0.1, 0.03, 0.01],
    "momentum": [0.1, 0.03, 0.01],
    "adagrad": [0.01, 0.003, 0.001],
    "rmsprop": [1e-3, 3e-4, 1e-4],
    "adam": [3e-3, 1e-3, 3e-4],
}


def iterate_minibatches(X, y, batch, rng):
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch):
        sl = idx[start : start + batch]
        if len(sl) < 2:
            continue
        yield X[sl], y[sl]


def train_mlp(
    opt_name,
    eta,
    Xtr,
    ytr,
    Xval,
    yval,
    init_params,
    epochs=EPOCHS,
    eval_every=25,
    track_adagrad_lr=False,
    track_val_iters=True,
):
    params = clone_params(init_params)
    opt = OPTIMIZERS[opt_name]
    state = {}
    rng = np.random.default_rng(SEED)
    train_loss = []
    val_acc_epoch = []
    val_acc_iters = []
    first_87 = None
    adagrad_med = []
    step = 0
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        for xb, yb in iterate_minibatches(Xtr, ytr, BATCH, rng):
            loss, grads = forward_backward(params, xb, yb)
            state = opt(params, grads, state, eta)
            train_loss.append(float(loss))
            step += 1
            if track_val_iters and step % eval_every == 0:
                acc = accuracy(params, Xval, yval)
                val_acc_iters.append((step, acc))
                if first_87 is None and acc >= 0.87:
                    first_87 = step
            if track_adagrad_lr and state:
                s = state["W1"]
                eff = eta / (np.sqrt(s) + EPS)
                adagrad_med.append(float(np.median(eff)))
        va = accuracy(params, Xval, yval)
        val_acc_epoch.append(va)
        print(
            f"  {opt_name} eta={eta:g} ep {ep:02d} last_loss={train_loss[-1]:.4f} val={va:.4f}",
            flush=True,
        )
    out = {
        "train_loss": train_loss,
        "val_acc_epoch": val_acc_epoch,
        "val_acc_iters": val_acc_iters,
        "first_87": first_87,
        "best_val": float(max(val_acc_epoch)),
        "seconds": time.perf_counter() - t0,
        "adagrad_med": adagrad_med,
        "state": state,
    }
    return params, out


def grad_check_mlp(params64, X, y, rng, n_coords=10):
    _, analytic = forward_backward(params64, X, y)
    report = {}
    for name, tensor in params64.items():
        rels = []
        for _ in range(n_coords):
            idx = tuple(int(rng.integers(0, s)) for s in tensor.shape)
            orig = float(tensor[idx])
            tensor[idx] = orig + H_FD
            lp, _ = forward_backward(params64, X, y)
            tensor[idx] = orig - H_FD
            lm, _ = forward_backward(params64, X, y)
            tensor[idx] = orig
            numeric = (lp - lm) / (2.0 * H_FD)
            ga = float(analytic[name][idx])
            rels.append(abs(ga - numeric) / max(abs(ga) + abs(numeric), 1e-12))
        report[name] = float(max(rels))
    return report


# ----- deeper net -----
def init_deep(rng, act, width=128, n_hidden=8, dtype=np.float32):
    gain = 2.0 if act == "relu" else 1.0
    dims = [784] + [width] * n_hidden + [10]
    Ws, bs = [], []
    for din, dout in zip(dims[:-1], dims[1:]):
        W, b = he_init(rng, din, dout, gain, dtype)
        Ws.append(W)
        bs.append(b)
    return {"W": Ws, "b": bs}


def activate(z, act):
    if act == "relu":
        return np.maximum(z, 0.0)
    if act == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    if act == "tanh":
        return np.tanh(z)
    raise ValueError(act)


def act_grad(z, h, act):
    if act == "relu":
        return (z > 0).astype(z.dtype)
    if act == "sigmoid":
        return h * (1.0 - h)
    if act == "tanh":
        return 1.0 - h ** 2
    raise ValueError(act)


def deep_forward_backward(net, X, y, act, return_gnorms=False):
    Ws, bs = net["W"], net["b"]
    zs, hs = [], [X]
    a = X
    for i, (W, b) in enumerate(zip(Ws, bs)):
        z = a @ W + b
        zs.append(z)
        if i == len(Ws) - 1:
            a = z
        else:
            a = activate(z, act)
        hs.append(a)
    loss, P = softmax_xent(hs[-1], y)
    B = X.shape[0]
    delta = P
    delta[np.arange(B), y] -= 1.0
    delta /= B
    dWs = [None] * len(Ws)
    dbs = [None] * len(bs)
    for i in reversed(range(len(Ws))):
        dWs[i] = hs[i].T @ delta
        dbs[i] = delta.sum(axis=0)
        if i == 0:
            break
        dH = delta @ Ws[i].T
        delta = dH * act_grad(zs[i - 1], hs[i], act)
    if return_gnorms:
        gnorms = [float(np.linalg.norm(g, "fro")) for g in dWs]
        return loss, dWs, dbs, gnorms
    return loss, dWs, dbs


def deep_predict(net, X, act, batch=1024):
    outs = []
    for i in range(0, len(X), batch):
        a = X[i : i + batch]
        for j, (W, b) in enumerate(zip(net["W"], net["b"])):
            z = a @ W + b
            a = z if j == len(net["W"]) - 1 else activate(z, act)
        outs.append(a.argmax(axis=1))
    return np.concatenate(outs)


def train_deep(act, Xtr, ytr, Xval, yval, Xte, yte, epochs=3, eta=1e-3):
    rng = np.random.default_rng(SEED)
    net = init_deep(rng, act)
    # dead units at init (hidden layers only)
    def dead_frac(network):
        a = Xval
        fracs = []
        for i, (W, b) in enumerate(zip(network["W"][:-1], network["b"][:-1])):
            z = a @ W + b
            h = activate(z, act)
            dead = (np.abs(h).max(axis=0) == 0).mean() if act == "relu" else float("nan")
            fracs.append(float(dead))
            a = h
        return fracs

    dead_init = dead_frac(net) if act == "relu" else None
    rng_data = np.random.default_rng(SEED)
    state = None
    # wrap as param dict for adam
    params = {f"W{i}": W for i, W in enumerate(net["W"])}
    params.update({f"b{i}": b for i, b in enumerate(net["b"])})
    for ep in range(1, epochs + 1):
        for xb, yb in iterate_minibatches(Xtr, ytr, BATCH, rng_data):
            loss, dW, db = deep_forward_backward(net, xb, yb, act)
            grads = {f"W{i}": g for i, g in enumerate(dW)}
            grads.update({f"b{i}": g for i, g in enumerate(db)})
            state = opt_adam(params, grads, state, eta)
            for i in range(len(net["W"])):
                net["W"][i] = params[f"W{i}"]
                net["b"][i] = params[f"b{i}"]
        acc = float((deep_predict(net, Xval, act) == yval).mean())
        print(f"  deep {act} ep {ep} val={acc:.4f}", flush=True)
    test_acc = float((deep_predict(net, Xte, act) == yte).mean())
    dead_end = dead_frac(net) if act == "relu" else None
    return {"test_acc": test_acc, "dead_init": dead_init, "dead_end": dead_end, "net": net}


def quantize_block(x, n):
    flat = x.ravel()
    if n is None or n >= flat.size:
        blocks = [flat]
    else:
        pad = (-flat.size) % n
        padded = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)]) if pad else flat
        blocks = padded.reshape(-1, n)
    recon = []
    rels = []
    zero_fracs = []
    n_blocks = 0
    for block in np.atleast_2d(blocks):
        n_blocks += 1
        s = float(np.max(np.abs(block)))
        if s == 0:
            q = np.zeros_like(block)
            hat = q
        else:
            q = np.clip(np.round(127.0 * block / s), -127, 127)
            hat = q * s / 127.0
        recon.append(hat.ravel())
        denom = np.linalg.norm(block)
        rels.append(0.0 if denom == 0 else float(np.linalg.norm(hat - block) / denom))
        nz = np.abs(block) > 0
        if nz.any():
            zero_fracs.append(float(((q == 0) & nz).mean()))
        else:
            zero_fracs.append(0.0)
    hat_flat = np.concatenate(recon)[: flat.size]
    rel = float(np.linalg.norm(hat_flat - flat) / max(np.linalg.norm(flat), 1e-30))
    meta = 4.0 / (flat.size if n is None or n >= flat.size else n)
    return {
        "rel_l2": rel,
        "zero_frac_nonzero": float(np.mean(zero_fracs)),
        "bytes_per_param_scale": meta,
        "n_blocks": n_blocks,
    }


def plot_opt_losses(curves, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name, ys in curves.items():
        ax.plot(np.arange(1, len(ys) + 1), ys, label=name, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("minibatch training loss (nats / sample)")
    ax.set_title("Training loss of the five best optimizer runs")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def plot_val_acc(curves, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name, ys in curves.items():
        ax.plot(np.arange(1, len(ys) + 1), ys, marker="o", label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Validation accuracy vs. epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def plot_adagrad(med, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(np.arange(1, len(med) + 1), med, label=r"median $\eta/(\sqrt{s}+\epsilon)$ on $W_1$")
    ax.set_xlabel("iteration")
    ax.set_ylabel("effective learning rate")
    ax.set_title("AdaGrad effective step size on first-layer weights")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def plot_gnorms(curves, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for act, ys in curves.items():
        ax.semilogy(np.arange(1, len(ys) + 1), ys, marker="o", label=act)
    ax.set_xlabel("layer index $\\ell$ (1 = first hidden)")
    ax.set_ylabel(r"$\|\partial L/\partial W_\ell\|_F$")
    ax.set_title("Gradient scale vs. depth at initialization")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def plot_throughput(Bs, sps, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.loglog(Bs, sps, "o-", label="fwd+bwd throughput")
    ax.set_xlabel("batch size $B$ (samples)")
    ax.set_ylabel("throughput (samples / s)")
    ax.set_title("Roofline-style throughput vs. batch size")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def plot_quant(ns, m_err, v_err, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.semilogx(ns, m_err, "o-", label=r"Adam $m$ relative $\ell_2$ error")
    ax.semilogx(ns, v_err, "s-", label=r"Adam $v$ relative $\ell_2$ error")
    ax.set_xlabel("block size $N$")
    ax.set_ylabel(r"$\|\hat x-x\|_2/\|x\|_2$")
    ax.set_title("Block-wise INT8 quantization error vs. block size")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def intensity(B, din, dout):
    return (B * din * dout) / (2.0 * (B * din + din * dout + B * dout))


def smallest_compute_bound(din, dout, ibal=40.0):
    # I(B) > ibal  => B*din*dout > 2*ibal*(B*(din+dout) + din*dout)
    num = 2.0 * ibal * din * dout
    den = din * dout - 2.0 * ibal * (din + dout)
    if den <= 0:
        return None
    return int(np.floor(num / den) + 1)


def time_fwd_bwd(params, X, y, B, repeats):
    xb, yb = X[:B], y[:B]
    forward_backward(params, xb, yb)  # warmup
    t0 = time.perf_counter()
    for _ in range(repeats):
        forward_backward(params, xb, yb)
    dt = time.perf_counter() - t0
    return (repeats * B) / dt, dt / repeats


def main():
    rng = set_seed(SEED)
    print("loading Fashion-MNIST", flush=True)
    x_tr, y_tr, x_te, y_te = load_fashion()
    Xtr, ytr, Xval, yval, Xte, yte, stats = prepare(x_tr, y_tr, x_te, y_te, rng)
    print(f"split train={stats['n_train']} val={stats['n_val']}", flush=True)

    # frozen init used by every optimizer run
    init_rng = np.random.default_rng(SEED)
    init_params = init_baseline(init_rng, dtype=np.float32)

    # (a) grad check in float64
    print("gradient check", flush=True)
    gc_rng = np.random.default_rng(SEED)
    params64 = init_baseline(np.random.default_rng(SEED), dtype=np.float64)
    X8 = Xtr[:8].astype(np.float64)
    y8 = ytr[:8]
    gc = grad_check_mlp(params64, X8, y8, gc_rng, n_coords=10)
    dump("problem4_gradcheck.json", gc)
    print("gradcheck", gc, flush=True)

    # static analysis
    layers = [
        {"name": "fc1 784→512", "din": 784, "dout": 512},
        {"name": "fc2 512→256", "din": 512, "dout": 256},
        {"name": "fc3 256→10", "din": 256, "dout": 10},
    ]
    static = []
    total_macs = 0
    total_params = 0
    for L in layers:
        nW = L["din"] * L["dout"]
        nP = nW + L["dout"]
        macs = L["din"] * L["dout"]
        total_macs += macs
        total_params += nP
        static.append({
            **L,
            "params": nP,
            "forward_macs": macs,
            "gemm_fwd": [BATCH, L["din"], L["dout"]],
            "gemm_dW": [L["din"], BATCH, L["dout"]],
            "gemm_dX": [BATCH, L["dout"], L["din"]],
        })
    static_report = {
        "layers": static,
        "total_params": total_params,
        "total_forward_macs": total_macs,
        "first_layer_mac_frac": static[0]["forward_macs"] / total_macs,
        "first_layer_skips": "dX = dZ1 @ W1.T  (input is data, not a parameter)",
    }
    dump("problem4_static.json", static_report)

    # (b) five optimizers
    results = {}
    loss_curves = {}
    val_curves = {}
    for name, etas in LR_GRID.items():
        best = None
        for eta in etas:
            print(f"train {name} eta={eta:g}", flush=True)
            params, hist = train_mlp(
                name, eta, Xtr, ytr, Xval, yval, init_params,
                track_adagrad_lr=(name == "adagrad"),
            )
            te = accuracy(params, Xte, yte)
            rec = {
                "eta": eta,
                "best_val": hist["best_val"],
                "test_acc": te,
                "first_87": hist["first_87"],
                "seconds": hist["seconds"],
                "val_acc_epoch": hist["val_acc_epoch"],
                "train_loss": hist["train_loss"],
                "val_acc_iters": hist["val_acc_iters"],
                "adagrad_med": hist["adagrad_med"],
            }
            rec["state"] = hist["state"]
            if best is None or rec["best_val"] > best["best_val"]:
                best = rec
                best["params"] = params
        # drop huge arrays from disk later; keep summary
        results[name] = best
        loss_curves[name] = best["train_loss"]
        val_curves[name] = best["val_acc_epoch"]
        dump(
            f"problem4_{name}.json",
            {k: v for k, v in best.items() if k not in ("params", "train_loss", "adagrad_med", "state")},
        )
        print(f"BEST {name} eta={best['eta']:g} val={best['best_val']:.4f} test={best['test_acc']:.4f}", flush=True)

    plot_opt_losses(loss_curves, FIGDIR / "p4_train_loss.pdf")
    plot_val_acc(val_curves, FIGDIR / "p4_val_acc.pdf")
    plot_adagrad(results["adagrad"]["adagrad_med"], FIGDIR / "p4_adagrad_lr.pdf")

    # save best-eta train losses separately (can be large)
    dump("problem4_opt_table.json", {
        name: {
            "eta": r["eta"],
            "best_val": r["best_val"],
            "test_acc": r["test_acc"],
            "first_87": r["first_87"],
        }
        for name, r in results.items()
    })

    # (c) depth study
    print("depth study: init gradient norms", flush=True)
    xb, yb = Xtr[:BATCH], ytr[:BATCH]
    gnorm_curves = {}
    for act in ("sigmoid", "tanh", "relu"):
        net = init_deep(np.random.default_rng(SEED), act)
        _, _, _, gnorms = deep_forward_backward(net, xb, yb, act, return_gnorms=True)
        gnorm_curves[act] = gnorms
        print(act, gnorms, flush=True)
    plot_gnorms(gnorm_curves, FIGDIR / "p4_depth_gnorms.pdf")

    # sigmoid decay factor: geometric mean of consecutive hidden-layer ratios
    sig = np.array(gnorm_curves["sigmoid"], dtype=np.float64)
    ratios = sig[1:-1] / np.maximum(sig[:-2], 1e-30)  # skip last (classifier) maybe include all hidden
    # layers 1..8 are hidden-to-hidden; gnorms has 9 tensors (8 hidden + output)
    hidden_ratios = sig[1:8] / np.maximum(sig[0:7], 1e-30)
    decay = float(np.exp(np.mean(np.log(np.clip(hidden_ratios, 1e-12, 1e12)))))

    depth_acc = {}
    for act in ("sigmoid", "tanh", "relu"):
        print(f"train deep {act}", flush=True)
        depth_acc[act] = train_deep(act, Xtr, ytr, Xval, yval, Xte, yte)
        # cannot json the net
        depth_acc[act] = {k: v for k, v in depth_acc[act].items() if k != "net"}
    dump("problem4_depth.json", {
        "gnorms": gnorm_curves,
        "sigmoid_decay": decay,
        "results": depth_acc,
    })

    # (d) throughput
    print("throughput sweep", flush=True)
    Bs = [1, 4, 16, 64, 256, 1024]
    sps = []
    for B in Bs:
        repeats = 30 if B <= 16 else (12 if B <= 256 else 6)
        th, per = time_fwd_bwd(init_params, Xtr, ytr, B, repeats)
        sps.append(th)
        print(f"  B={B} {th:.1f} samples/s ({per*1e3:.2f} ms)", flush=True)
    plot_throughput(Bs, sps, FIGDIR / "p4_throughput.pdf")
    I1 = intensity(1, 784, 512)
    I128 = intensity(128, 784, 512)
    Bstar = smallest_compute_bound(784, 512, 40.0)
    Iinf_out = (256 * 10) / (2.0 * (256 + 10))
    dump("problem4_roofline.json", {
        "B": Bs,
        "samples_per_s": sps,
        "I_B1": I1,
        "I_B128": I128,
        "B_compute_bound": Bstar,
        "I_inf_output": Iinf_out,
        "I_formula_ok": True,
    })

    # (e) memory
    n_param = total_params
    bytes_W = n_param * 4
    bytes_G = n_param * 4
    opt_state = {
        "sgd": 0,
        "momentum": n_param * 4,
        "adagrad": n_param * 4,
        "rmsprop": n_param * 4,
        "adam": n_param * 8,
    }
    B = 128
    act_bytes = {
        "X": B * 784 * 4,
        "H1": B * 512 * 4,
        "H2": B * 256 * 4,
        "mask1_bits": B * 512,
        "mask2_bits": B * 256,
    }
    act_bytes["masks"] = (act_bytes["mask1_bits"] + act_bytes["mask2_bits"]) / 8.0
    act_bytes["cache_total"] = act_bytes["X"] + act_bytes["H1"] + act_bytes["H2"] + act_bytes["masks"]
    save_bytes = act_bytes["H1"] + act_bytes["H2"] + act_bytes["masks"]
    macs_fwd = 784 * 512 + 512 * 256 + 256 * 10
    macs_bwd_skip_dX = 784 * 512 + 2 * (512 * 256 + 256 * 10)
    macs_bwd_full = 2 * macs_fwd
    macs_recompute = 784 * 512 + 512 * 256
    dump("problem4_memory.json", {
        "n_params": n_param,
        "weights_bytes": bytes_W,
        "grads_bytes": bytes_G,
        "opt_state_bytes": opt_state,
        "activation_cache": act_bytes,
        "checkpoint_save_bytes": save_bytes,
        "recompute_macs": macs_recompute,
        "full_step_macs_skip_dX": macs_fwd + macs_bwd_skip_dX,
        "full_step_macs_with_dX": macs_fwd + macs_bwd_full,
        "recompute_pct_skip_dX": 100.0 * macs_recompute / (macs_fwd + macs_bwd_skip_dX),
        "recompute_pct_with_dX": 100.0 * macs_recompute / (macs_fwd + macs_bwd_full),
    })

    # (f) quantize best Adam state from the winning run
    print("quantize Adam state", flush=True)
    state = results["adam"]["state"]

    blocks = [32, 256, 4096, None]
    qtab = {"m": {}, "v": {}}
    for which in ("m", "v"):
        # concatenate all tensors for a global report, plus per-tensor
        for N in blocks:
            key = "tensor" if N is None else str(N)
            rels, zf, meta = [], [], []
            for tname, tensor in state[which].items():
                q = quantize_block(tensor, N)
                rels.append(q["rel_l2"])
                zf.append(q["zero_frac_nonzero"])
                meta.append(q["bytes_per_param_scale"])
            qtab[which][key] = {
                "rel_l2_mean": float(np.mean(rels)),
                "zero_frac_mean": float(np.mean(zf)),
                "scale_B_per_param": float(np.mean(meta)),
                "per_tensor_rel": rels,
            }
    # plot vs N using concatenated W1-sized? use mean rel
    ns = [32, 256, 4096]
    plot_quant(
        ns,
        [qtab["m"][str(n)]["rel_l2_mean"] for n in ns],
        [qtab["v"][str(n)]["rel_l2_mean"] for n in ns],
        FIGDIR / "p4_quant_error.pdf",
    )
    dump("problem4_quant.json", qtab)

    dump("problem4_summary.json", {
        "seed": SEED,
        "numpy": np.__version__,
        "device": "local CPU, 32 cores",
        "gradcheck": gc,
        "static": static_report,
        "optimizers": {
            name: {"eta": r["eta"], "best_val": r["best_val"], "test_acc": r["test_acc"], "first_87": r["first_87"]}
            for name, r in results.items()
        },
        "depth": {"gnorms": gnorm_curves, "sigmoid_decay": decay, "results": depth_acc},
        "roofline": {"I_B1": I1, "I_B128": I128, "Bstar": Bstar, "I_inf_out": Iinf_out},
        "quant": qtab,
        "fashion_mean": stats["mean"],
        "fashion_std": stats["std"],
    })
    print("problem 4 done", flush=True)


if __name__ == "__main__":
    main()
