"""
EE 511 Homework 1, Problem 4
LeNet-5 on MNIST / Fashion-MNIST, static analysis, ablations, SVD compression.

Global seed 511 unless a sub-run says otherwise.
"""

from __future__ import annotations

import json
import random
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 511
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
FASHION_MEAN, FASHION_STD = 0.2860, 0.3530
EPOCHS = 15
HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
CKPTDIR = HERE / "checkpoints"
RESDIR = HERE / "results"
for d in (FIGDIR, CKPTDIR, RESDIR):
    d.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
MNIST_CLASSES = [str(i) for i in range(10)]
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LeNet5(nn.Module):
    def __init__(
        self,
        activation: str = "tanh",
        pooling: str = "avg",
        pad_input: bool = True,
        c5_activation: bool = True,
    ):
        super().__init__()
        act = nn.Tanh if activation == "tanh" else nn.ReLU
        pool = nn.AvgPool2d if pooling == "avg" else nn.MaxPool2d
        self.pad_input = pad_input
        self.c5_k = 5 if pad_input else 4

        self.c1 = nn.Conv2d(1, 6, 5, stride=1, padding=0)
        self.a1 = act()
        self.s2 = pool(2, stride=2)
        self.c3 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.a3 = act()
        self.s4 = pool(2, stride=2)
        self.c5 = nn.Conv2d(16, 120, self.c5_k, stride=1, padding=0)
        self.a5 = act() if c5_activation else nn.Identity()
        self.f6 = nn.Linear(120, 84)
        self.a6 = act()
        self.out = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.c1(x))
        x = self.s2(x)
        x = self.a3(self.c3(x))
        x = self.s4(x)
        x = self.a5(self.c5(x))
        x = torch.flatten(x, 1)
        x = self.a6(self.f6(x))
        return self.out(x)


def make_transform(mean: float, std: float, pad: bool) -> transforms.Compose:
    ops = []
    if pad:
        ops.append(transforms.Pad(2, fill=0))
    ops += [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    return transforms.Compose(ops)


def get_loaders(
    dataset_name: str,
    batch_size: int,
    pad: bool,
    seed: int,
    mean: float,
    std: float,
):
    tfm = make_transform(mean, std, pad)
    root = HERE / "data"
    if dataset_name == "mnist":
        train_full = datasets.MNIST(root, train=True, download=True, transform=tfm)
        test_set = datasets.MNIST(root, train=False, download=True, transform=tfm)
    else:
        train_full = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
        test_set = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)

    n_train = int(0.9 * len(train_full))
    n_val = len(train_full) - n_train
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, test_set


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss_sum += crit(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n


def train_one(
    name: str,
    *,
    seed: int = SEED,
    activation: str = "tanh",
    pooling: str = "avg",
    lr: float = 0.01,
    batch_size: int = 64,
    pad_input: bool = True,
    c5_activation: bool = True,
    dataset_name: str = "mnist",
    mean: float = MNIST_MEAN,
    std: float = MNIST_STD,
    epochs: int = EPOCHS,
) -> dict:
    set_seed(seed)
    model = LeNet5(activation, pooling, pad_input, c5_activation).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    train_loader, val_loader, test_loader, _ = get_loaders(
        dataset_name, batch_size, pad_input, seed, mean, std
    )

    history = []
    best_val_acc = -1.0
    best_state = None
    best_epoch = 1
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        run_loss, run_correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        epoch_s = time.perf_counter() - t0
        tr_loss, tr_acc = run_loss / n, run_correct / n
        va_loss, va_acc = evaluate(model, val_loader)
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "epoch_s": epoch_s,
        }
        history.append(row)
        print(
            f"[{name} seed={seed}] ep {epoch:02d}  "
            f"tr {tr_loss:.4f}/{tr_acc:.4f}  va {va_loss:.4f}/{va_acc:.4f}  {epoch_s:.1f}s"
        )
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    te_loss, te_acc = evaluate(model, test_loader)
    ckpt_path = CKPTDIR / f"{name}_seed{seed}.pt"
    torch.save(best_state, ckpt_path)
    result = {
        "name": name,
        "seed": seed,
        "activation": activation,
        "pooling": pooling,
        "lr": lr,
        "batch_size": batch_size,
        "pad_input": pad_input,
        "c5_activation": c5_activation,
        "dataset": dataset_name,
        "mean": mean,
        "std": std,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": te_loss,
        "test_acc": te_acc,
        "mean_epoch_s": float(np.mean([r["epoch_s"] for r in history])),
        "ckpt": str(ckpt_path),
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }
    (RESDIR / f"{name}_seed{seed}.json").write_text(json.dumps(result, indent=2))
    print(f"[{name}] best ep {best_epoch} val_acc={best_val_acc:.4f} test_acc={te_acc:.4f}")
    return result


def layer_static_table(model: LeNet5) -> dict:
    named = {n: p.numel() for n, p in model.named_parameters()}
    return {
        "C1": named["c1.weight"] + named["c1.bias"],
        "C3": named["c3.weight"] + named["c3.bias"],
        "C5": named["c5.weight"] + named["c5.bias"],
        "F6": named["f6.weight"] + named["f6.bias"],
        "Out": named["out.weight"] + named["out.bias"],
        "total": sum(named.values()),
    }


def check_border(loader: DataLoader) -> dict:
    x, _ = next(iter(loader))
    img = x[0, 0]
    border = torch.cat([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
    # interior background-ish: corners of the 28x28 content after pad=2 are still often bg
    return {
        "border_mean": float(border.mean()),
        "corner": float(img[0, 0]),
        "center": float(img[14, 14]),
        "shape": list(img.shape),
    }


@torch.no_grad()
def test_details(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    ys, ps, logits_all = [], [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        ys.append(y)
        ps.append(logits.argmax(1).cpu())
        logits_all.append(logits.cpu())
    y = torch.cat(ys)
    p = torch.cat(ps)
    cm = np.zeros((10, 10), dtype=int)
    for yi, pi in zip(y.tolist(), p.tolist()):
        cm[yi, pi] += 1
    prec, rec, f1 = [], [], []
    for c in range(10):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        prec.append(pr)
        rec.append(rc)
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"cm": cm.tolist(), "precision": prec, "recall": rec, "f1": f1}


def plot_curves(history: list[dict], stem: str) -> None:
    ep = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(ep, [r["train_loss"] for r in history], "o-", label="train loss")
    ax.plot(ep, [r["val_loss"] for r in history], "s--", label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("LeNet-5 baseline: loss vs epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{stem}_loss.pdf")
    fig.savefig(FIGDIR / f"{stem}_loss.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(ep, [r["train_acc"] for r in history], "o-", label="train acc")
    ax.plot(ep, [r["val_acc"] for r in history], "s--", label="val acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("LeNet-5 baseline: accuracy vs epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{stem}_acc.pdf")
    fig.savefig(FIGDIR / f"{stem}_acc.png", dpi=160)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, labels: list[str], stem: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(10), labels, rotation=45, ha="right")
    ax.set_yticks(range(10), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{stem}.pdf")
    fig.savefig(FIGDIR / f"{stem}.png", dpi=160)
    plt.close(fig)


def plot_misclassified(model: nn.Module, test_set, mean: float, std: float, stem: str) -> None:
    model.eval()
    shown = 0
    fig, axes = plt.subplots(2, 4, figsize=(8.4, 4.4))
    axes = axes.ravel()
    loader = DataLoader(test_set, batch_size=256, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            pred = logits.argmax(1).cpu()
            for i in range(x.size(0)):
                if pred[i] == y[i]:
                    continue
                img = x[i, 0] * std + mean
                ax = axes[shown]
                ax.imshow(img.numpy(), cmap="gray")
                ax.set_title(f"true {y[i].item()} / pred {pred[i].item()}", fontsize=8)
                ax.axis("off")
                shown += 1
                if shown == 8:
                    fig.tight_layout()
                    fig.savefig(FIGDIR / f"{stem}.pdf")
                    fig.savefig(FIGDIR / f"{stem}.png", dpi=160)
                    plt.close(fig)
                    return


def svd_compress_and_eval(ckpt: str, test_loader: DataLoader, ranks: list[int]) -> dict:
    model = LeNet5().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    W5 = model.c5.weight.detach().cpu().numpy().reshape(120, 400)
    W6 = model.f6.weight.detach().cpu().numpy()  # 84 x 120
    U5, S5, Vt5 = np.linalg.svd(W5, full_matrices=False)
    U6, S6, Vt6 = np.linalg.svd(W6, full_matrices=False)

    base_loss, base_acc = evaluate(model, test_loader)
    rows = []
    for r in ranks:
        r5 = min(r, 120, 400)
        r6 = min(r, 84, 120)
        W5r = (U5[:, :r5] * S5[:r5]) @ Vt5[:r5, :]
        W6r = (U6[:, :r6] * S6[:r6]) @ Vt6[:r6, :]
        m = LeNet5().to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        with torch.no_grad():
            m.c5.weight.copy_(torch.from_numpy(W5r.reshape(120, 16, 5, 5)))
            m.f6.weight.copy_(torch.from_numpy(W6r))
        loss, acc = evaluate(m, test_loader)
        p5 = r5 * (120 + 400)
        p6 = r6 * (84 + 120)
        rows.append(
            {
                "r": r,
                "r_c5": r5,
                "r_f6": r6,
                "test_acc": acc,
                "test_loss": loss,
                "factored_params": p5 + p6,
            }
        )
        print(f"[svd] r={r:3d} acc={acc:.4f} factored_params={p5+p6}")

    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    rs = [row["r"] for row in rows]
    ax1.plot(rs, [row["test_acc"] for row in rows], "o-", label="compressed test acc")
    ax1.axhline(base_acc, color="C1", linestyle="--", label="uncompressed baseline")
    ax1.set_xlabel("rank $r$")
    ax1.set_ylabel("test accuracy")
    ax1.set_title("Post-hoc SVD compression of C5 and F6")
    ax2 = ax1.twinx()
    ax2.plot(rs, [row["factored_params"] for row in rows], "s--", color="C2", label="factored params")
    ax2.set_ylabel("factored parameter count")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "svd_acc_vs_rank.pdf")
    fig.savefig(FIGDIR / "svd_acc_vs_rank.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.semilogy(np.arange(1, len(S5) + 1), S5, "o-", markersize=3, label="C5 (120×400)")
    ax.semilogy(np.arange(1, len(S6) + 1), S6, "s-", markersize=3, label="F6 (84×120)")
    ax.set_xlabel("singular-value index")
    ax.set_ylabel("singular value")
    ax.set_title("Singular-value spectra of C5 and F6 (log $y$)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "svd_spectra.pdf")
    fig.savefig(FIGDIR / "svd_spectra.png", dpi=160)
    plt.close(fig)

    return {
        "baseline_acc": base_acc,
        "baseline_params_c5_f6": 120 * 400 + 84 * 120,
        "rows": rows,
        "S5": S5.tolist(),
        "S6": S6.tolist(),
    }


def main() -> None:
    print(f"PyTorch {torch.__version__}, device={DEVICE}, CPU 32 cores, seed={SEED}")
    set_seed(SEED)
    model = LeNet5()
    print("framework param counts:", layer_static_table(model))

    # one padded tensor check
    tfm = make_transform(MNIST_MEAN, MNIST_STD, pad=True)
    raw = datasets.MNIST(HERE / "data", train=True, download=True, transform=tfm)
    x0, _ = raw[0]
    border = torch.cat([x0[0, 0, :], x0[0, -1, :], x0[0, :, 0], x0[0, :, -1]])
    print("padded shape", tuple(x0.shape), "border mean", float(border.mean()), "corner", float(x0[0, 0, 0]))

    runs = []
    runs.append(train_one("baseline", seed=511))
    runs.append(train_one("baseline_s2", seed=512))
    runs.append(train_one("baseline_s3", seed=513))
    runs.append(train_one("relu", activation="relu"))
    runs.append(train_one("maxpool", pooling="max"))
    runs.append(train_one("lr_0.1", lr=0.1))
    runs.append(train_one("lr_0.001", lr=0.001))
    runs.append(train_one("bs_16", batch_size=16))
    runs.append(train_one("bs_256", batch_size=256))

    baseline = json.loads((RESDIR / "baseline_seed511.json").read_text())
    plot_curves(baseline["history"], "lenet_baseline")

    # reload best baseline for test figures
    set_seed(SEED)
    model = LeNet5().to(DEVICE)
    model.load_state_dict(torch.load(baseline["ckpt"], map_location=DEVICE, weights_only=True))
    _, val_loader, test_loader, test_set = get_loaders(
        "mnist", 64, True, SEED, MNIST_MEAN, MNIST_STD
    )
    details = test_details(model, test_loader)
    plot_confusion(np.array(details["cm"]), MNIST_CLASSES, "cm_mnist", "MNIST confusion matrix")
    plot_misclassified(model, test_set, MNIST_MEAN, MNIST_STD, "misclassified_mnist")
    (RESDIR / "baseline_test_details.json").write_text(json.dumps(details, indent=2))

    # pick best config by validation acc among the 511-seed runs (not the extra seeds)
    candidates = [r for r in runs if r["seed"] == 511 and r["dataset"] == "mnist"]
    best = max(candidates, key=lambda r: r["best_val_acc"])
    print("best config by val acc:", best["name"], best["best_val_acc"])

    fashion = train_one(
        "fashion",
        seed=SEED,
        activation=best["activation"],
        pooling=best["pooling"],
        lr=best["lr"],
        batch_size=best["batch_size"],
        pad_input=best["pad_input"],
        c5_activation=best["c5_activation"],
        dataset_name="fashion",
        mean=FASHION_MEAN,
        std=FASHION_STD,
    )
    fmodel = LeNet5(
        best["activation"], best["pooling"], best["pad_input"], best["c5_activation"]
    ).to(DEVICE)
    fmodel.load_state_dict(torch.load(fashion["ckpt"], map_location=DEVICE, weights_only=True))
    _, _, ftest, _ = get_loaders("fashion", 64, best["pad_input"], SEED, FASHION_MEAN, FASHION_STD)
    fdet = test_details(fmodel, ftest)
    plot_confusion(np.array(fdet["cm"]), FASHION_CLASSES, "cm_fashion", "Fashion-MNIST confusion matrix")
    (RESDIR / "fashion_test_details.json").write_text(json.dumps(fdet, indent=2))

    svd = svd_compress_and_eval(baseline["ckpt"], test_loader, [1, 2, 4, 8, 16, 32, 64, 120])
    (RESDIR / "svd.json").write_text(json.dumps(svd, indent=2))

    summary = {
        "device": "local CPU, 32 cores",
        "torch": torch.__version__,
        "seed": SEED,
        "mnist_norm": [MNIST_MEAN, MNIST_STD],
        "fashion_norm": [FASHION_MEAN, FASHION_STD],
        "framework_params": layer_static_table(LeNet5()),
        "runs": [
            {
                k: r[k]
                for k in (
                    "name", "seed", "test_acc", "test_loss", "best_val_acc",
                    "best_epoch", "mean_epoch_s", "lr", "batch_size",
                    "activation", "pooling", "dataset",
                )
            }
            for r in runs + [fashion]
        ],
        "best_name": best["name"],
    }
    (RESDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done. summary written.")


if __name__ == "__main__":
    main()
