# EE 511 Homework 2 code

Seed `511` everywhere. NumPy-only models (no autograd). Local CPU.

## Dependencies

```
python -m pip install numpy matplotlib torch torchvision
```

`torch` / `torchvision` are used only to download CIFAR-10 and Fashion-MNIST.

## Exact commands

From this directory:

```
python problem3.py
python problem4.py
```

- `problem3.py` — Problem 3 (linear classifiers on CIFAR-10). Writes `figures/` and `results/`.
- `problem4.py` — Problem 4 (MLP + five optimizers on Fashion-MNIST). Writes `figures/` and `results/`.
