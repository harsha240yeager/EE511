# EE 511 Homework 1 code

Seed `511` everywhere. Local CPU. Every number in the report comes from these scripts.

## Dependencies

```
python -m pip install numpy matplotlib torch torchvision
```

## Exact commands

From this directory (the unzipped folder):

```
python problem3.py
python problem4.py
```

- `problem3.py` — Problem 3 (PCA / SVD). Writes plots to `figures/`.
- `problem4.py` — Problem 4 (LeNet-5). Trains on CPU, writes plots to `figures/` and metrics to `results/`. First run downloads MNIST / Fashion-MNIST.

Do not tune on the test set. Device is local CPU (no AWS).
