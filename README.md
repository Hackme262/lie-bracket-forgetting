# Does Learning Order Explain Forgetting?

[![arXiv](https://img.shields.io/badge/arXiv-2611.20087-b31b1b.svg)](https://arxiv.org/abs/2611.20087)

Code and results for the paper ([arXiv:2611.20087](https://arxiv.org/abs/2611.20087), [PDF](https://arxiv.org/pdf/2611.20087))

> **Does Learning Order Explain Forgetting? The Lie Bracket as an Early-Warning Signal for Catastrophic Forgetting Without Old Data**
> Ephraim Kofi Ayi Kwapong, Responsible Artificial Intelligence Lab (RAIL), KNUST, Ghana.

The paper tests whether the Lie bracket of two tasks' gradient fields, `H_B g_A - H_A g_B`, predicts how much an old task `A` is forgotten while a new task `B` is learned, using only a small summary stored when `A` finishes (its gradient and its top 20 Hessian eigenpairs) and none of `A`'s data. It is compared with the penalty of Elastic Weight Consolidation (EWC), a stored Taylor estimate, and re-evaluation on the old data.

## What is here

| Folder | Contents |
|---|---|
| `notebooks/` | The five Jupyter notebooks that produced every result. Saved outputs are included. |
| `results/` | Split MNIST, first run with the full bracket (uses old data). |
| `results_stored/` | Split MNIST, stored summary (no old data): three learning rates, tiny buffers, order, mechanism and curvature tests. |
| `results_fashion/` | Split Fashion-MNIST. |
| `results_cifar_feat/` | Split CIFAR-10 on frozen ResNet-18 features (old task well learned). |
| `results_cifar_feat_K50/`, `results_cifar_feat_K100/` | The same with a summary of 50 and 100 eigenpairs (fast scenario, 3 seeds). |
| `results_cifar/` | Split CIFAR-10 with a small CNN trained from pixels (old task half learned). |
| `results_seq_mnist/`, `results_seq_fashion/`, `results_seq_cifar_feat/` | Five-task sequences (added in v2 of the paper): one row per (old task, new task) event, 200 per scenario. |

Each `*.csv` file has one row per run (ordered task pair and seed), or, for the sequences, one row per (old task, new task) event. Each `*.json` file holds the rank correlations printed by the notebook. File name prefixes: `order_` (order test), `mech_` (slope and curvature test), `curv_` (extreme eigenvalues), `diag_` (diagnostics on the CNN), `buffer_` (tiny buffers).

## Notebooks

| Notebook | Setup | Writes to |
|---|---|---|
| `forgetting_mnist.ipynb` | Split MNIST, MLP 784-256-256 (tanh), full bracket | `results/` |
| `forgetting_mnist_stored.ipynb` | Split MNIST, stored summary, learning-rate sweep, per-step test, tiny buffers, order / mechanism / curvature tests | `results_stored/` |
| `forgetting_welllearned.ipynb` | Set `DATASET = "cifar_feat"` or `"fashion"` in Cell 1 | `results_cifar_feat/`, `results_fashion/` |
| `forgetting_cifar10.ipynb` | Split CIFAR-10, small CNN (about 137k weights), diagnostics | `results_cifar/` |
| `forgetting_sequence.ipynb` | Five tasks learned one after another (5 seeds x 4 orders); set `DATASET` to `mnist`, `cifar_feat` or `fashion` in Cell 1 | `results_seq_<dataset>/` |

All runs use plain SGD (no momentum), batch size 64, learning rate 0.05 unless stated, 20 ordered task pairs and 5 seeds (100 runs per setting). The two scenarios are `task` (separate outputs per task, slow forgetting) and `domain` (shared outputs, fast forgetting).

## Running

1. Install PyTorch and torchvision for your GPU, plus `numpy`, `scipy`, `pandas` and `matplotlib`. The results were produced with torch 2.5.1 (CUDA 12.1) and torchvision 0.20.1 on one NVIDIA RTX 3090.
2. In Cell 1 of each notebook, change `BASE = r"F:\forgetting"` to a folder on your machine. Data is downloaded to `BASE/data` and results are written to `BASE/results*`.
3. Run the cells in order. Each cell's heading says what it should print. The cells after the numbered ones (order, mechanism and curvature tests) depend on the earlier cells.

`forgetting_welllearned.ipynb` downloads ImageNet-pretrained ResNet-18 weights through torchvision the first time it runs and caches the CIFAR-10 features in `BASE/data`.

Runs are seeded, but exact numbers can differ slightly across GPUs and library versions.

## Citation

```
@article{kwapong2026order,
  title   = {Does Learning Order Explain Forgetting? The Lie Bracket as an Early-Warning Signal for Catastrophic Forgetting Without Old Data},
  author  = {Kwapong, Ephraim Kofi Ayi},
  journal = {arXiv preprint arXiv:2611.20087},
  year    = {2026}
}
```

## Licence

MIT. See `LICENSE`.
