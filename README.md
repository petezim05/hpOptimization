# MLP Hyperparameter Optimization

Exploratory study on how MLP architecture (depth and width) affects training performance on a sin(x) approximation task.

## Overview

A multilayer perceptron (MLP) is trained to approximate sin(x) over [-2π, 2π]. Different architectures are evaluated by comparing their training loss curves. Results are logged to a CSV so many runs can be compared on a single graph.

## Files

| File | Purpose |
|---|---|
| `net.py` | Builds an MLP of arbitrary depth and width from a list of layer sizes |
| `trainer.py` | Training loop — runs Adam on random batches, returns per-epoch MSE losses |
| `main.py` | Configures one run, calls the trainer, appends results to `losses.csv` |
| `grapher.py` | Reads `losses.csv` and plots all runs as a multi-line log-scale graph |
| `agent.py` | Reinforcement learning agent (in development) |

## Usage

**1. Run a training configuration**

Edit the `params` list in `main.py` to set the hidden layer widths, then run:

```
python main.py
```

For example, `params = [32, 32]` trains a network with architecture `1 -> 32 -> 32 -> 1`. Results are appended as a new row in `losses.csv`. Run `main.py` with different `params` values to accumulate multiple runs.

**2. Graph all runs**

```
python grapher.py
```

Plots every run in `losses.csv` as a separate line. The Y axis is log-scale MSE loss. Each point represents the average loss over a 250-epoch window. A red benchmark line is drawn at MSE = 0.001.

## Hyperparameters

Configured at the top of `trainer.py`:

| Parameter | Default | Description |
|---|---|---|
| `LR` | 1e-3 | Adam learning rate |
| `EPOCHS` | 1000 | Training epochs per run |
| `BATCH_SIZE` | 64 | Random samples per batch |

## Output

`losses.csv` — one row per run, columns:

```
architecture, lr, epochs, batch_size, epoch_1, epoch_2, ..., epoch_N
```

`results.txt` — human-readable summary of the most recent run.

## Dependencies

- PyTorch
- NumPy
- Matplotlib
