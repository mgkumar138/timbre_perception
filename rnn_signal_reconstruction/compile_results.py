# plot_metrics_across_seeds.py
import numpy as np
import matplotlib.pyplot as plt
import os

def load_multiple_arrays(prefix, seeds, outdir="results"):
    arrs = []
    for seed in seeds:
        fname = f"{outdir}/{prefix}_{seed}seed.npz"
        arr = np.load(fname)
        arrs.append(arr)
    arrs = np.stack(arrs) # shape [n_seeds, n_epochs]
    return arrs

def plot_mean_sem(arrs, label, color=None):
    mean = arrs.mean(axis=0)
    sem = arrs.std(axis=0)/np.sqrt(arrs.shape[0])
    epochs = np.arange(1, len(mean)+1)
    plt.plot(epochs, mean, label=label, color=color)
    plt.fill_between(epochs, mean-sem, mean+sem, alpha=0.2, color=color)

if __name__ == "__main__":
    seeds = [0, 1, 2]  # <== set to match your actual runs
    prefix = "run"     # <== must match fname_prefix from saving script
    outdir = "recon_sweep"
    # Load results
    train_accs = load_multiple_arrays(prefix, "train_acc", seeds, outdir)
    test_accs  = load_multiple_arrays(prefix, "test_acc", seeds, outdir)
    train_losses = load_multiple_arrays(prefix, "train_loss", seeds, outdir)
    test_losses  = load_multiple_arrays(prefix, "test_loss", seeds, outdir)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plot_mean_sem(train_accs, "Train acc", color='C0')
    plot_mean_sem(test_accs,  "Test acc", color='C1')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy across seeds (mean ± SEM)")

    plt.subplot(1,2,2)
    plot_mean_sem(train_losses, "Train loss", color='C2')
    plot_mean_sem(test_losses,  "Test loss", color='C3')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss across seeds (mean ± SEM)")
    plt.tight_layout()
    plt.show()