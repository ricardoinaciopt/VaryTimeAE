import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.manifold import TSNE

TITLE_FONT_SIZE = 16


def avg_over_dim(data: np.ndarray, axis: int) -> np.ndarray:
    return np.mean(data, axis=axis)


def plot_random_orig_recon_grid(
    X,
    Xrec,
    mask,
    *,
    nrows=2,
    ncols=5,
    feature=0,
    seed=0,
    sharey=False,
    title="Random original vs reconstruction",
):
    X = np.asarray(X)
    Xrec = np.asarray(Xrec)
    mask = np.asarray(mask)

    N = X.shape[0]
    rng = np.random.default_rng(seed)
    idxs = rng.choice(N, size=nrows * ncols, replace=False)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), sharey=sharey
    )
    axes = np.asarray(axes).reshape(-1)

    for ax, i in zip(axes, idxs):
        m = mask[i, :, 0].astype(bool)
        if not np.any(m):
            ax.set_title(f"idx={i} (empty mask)")
            ax.axis("off")
            continue
        last = np.where(m)[0][-1] + 1

        ax.plot(X[i, :last, feature], label="orig")
        ax.plot(Xrec[i, :last, feature], label="recon", alpha=0.85)
        ax.set_title(f"idx={i}, len={last}")
        ax.grid(True, alpha=0.15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def to_masked_fixed_length_vectors(x: np.ndarray, m: np.ndarray, K: int) -> np.ndarray:
    if m.ndim == 3:
        m = m[..., 0]
    m = m.astype(bool)

    N, T, F = x.shape
    out = np.zeros((N, K), dtype=np.float32)

    for i in range(N):
        valid = x[i, m[i], 0]  # univariate
        if valid.size == 0:
            continue
        L = min(valid.size, K)
        out[i, :L] = valid[:L]
    return out


def visualize_and_save_tsne_latent_pairs(
    z1: np.ndarray,
    samples1_name: str,
    z2: np.ndarray,
    samples2_name: str,
    scenario_name: str,
    save_dir: str,
    max_samples: int = 1000,
    perplexity_max: int = 40,
    n_iter: int = 600,
    seed: int = 42,
    draw_pair_lines: bool = True,
) -> None:
    """
    t-SNE on latent embeddings (N,D) for two aligned sets (same N).
    Draws optional lines connecting each pair i: (z1_i -> z2_i).
    """
    if z1.shape[0] != z2.shape[0]:
        raise ValueError("z1 and z2 must have the same number of samples.")
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError("z1 and z2 must be 2D arrays: (N,D).")
    if z1.shape[1] != z2.shape[1]:
        raise ValueError("z1 and z2 must have the same embedding dimension.")

    used = min(z1.shape[0], max_samples)
    Z = np.vstack([z1[:used], z2[:used]])  # (2*used, D)

    n_points = Z.shape[0]
    perplexity = min(perplexity_max, max(5, (n_points - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    Y = tsne.fit_transform(Z)

    y1 = Y[:used]
    y2 = Y[used:]

    plt.figure(figsize=(8, 8))
    plt.scatter(y1[:, 0], y1[:, 1], label=samples1_name, alpha=0.55, s=40)
    plt.scatter(y2[:, 0], y2[:, 1], label=samples2_name, alpha=0.55, s=40)

    if draw_pair_lines:
        for i in range(used):
            plt.plot(
                [y1[i, 0], y2[i, 0]], [y1[i, 1], y2[i, 1]], alpha=0.15, linewidth=0.7
            )

    plt.title(f"t-SNE (latent) for {scenario_name}")
    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(save_dir, f"{scenario_name}_latent_tsne.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


def plot_latent_space_samples(vae, n: int, figsize: tuple) -> None:
    """
    Plot samples from a 2D latent space.

    Args:
        vae: The VAE model with a method to generate samples from latent space.
        n (int): Number of points in each dimension of the grid.
        figsize (tuple): Figure size for the plot.
    """
    scale = 3.0
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    grid_size = len(grid_x)

    # Generate the latent space grid
    Z2 = np.array([[x, y] for x in grid_x for y in grid_y])

    # Generate samples from the VAE given the latent space coordinates
    X_recon = vae.get_prior_samples_given_Z(Z2)
    X_recon = np.squeeze(X_recon)

    fig, axs = plt.subplots(grid_size, grid_size, figsize=figsize)

    # Plot each generated sample
    for k, (i, yi) in enumerate(enumerate(grid_y)):
        for j, xi in enumerate(grid_x):
            axs[i, j].plot(X_recon[k])
            axs[i, j].set_title(f"z1={np.round(xi, 2)}; z2={np.round(yi, 2)}")
            k += 1

    fig.suptitle("Generated Samples From 2D Embedded Space", fontsize=TITLE_FONT_SIZE)
    fig.tight_layout()
    plt.show()
