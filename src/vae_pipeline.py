import os


from data_utils import (
    load_yaml_file,
    load_data_with_mask,
    scale_data_with_mask,
    inverse_transform_data,
    save_scaler,
    save_data,
)
import paths
from vae.vae_utils import (
    instantiate_vae_model,
    train_vae,
    save_vae_model,
    get_posterior_samples,
    load_vae_model,
)
from visualize import (
    plot_latent_space_samples,
    visualize_and_save_tsne_latent_pairs,
    plot_random_orig_recon_grid,
)


import numpy as np
import matplotlib.pyplot as plt

import numpy as np


def masked_mse_per_series(X, Xhat, mask):
    X = np.asarray(X, dtype=np.float32)
    Xhat = np.asarray(Xhat, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]  # (N,T)

    M3 = mask[..., None]  # (N,T,1)
    num = np.sum(((X - Xhat) ** 2) * M3, axis=(1, 2))  # (N,)
    den = (np.sum(M3, axis=(1, 2)) * X.shape[2]) + 1e-8  # (N,)
    return (num / den).astype(np.float32)  # (N,)


def make_alpha_per_series(alpha_base, rec_err_per_series, clip=(0.5, 2.0)):
    med = np.median(rec_err_per_series) + 1e-12
    scale = np.sqrt(rec_err_per_series / med).astype(np.float32)  # (N,)
    lo, hi = clip
    scale = np.clip(scale, lo, hi)
    return (alpha_base * scale).astype(np.float32)  # (N,)


def estimate_latent_gaussian(vae, X, mask=None, max_n=4096):
    Xs = X[:max_n]
    if mask is not None:
        Xs = Xs * mask[:max_n]
    z, _, _ = vae.encoder(Xs, training=False)
    z = z.numpy().astype(np.float32)  # (n,L)

    mu = z.mean(axis=0)  # (L,)
    Zc = z - mu[None, :]  # centered
    cov = (Zc.T @ Zc) / max(1, (Zc.shape[0] - 1))  # (L,L)

    # small ridge for numerical stability
    cov += 1e-6 * np.eye(cov.shape[0], dtype=np.float32)

    # Cholesky for fast sampling: cov = L L^T
    L = np.linalg.cholesky(cov).astype(np.float32)
    return mu.astype(np.float32), L  # (L,), (L,L)


def estimate_latent_scale(vae, X, max_n=2048):
    Xs = X[:max_n]
    z_mean, _, _ = vae.encoder(Xs, training=False)  # (B,L)
    z = z_mean.numpy().astype(np.float32)
    sigma = z.std(axis=0) + 1e-8  # (L,)
    return sigma


def masked_mse_np(a, b, mask):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]  # (B,T)
    M3 = mask[..., None]  # (B,T,1)
    num = np.sum(((a - b) ** 2) * M3)
    den = np.sum(M3) * a.shape[2] + 1e-8
    return float(num / den)


def get_ae_variants_anisotropic_adaptive(
    vae,
    X,  # (N,T,F) scaled
    mask,  # (N,T,1)
    n_variants,
    alpha_base,  # scalar
    z_chol,  # (L,L) from estimate_latent_gaussian
    seed=0,
    max_alpha_clip=(0.5, 2.0),
):
    X = np.asarray(X, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    X_in = X * mask

    # recon for per-series error
    X_rec = vae.predict(X_in, verbose=0).astype(np.float32)
    X_rec = X_rec * mask

    rec_err = masked_mse_per_series(X_in, X_rec, mask)  # (N,)
    alpha_i = make_alpha_per_series(alpha_base, rec_err, clip=max_alpha_clip)  # (N,)

    # encode originals
    z_mean, _, _ = vae.encoder(X_in, training=False)  # (N,L)
    z_mean = z_mean.numpy().astype(np.float32)
    N, Ldim = z_mean.shape

    rng = np.random.default_rng(seed)
    eps = rng.normal(size=(N, n_variants, Ldim)).astype(np.float32)  # (N,V,L)

    # anisotropic noise: eps @ chol^T has covariance ~ cov
    # shape: (N,V,L)
    noise = eps @ z_chol.T

    # adaptive scaling per series
    z = z_mean[:, None, :] + (alpha_i[:, None, None] * noise)  # (N,V,L)
    z_flat = z.reshape(N * n_variants, Ldim)

    X_var = vae.decoder.predict(z_flat, verbose=0).astype(np.float32)  # (N*V,T,F)
    T, F = X_var.shape[1], X_var.shape[2]
    X_var = X_var.reshape(N, n_variants, T, F)

    # zero padded region in outputs
    X_var = X_var * mask[:, None, :, :]

    return X_var, alpha_i, rec_err


def get_ae_variants(vae, x_ref, n_variants, sigma_latent, alpha=0.25, seed=0):
    if x_ref.ndim != 3:
        raise ValueError(f"x_ref must be (B,T,F). Got shape {x_ref.shape}")

    z_mean, _, _ = vae.encoder(x_ref, training=False)  # (B,L)
    z_mean = z_mean.numpy().astype(np.float32)

    rng = np.random.default_rng(seed)
    B, L = z_mean.shape
    eps = rng.normal(size=(B, n_variants, L)).astype(np.float32)

    z = z_mean[:, None, :] + (alpha * sigma_latent[None, None, :]) * eps  # (B,N,L)
    z = z.reshape(B * n_variants, L)

    x_var = vae.decoder.predict(z, verbose=0)  # (B*N,T,F)
    T, F = x_var.shape[1], x_var.shape[2]
    return x_var.reshape(B, n_variants, T, F)


def pick_alpha(vae, X, M, alphas, z_chol_small, ncheck=256, target_ratio=0.30, seed=0):
    Xs = X[:ncheck]
    Ms = M[:ncheck]

    x_rec = vae.predict(Xs, verbose=0)
    base = masked_mse_np(Xs, x_rec, Ms)

    best = None
    target = 1.0 + float(target_ratio)

    for a in alphas:
        V, _, _ = get_ae_variants_anisotropic_adaptive(
            vae,
            Xs,
            Ms,
            n_variants=1,
            alpha_base=a,
            z_chol=z_chol_small,
            seed=seed,
            max_alpha_clip=(
                1.0,
                1.0,
            ),  # disable adaptivity inside alpha search
        )
        V = V[:, 0]
        mse = masked_mse_np(Xs, V, Ms)
        ratio = mse / (base + 1e-12)
        score = abs(ratio - target)
        if best is None or score < best[0]:
            best = (score, a, mse, ratio, base)

    _, a_best, mse_best, ratio_best, base = best
    return a_best, base, mse_best, ratio_best


def sample_prior_mask_from_lengths(
    lengths: np.ndarray, T: int, n: int, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    L = rng.choice(lengths, size=n, replace=True)
    m = np.zeros((n, T, 1), dtype=np.float32)
    for i, li in enumerate(L):
        m[i, : int(li), 0] = 1.0
    return m


def run_vae_pipeline(dataset_name: str, vae_type: str):
    data, mask = load_data_with_mask(data_dir=paths.DATASETS_DIR, dataset=dataset_name)

    train_data = data
    train_mask = mask

    scaled_train_data, _, scaler = scale_data_with_mask(
        train_data, train_data, train_mask, train_mask
    )
    scaled_train_data = scaled_train_data * train_mask

    hyperparameters = load_yaml_file(paths.HYPERPARAMETERS_FILE_PATH)[vae_type]

    _, sequence_length, feature_dim = scaled_train_data.shape
    vae_model = instantiate_vae_model(
        vae_type=vae_type,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        **hyperparameters,
    )

    train_vae(
        vae=vae_model,
        train_data=scaled_train_data,
        max_epochs=200,
        verbose=1,
        train_mask=train_mask,
    )

    _, z_chol_small = estimate_latent_gaussian(
        vae_model, scaled_train_data, mask=train_mask, max_n=2048
    )

    alphas = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1]
    alpha_best, mse_rec_base, mse_var_base, ratio = pick_alpha(
        vae_model,
        scaled_train_data,
        train_mask,
        alphas=alphas,
        z_chol_small=z_chol_small,
        ncheck=256,
        target_ratio=0.20,
        seed=0,
    )
    print("Chosen alpha:", alpha_best)
    print("Recon masked MSE (base):", mse_rec_base)
    print("Variant masked MSE (base):", mse_var_base)
    print("Variant/Recons ratio (base):", ratio)

    _, z_chol = estimate_latent_gaussian(
        vae_model, scaled_train_data, mask=train_mask, max_n=4096
    )

    model_save_dir = os.path.join(paths.MODELS_DIR, dataset_name)
    save_scaler(scaler=scaler, dir_path=model_save_dir)
    save_vae_model(vae=vae_model, dir_path=model_save_dir)

    N_variants = 2  # number of variants of each series to generate

    x_variants, alpha_per_series, rec_err = get_ae_variants_anisotropic_adaptive(
        vae_model,
        scaled_train_data,
        train_mask,
        n_variants=N_variants,
        alpha_base=alpha_best,
        z_chol=z_chol,
        seed=42,
        max_alpha_clip=(0.8, 1.2),
    )

    print(
        "alpha_per_series min/median/max:",
        float(alpha_per_series.min()),
        float(np.median(alpha_per_series)),
        float(alpha_per_series.max()),
    )
    print(
        "rec_err per-series min/median/max:",
        float(rec_err.min()),
        float(np.median(rec_err)),
        float(rec_err.max()),
    )

    x_rec = vae_model.predict(scaled_train_data * train_mask, verbose=0).astype(
        np.float32
    )
    x_rec = x_rec * train_mask

    mse_rec = masked_mse_np(scaled_train_data, x_rec, train_mask)
    mse_v0 = masked_mse_np(scaled_train_data, x_variants[:, 0], train_mask)
    mse_v1 = masked_mse_np(scaled_train_data, x_variants[:, 1], train_mask)
    mse_v01 = masked_mse_np(x_variants[:, 0], x_variants[:, 1], train_mask)

    print("QA (scaled):")
    print("  recon mse:", mse_rec)
    print("  var0  mse:", mse_v0)
    print("  var1  mse:", mse_v1)
    print("  var0-var1 mse:", mse_v01)

    variants_flat = x_variants.reshape(-1, x_variants.shape[2], x_variants.shape[3])
    variants_flat_inv = inverse_transform_data(variants_flat, scaler)

    save_data(
        data=variants_flat_inv,
        output_file=os.path.join(
            os.path.join(paths.GEN_DATA_DIR, dataset_name),
            f"{vae_type}_{dataset_name}_ae_variants_alpha={alpha_best:.3f}_N={N_variants}.npz",
        ),
    )

    orig_inv = inverse_transform_data(scaled_train_data, scaler)
    rec_inv = inverse_transform_data(x_rec, scaler)

    plot_random_orig_recon_grid(
        orig_inv,
        rec_inv,
        train_mask,
        seed=42,
        title="Original scale: originals vs reconstructions (random 2×5)",
    )

    i = 10
    m = train_mask[i, :, 0].astype(bool)
    last = np.where(m)[0][-1] + 1

    plt.figure(figsize=(10, 3))
    plt.plot(scaled_train_data[i, :last, 0], label="orig")
    for k in range(x_variants.shape[1]):
        plt.plot(x_variants[i, k, :last, 0], alpha=0.8, label=f"var{k}")
    plt.title(f"Overlay original+variants (series {i})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    x_var1 = x_variants[:, 0]

    z_orig, _, _ = vae_model.encoder(scaled_train_data * train_mask, training=False)
    z_var1, _, _ = vae_model.encoder(x_var1 * train_mask, training=False)

    visualize_and_save_tsne_latent_pairs(
        z1=z_orig.numpy(),
        samples1_name="Original (z)",
        z2=z_var1.numpy(),
        samples2_name="Variant (z)",
        scenario_name=f"Model-{vae_type} Dataset-{dataset_name}",
        save_dir=os.path.join(paths.TSNE_DIR, dataset_name),
        max_samples=2000,
        draw_pair_lines=True,
    )

    if hyperparameters["latent_dim"] == 2:
        plot_latent_space_samples(vae=vae_model, n=8, figsize=(15, 15))

    loaded_model = load_vae_model(vae_type, model_save_dir)
    x_decoded = get_posterior_samples(vae_model, scaled_train_data)
    new_x_decoded = loaded_model.predict(scaled_train_data)
    print(
        "Preds from orig and loaded models equal:",
        np.allclose(x_decoded, new_x_decoded, atol=1e-5),
    )


if __name__ == "__main__":
    # check `/data/` for available datasets
    # in src/config/hyperparameters.yaml is where you should set a custom seasonality
    dataset = "m3_m"

    # timevae is the one used by varytimeae
    model_name = "timeVAE"

    run_vae_pipeline(dataset, model_name)
