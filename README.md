# VaryTimeAE: Variable-Length Time Series Variants with Masked Reconstruction + Latent Perturbations

This repository is based on the **TimeVAE** implementation by Abhishek Udesai:  
https://github.com/abudesai/timeVAE

It adapts the original codebase to support **variable-length** time series and to generate $N$ variants per input series (data augmentation), rather than unconditional generation from a learned prior.

For the original methodology and architectural motivation (including the level/trend/seasonality decoder components), see the TimeVAE paper:  
**TIMEVAE: A VARIATIONAL AUTO-ENCODER FOR MULTIVARIATE TIME SERIES GENERATION**  
https://arxiv.org/abs/2111.08095

---

## What stays the same (baseline TimeVAE components)

VaryTimeAE inherits the original design choices from TimeVAE:

- A sequence encoder/decoder VAE-style architecture implemented in `./src/vae/`.
- TimeVAE’s optional *interpretable decoder* decomposition (level, trend, seasonality, residual connection) when enabled via hyperparameters.
- Baseline model variants:
  - <s>**Dense VAE** (`vae_dense_model.py`)</s> **UNCHANGED**
  - <s>**Convolutional VAE** (`vae_conv_model.py`): “base model” in the paper</s> **UNCHANGED**
  - **TimeVAE** (`timevae.py`): with optional interpretable decoder components
- Training utilities and pipeline structure (instantiate/train/save/visualize).

---

## What changes in VaryTimeAE

### 1) Variable-length handling via padding + mask

Real datasets often contain series with different lengths. We batch them by padding to a common maximum length  $T$, and we store a binary mask  $m$:

-  $x \in \mathbb{R}^{T \times F}$: padded series
-  $m \in \{0,1\}^{T \times 1}$: mask (1 = observed timestep, 0 = padded)

All learning and evaluation is performed only on observed positions.

---

### 2) Mask-aware scaling

Scaling must not treat padding as data. We fit/transform using only observed values and then explicitly zero padded points in scaled space:

$
x^{scaled} \leftarrow \text{Scale}(x \mid m), \qquad
x^{scaled} \leftarrow x^{scaled} \odot m
$

This prevents padded tails from leaking into the model as a spurious pattern.

---

### 3) Training objective: masked reconstruction 

**Key change:** we train in **autoencoder mode**, optimizing only a masked reconstruction loss.

Forward pass:
1. Mask input:  $\tilde{x} = x \odot m$  
2. Encode:  $z = \mathrm{Enc}(\tilde{x})$  
3. Decode:  $\hat{x} = \mathrm{Dec}(z)$  
4. Mask output:  $\hat{x} \leftarrow \hat{x} \odot m$

Masked MSE loss:
$\mathcal{L}_{rec}=\frac{\sum_{t=1}^{T} m_t \lVert x_t - \hat{x}_t\rVert^2}{\sum_{t=1}^{T} m_t}$

**Difference vs TimeVAE:**  
TimeVAE trains with a VAE objective:
$\mathcal{L}_{VAE}=\mathcal{L}_{rec} + \beta \,\mathrm{KL}(q_\phi(z|x)\,\|\,p(z))$
which encourages  $q(z|x)$ to align with a global prior  $p(z)=\mathcal{N}(0,I)$, enabling meaningful **prior sampling**.

In **AE-mode**, the KL term is removed, because the goal here is:
- high-fidelity reconstruction for each series
- controlled, *input-conditioned* variant generation

---

## Reconstruction and Variant Generation

### Reconstruction

A reconstruction is:
$
\hat{x} = \mathrm{Dec}(\mathrm{Enc}(x \odot m)) \odot m
$

This is used for sanity checks and for measuring reconstruction quality (masked MSE).

---

### Variant generation

We generate  $N$ variants per original series by sampling locally around its latent code:

1) Encode each series:
$
z_i = \mathrm{Enc}(x_i \odot m_i)
$

2) Perturb latent code:
$
z_{i,k} = z_i + \alpha_i \,\varepsilon_{i,k}, \qquad k=1,\dots,N
$

3) Decode and mask output:
$
x^{var}_{i,k} = \mathrm{Dec}(z_{i,k}) \odot m_i
$

This yields variants that are close to each original series (fidelity) while not being identical (diversity).

---

## Two quality improvements used in VaryTimeAE 

### Adaptive noise scale per series

To keep variants similar across series, we scale  $\alpha$ using per-series reconstruction error  $e_i$:

$\alpha_i = \alpha \cdot \sqrt{\frac{e_i}{\mathrm{median}(e)}} \quad \text{(clipped)}$

This avoids over-perturbing smooth series and under-perturbing noisy series.

---

### Anisotropic latent noise

Instead of isotropic noise  $\varepsilon \sim \mathcal{N}(0,I)$, estimate latent covariance  $\Sigma$ from encoded latents and sample:

$
\varepsilon \sim \mathcal{N}(0,\Sigma)
$

using Cholesky factorization  $\Sigma = LL^\top$:
$
\varepsilon = \eta L^\top, \qquad \eta \sim \mathcal{N}(0,I)
$

This perturbs latents along realistic dataset directions, improving plausibility.

---

## Prior vs Posterior in AE-mode

Because **AE-mode removes KL**, the latent space is **not constrained** to match a known prior distribution (e.g.,  $\mathcal{N}(0,I)$). Therefore:

- **Prior sampling**  $z \sim \mathcal{N}(0,I)$ is not guaranteed to produce valid samples.
- Generation is **conditional**: variants are drawn around the encoded latent $z_i$ of each series.

---

## Practical outputs

Given $N$ input series and a chosen $N_{variants}:

- Reconstructions: $(\hat{x}_i)$ for each input $(x_i)$
- Variants: $(x^{var}_{i,k})$ for $k=1..N_{variants}$

Outputs are saved under:
- `./outputs/models/<dataset_name>/`
- `./outputs/gen_data/<dataset_name>/`
- `./outputs/tsne/<dataset_name>/`

---

## Project Structure

```plaintext
TimeVAE/
├── data/                         # Folder for datasets
├── outputs/                      # Folder for model outputs
│   ├── gen_data/                 # Folder for generated samples / variants
│   ├── models/                   # Folder for model artifacts
│   └── tsne/                     # Folder for t-SNE plots
├── src/                          # Source code
│   ├── config/                   # Configuration files
│   │   └── hyperparameters.yaml  # Hyperparameters settings
│   ├── vae/                      # VAE models implementation (VaryTimeAE)
│   │   ├── timevae.py            # adapted TimeVAE model 
│   │   ├── vae_base.py           # Base class; modified for masked AE-mode
│   │   ├── vae_conv_model.py     # Convolutional VAE model (paper base model)
│   │   ├── vae_dense_model.py    # Dense VAE model
│   │   └── vae_utils.py          # instantiate/train/save/load utilities
│   ├── data_utils.py             # load/scale logic (mask-aware scaling added)
│   ├── paths.py                  # path variables
│   ├── vae_pipeline.py           # main pipeline script (variants generation)
│   └── visualize.py              # visualization (incl. latent t-SNE for pairs)
├── LICENSE.md
├── README.md
└── requirements.txt
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage (Variants per Original)

1. **Prepare the Dataset**: Place your dataset in `./data/` as an `.npz` file containing:
    * `data`: Array shaped (N, T, F) (padded).
    * `mask`: Array shaped (N, T, 1) (1 for observed, 0 for padding).
    * `lengths` (optional): True lengths per series.

2. **Configure Pipeline**: Set the `dataset` and `model_name` in `./src/vae_pipeline.py`:
   ```python
   dataset = "m3_m"
   model_name = "timeVAE"
   ```

3. **Execute**: Run the following command in your terminal:
   ```bash
   python src/vae_pipeline.py
   ```

---

## Converting long CSV data to `.npz` (Notebook: `convert_long_npz.ipynb`)

This repository expects datasets in **`.npz` format** with at least:
- `data`: padded array of shape **(N, T, F)**
- `mask`: binary mask of shape **(N, T, 1)** indicating observed timesteps
- `lengths`: true lengths per series (**(N, )**)

To convert a **long-format CSV** with columns **(`unique_id`, `ds`, `y`)** into the required `.npz` format, use the notebook:

**`convert_long_npz.ipynb`**

### What it does
Given a long dataframe:
- Groups rows by `unique_id`
- Sorts by timestamp `ds`
- Pads all series to the maximum length `T`
- Creates a mask \(m_t\) with 1s for observed values and 0s for padded tail
- Saves a compressed `.npz` file containing: `data`, `mask`, `lengths`, and metadata (`unique_id`, `ds_list`)

### Where to place the converted datasets

The pipeline looks for datasets under /data. After conversion, move or copy the resulting files:

```bash
cp npz_data_converted/*.npz data/
```

---

## Pipeline Outputs

* **Models**: `./outputs/models/<dataset_name>/`
* **Variants**: `./outputs/gen_data/<dataset_name>/`
* **t-SNE Plots**: `./outputs/tsne/<dataset_name>/`

---

## Acknowledgements

* **Base Implementation**: [abudesai/timeVAE](https://github.com/abudesai/timeVAE)
* **Paper**: [TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation](https://arxiv.org/abs/2111.08095)