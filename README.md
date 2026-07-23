# NO-SINDy: Mesh-invariant sparse identification of non-linear dynamics

A JAX/Flax reimplementation and extension of Champion, Lusch, Kutz & Brunton (2019),
*"Data-driven discovery of coordinates and governing equations"* (SINDy autoencoders).
The core idea is unchanged: encode a high-dimensional field into a low-dimensional
latent state `z`, discover a sparse polynomial ODE `dz/dt = Θ(z)·Ξ` governing it, and
decode back — training all three pieces jointly so the discovered coordinates and the
discovered dynamics support each other.

The extension in this project: the encoder and decoder are **point-cloud / neural-operator
style** rather than fixed-vector MLPs. Instead of mapping a fixed-length, fixed-order
vector `x ∈ R^128`, the encoder consumes a set of `(coordinate, value)` pairs and is
permutation-invariant and (in principle) resolution-invariant — the same trained model
can be queried on a different number, order, or subset of spatial points than it was
trained on. This is the actual thing under test in the mesh-subsampling training mode
and the resolution-convergence checks in the analysis notebooks; see
`COMPARISON_TO_CHAMPION_2019.md` for a detailed file-by-file diff against the original
paper's implementation.

## Architecture

**Encoder** (`src/encoder/pooling_encoder.py`, `src/utils/networks/pooling.py`): a
`PoolingEncoder` that concatenates each query point's coordinate with its field value,
runs a shared per-point MLP (`DeepSetPooling`, widths set by `cfg.model.ENCODER_FEATURES`),
mean-pools over points, then projects to `z ∈ R^{latent_dim}`. Permutation-invariant and
point-count-invariant by construction (DeepSets, Zaheer et al. 2017). An attention-based
pooling alternative (`MultiheadAttentionPooling`) exists in the same file but isn't wired
up as a config option yet.

**Decoder** (`src/decoder/`), selected via `cfg.model.DECODER`:
- `"linear"` — `LinearDecoder`: a trunk MLP over query coordinates only, combined with `z`
  via an exactly-linear (no bias) contraction — a DeepONet with no branch net, `z` itself
  serving as the branch coefficients.
- `"nonlinear"` — `NonlinearDecoder`: one MLP over `concat(z, x)`, nonlinear in `z` but `z`
  and `x` share a single network.
- `"deeponet"` — `DeepONetDecoder`: a genuine branch(`z`) + trunk(`x`) DeepONet, nonlinear
  in `z`, current default for both experiments.

**SINDy layer** (`src/training.py`): `theta_jax` builds the polynomial(+sine) feature
library `Θ(z)` in JAX (matching `pysindy.PolynomialLibrary`'s term ordering, cross-checked
against it at runtime); `xi` is a trainable `(n_features, latent_dim)` matrix, gated by a
`mask` that starts all-ones and only shrinks. `SINDyAE.__call__` uses **forward-mode
automatic differentiation** (`jax.jvp`), not hand-derived per-layer backprop, to compute
the encoder's own latent velocity `dz_enc` and the decoder's field velocity `du_dec` —
architecture-agnostic, at the cost of only supporting first-order SINDy models currently.

**Loss** (`src/losses.py`), five terms, each skippable at trace time via a zero weight:
reconstruction (`loss_rec`), latent consistency `dz_enc` vs. `dz_sindy` (`loss_dz`),
field-derivative consistency `du_dec` vs. true `u_dot` (`loss_dx`), L1 sparsity on
`mask·xi` (`loss_sp`), and a gauge-fix pinning `Cov(z) ≈ I` across the batch (`loss_var`)
— this last term has no equivalent in Champion et al.'s original formulation; it exists
here to break the latent rescaling symmetry that the (default, exactly-linear) decoder
makes easy for gradient descent to exploit otherwise.

**Sparsification** (`run_phase` in `src/training.py`), selected via `cfg.loss.SPARSITY_METHOD`:
- `"relative_threshold"` — hard sequential thresholding of the co-trained `xi` (`|xi| <
  THRESHOLD → 0`), matching Champion et al.'s actual training-time mechanism.
- `"sr3"` — freezes the encoder, re-encodes a fresh sample, and refits `xi` from scratch
  via `pysindy.SR3` (L0-regularized sparse regression). Two safety guards wrap this path,
  both added after observing real failure modes during tuning: a refit that comes back
  with zero active terms is rejected (the mask update is monotone, so a bad refit would
  otherwise permanently kill the SINDy layer), and a refit whose optimizer didn't
  converge (pysindy's `ConvergenceWarning`) is rejected too (non-converged coefficients
  were observed to compound into runaway magnitude across successive refits).

**Training loop** (`train`/`run_phase`/`make_train_step` in `src/training.py`): the full
training dataset is moved to device once; each step's batch/point sampling (row
selection, and per-row point subsampling when `cfg.model.SUBSAMPLE_POINTS=True`) happens
**inside** the jitted `train_step` itself, via `jax.random.choice`/`vmap`, rather than as
host-side numpy code between steps. Loss-history bookkeeping is buffered and synced to
host in batches rather than every step. Both choices exist specifically to keep the host
from serializing against the GPU's async dispatch — the difference is large in practice.
Checkpointing (`save_checkpoint`/`load_checkpoint`) is periodic and resume-aware, since
training runs are typically launched on a remote machine and can be interrupted.

## Experiments

**`experiments/lorenz63/`** — the Lorenz-63 attractor, analytically lifted into a
128-point 1D field via Legendre modes (`training_data.py`, ported from Champion et al.'s
own example). Ground-truth latent state is known, so `analysis.ipynb` can directly check
encoder/true-state correlation, recover an affine coordinate transform into the textbook
Lorenz frame, and run a resolution-convergence sweep. `LATENT_DIM=3`, no sine terms
needed.

**`experiments/reaction_diffusion/`** — a lambda-omega reaction-diffusion PDE producing a
rotating spiral wave, solved via `rd_solver/reaction_diffusion.m` (MATLAB; run once,
produces `reaction_diffusion.mat`) on a `100×100` (10,000-point) grid. No ground-truth
low-dimensional state ships with this system, but one is derivable analytically: the
lambda-omega reaction kinetics reduce to a Stuart-Landau/Hopf-normal-form oscillator, so
a correctly-fit sparse model should keep only linear self-terms and the four cubic terms
per latent dimension (no bias, no quadratic terms, no sine terms), `LATENT_DIM=2`.
Training/tuning for this experiment is still in progress — see current config comments in
`experiments/reaction_diffusion/config.py` for the reasoning behind current loss weights
and known failure modes already fixed (SR3 miscalibration, sparsity/consistency loss
imbalance).

## Repo structure

```
src/
  encoder/           PoolingEncoder (point-cloud encoder) + base class
  decoder/           LinearDecoder, NonlinearDecoder, DeepONetDecoder + base class
  utils/networks/     MLP, pooling variants (DeepSetPooling live; attention/kernel unused)
  positional_encodings/  IdentityEncoding (live), Fourier encodings (defined, unused)
  losses.py          sindy_ae_loss: the 5-term loss described above
  training.py        theta_jax, SINDyAE model, on-device train_step, run_phase/train,
                      sr3_refit, checkpointing
experiments/
  lorenz63/           config.py, training_data.py, analysis.ipynb
  reaction_diffusion/ config.py, training_data.py, train.py, analysis.ipynb, rd_solver/
COMPARISON_TO_CHAMPION_2019.md   detailed diff against the original paper's TF1 implementation
```

## Setup

```
pip install -r requirements.txt
```

Reaction-diffusion additionally requires running `rd_solver/reaction_diffusion.m` once in
MATLAB before training (produces `experiments/reaction_diffusion/reaction_diffusion.mat`).

## Usage

Each experiment has its own entry point, run from the repo root:

```
python -m src.training                          # Lorenz-63 (src/training.py's __main__)
python experiments/reaction_diffusion/train.py   # reaction-diffusion (fixes its own sys.path
                                                  # to import src.*, so it's runnable as a plain
                                                  # script rather than needing -m)
```

Both build their model/config from the corresponding `experiments/*/config.py`, resume
from a checkpoint automatically if one exists at the configured path, and checkpoint
periodically during training. `experiments/*/analysis.ipynb` load a finished (or
in-progress) checkpoint for diagnostics — loss curves, reconstruction quality, discovered
equations, and (Lorenz only) latent/true-state correlation and coordinate recovery.

## Key config knobs

| Field | Where | Meaning |
|---|---|---|
| `SUBSAMPLE_POINTS` / `N_SUB` | `model` | mesh-invariance training mode: each batch row gets its own random `N_SUB`-point subset of the grid instead of the fixed full grid |
| `ENCODER_FEATURES` | `model` | per-point MLP layer widths in the encoder (list — length sets depth, values set width) |
| `DECODER` | `model` | `"linear"` / `"nonlinear"` / `"deeponet"` |
| `LAMBDA_REC/DZ/DX/SP/VAR` | `loss` | the five loss term weights described above |
| `SPARSITY_METHOD` | `loss` | `"relative_threshold"` or `"sr3"` |
| `THRESH_START` / `THRESH_EVERY` | `loss` | when sparsification first fires, and how often after that |
| `SR3_LAM` / `SR3_NU` / `SR3_N_SAMPLES` | `loss` | SR3 optimizer regularization weight, relaxation coefficient, and refit sample size |

## Further reading

- `COMPARISON_TO_CHAMPION_2019.md` — a detailed, file-by-file comparison against Champion
  et al.'s original implementation, covering every architectural and methodological
  deviation. Worth reading before assuming this codebase does something "the paper's way."
- `experiments/lorenz63/MODEL_EXPLANATION.md` — a from-scratch derivation of the model;
  useful for the math, though check it against current `config.py`/`losses.py` values
  before trusting specific numbers, since it predates several since-changed defaults.
