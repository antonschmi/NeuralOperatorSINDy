# SINDy Autoencoders → NeuralOperatorSINDy: a file-by-file comparison

**Sources compared**

- Champion, Lusch, Kutz & Brunton (2019), *"Data-driven discovery of coordinates and governing equations"*, arXiv:1904.02107, and its reference implementation, [`kpchamp/SindyAutoencoders`](https://github.com/kpchamp/SindyAutoencoders) (TensorFlow 1.x).
- This project, `NeuralOperatorSINDy` (JAX/Flax), specifically the Lorenz-63 experiment, which is the only one currently wired end-to-end.

Every `.py` and `.ipynb` file in both repositories was read in full (not just grepped) for this comparison.

---

## 1. What the paper actually specifies

The SINDy-autoencoder architecture (Fig. 1, Eq. 1–7, Eq. 12–17, §S1) is:

- **Encoder** `φ(x)`: fixed-width fully-connected MLP, sigmoid activations on every layer except the last, mapping `x ∈ R^n → z ∈ R^d`.
- **Decoder** `ψ(z)`: mirror-image MLP, `z ∈ R^d → x̂ ∈ R^n`, same activation convention.
- **SINDy library** `Θ(z)`: fixed polynomial (± sine) feature map, hand-built once for the whole run.
- **SINDy coefficients** `Ξ ∈ R^{p×d}`: a *trainable* dense matrix, initialized to a constant 1.0, gated at test/train time by a binary `mask` that is *not* a trainable variable — it's supplied externally and updated by hard sequential thresholding.
- **Loss** (Eq. 7/13): `L_recon + λ1·L_dx/dt + λ2·L_dz/dt + λ3·L_reg`, where `L_dz/dt` needs `∇_x φ(x)·ẋ` and `L_dx/dt` needs `∇_z ψ(z)·Θ(z)Ξ`. The paper computes both gradients **analytically**, layer by layer, hard-coding the derivative of each supported activation (`sigmoid`, `relu`, `elu`, linear) — see `z_derivative`/`z_derivative_order2` in §S1.3.
- **Training**: full-batch-per-epoch Adam, sequential (non-shuffled) minibatches, hard thresholding of `Ξ` every 500 epochs, then a dedicated **refinement phase** (fixed mask, L1 term dropped) to de-bias the surviving coefficients — explicitly analogized in the paper to a post-LASSO debiased regression.
- **Three worked examples**, each with its own hand-tuned hyperparameters (Table S1–S3): Lorenz-63 (`d=3`, order 1), reaction-diffusion (`d=2`, order 1, `include_sine=True`), nonlinear pendulum (`d=1`, order 2, second-derivative SINDy + second-order analytic backprop through the network).

---

## 2. Repository maps

### 2.1 Champion et al. / `SindyAutoencoders` (all files, 43 total incl. binary model checkpoints)

```
src/
  autoencoder.py        # full_network(), define_loss(), MLP builders, Θ(z), z_derivative[_order2]
  sindy_utils.py         # numpy: library_size, sindy_library, sindy_fit (STLSQ), sindy_simulate[_order2]
  training.py            # train_network(): TF1 session loop, feed_dict, print_progress
examples/
  lorenz/example_lorenz.py                  # get_lorenz_data / generate_lorenz_data / lorenz_coefficients
  lorenz/train_lorenz.ipynb                 # params dict + train_network() call
  lorenz/analyze_lorenz_model{1,2}.ipynb    # load a saved model, plot/diagnose
  pendulum/example_pendulum.py              # get_pendulum_data / pendulum_to_movie (2nd-order, image data)
  pendulum/train_pendulum.ipynb
  pendulum/analyze_pendulum_model{1,2}.ipynb
  rd/example_reactiondiffusion.py           # get_rd_data() -- loads a precomputed .mat (MATLAB PDE solve)
  rd/train_reactiondiffusion.ipynb
  rd/analyze_rd_model{1,2}.ipynb
rd_solver/*.m            # MATLAB reaction-diffusion PDE solver (not Python)
```

### 2.2 This repo / `NeuralOperatorSINDy`

```
src/
  autoencoder/__init__.py         # generic Autoencoder(encoder, decoder) flax Module -- UNUSED (see §5)
  encoder/__init__.py             # Encoder base class + _apply_grid_encoder_operator (FNO helper) -- helper unused
  encoder/pooling_encoder.py      # PoolingEncoder: DeepSets point-cloud encoder -- LIVE
  decoder/__init__.py             # Decoder base class + _apply_grid_decoder_operator (FNO helper) -- helper unused
  decoder/linear_decoder.py       # LinearDecoder: DeepONet-style, exactly linear in z -- LIVE (default)
  decoder/non_linear_decoder.py   # NonlinearDecoder: MLP(concat(z,x)) -- LIVE (config alternative)
  utils/networks/__init__.py      # MLP, CNN, attention blocks (CNN/attention unused)
  utils/networks/pooling.py       # DeepSetPooling (LIVE), MLPKernelPooling / MultiheadAttentionPooling (unused)
  positional_encodings/__init__.py# IdentityEncoding (LIVE, default), Fourier encodings (defined, unused)
  pysindy/__init__.py, pysindy/sindy.py   # both just `import pysindy as ps` -- empty wrapper, no custom code
  losses.py                       # sindy_ae_loss(): 5-term loss incl. a term with NO Champion counterpart
  training.py                     # theta_jax, SINDyAE (flax Module), jax.jvp-based train loop, SR3 option
experiments/
  lorenz63/config.py              # dataclass configs (model/training/loss) -- LIVE, fully wired
  lorenz63/training_data.py       # near-verbatim port of example_lorenz.py + dead torch sindy_simulate()
  lorenz63/analysis.ipynb         # extensive: loss curves, reconstruction, latent recovery, forecasting,
                                   # coordinate-transform check, mesh-resolution convergence, loss landscape
  lorenz63/MODEL_EXPLANATION.md   # this project's own from-scratch derivation of the model (dated 2026-07-03,
                                   # now stale relative to config.py in a few places, see §6)
  reaction_diffusion/config.py    # exists, but ...
  reaction_diffusion/training_data.py   # ... is actually the PENDULUM data generator, verbatim (see §3.7)
  reaction_diffusion/analysis.ipynb     # 0 bytes -- empty/never populated
```

No pendulum experiment folder exists at all in this repo.

---

## 3. Core architecture: file-by-file

### 3.1 `autoencoder.py` (Champion) ↔ `autoencoder/`, `encoder/`, `decoder/`, `training.py:SINDyAE` (this repo)

| | Champion `full_network()` | This repo's `SINDyAE` (+ `PoolingEncoder`/`LinearDecoder`) |
|---|---|---|
| Encoder input | `x ∈ R^{[batch,128]}` (flat vector, fixed order) | `(u, x) ` — `u ∈ R^{[B,n,1]}` field values **and** `x ∈ R^{[B,n,1]}` coordinates, as a point cloud |
| Encoder body | `build_network_layers`: stacked `Dense→sigmoid`, widths `[64,32]`, Xavier init, zero-bias init | `PoolingEncoder`: concat `(x,u)` → shared per-point MLP (`DeepSetPooling`, GELU, widths `[64,64]`) → **mean-pool over points** → `Dense(latent_dim)` |
| What makes it an "encoder" | Fixed matrix multiply chain, tied to `input_dim=128` | Permutation-invariant (DeepSets, Zaheer et al. 2017) reduction — well-defined for *any* number/order of points |
| Decoder input | `z ∈ R^{[batch,3]}` | `(z, x)` — same `z`, plus arbitrary query coordinates `x`, which need **not** be the encoder's input grid |
| Decoder body | Mirror MLP, widths `[32,64]`, sigmoid, ends with a plain linear (unconstrained) output layer | `LinearDecoder` (default): trunk MLP `x → φ_1..φ_{d}(x)` (widths `64,64,64`, GELU), then `û(y)=Σ_j z_j φ_j(y)` — **exactly linear in `z`**, no bias |
| Alternative decoder | none | `NonlinearDecoder`: MLP over `concat(tile(z), x)` — nonlinear in `z`, closer in spirit to Champion's decoder, selectable via `cfg.model.DECODER` but **not the default** |
| Weight init | Xavier/Glorot for all weights, zero for biases (hard-coded) | Flax defaults (`lecun_normal`, zero bias) — `MLP` supports custom `kernel_init`/`bias_init` but the model never passes them |
| Activation | sigmoid everywhere (chosen specifically because its derivative is trivial to hard-code) | GELU everywhere — free choice, since derivatives are no longer hand-coded (see §3.2) |
| Model class | one procedural function returning a dict of tensors | a `flax.linen.Module` (`SINDyAE`) that owns `xi` as an `nn.Module` parameter and composes injected `encoder`/`decoder` submodules |
| Unused generic scaffolding | — | `src/autoencoder/Autoencoder` (variational mean/logvar split, `batch_stats`, a `.sindy()` method) is defined but **never imported anywhere else in the repo** — the live model bypasses it entirely and is written directly in `training.py` |

**Theoretically load-bearing difference:** Champion's decoder is a genuine nonlinear map `R^d → R^n`; this repo's *default* decoder is an exactly linear map `R^d → R^n` (for fixed `x`, fixed decoder weights). This is not a cosmetic detail — it changes which reconstruction functions are representable at all, and (see §4, item 2) it is the direct cause of a symmetry problem the loss function had to grow a new term to fight.

### 3.2 Derivative computation: `z_derivative`/`z_derivative_order2` (Champion) ↔ `jax.jvp` (this repo)

Champion's `autoencoder.py` computes `dz/dt` and `dx̂/dt` by **manually propagating tangents through each layer**, with a different closed-form derivative rule hard-coded per activation (`elu`, `relu`, `sigmoid`, linear) — and a *second* hand-derived version (`z_derivative_order2`) for second derivatives, needed only for the pendulum's second-order SINDy model. This is exactly the paper's Eq. 14–16 (S1.3) written out in TensorFlow.

This repo's `training.py:SINDyAE.__call__` instead uses **forward-mode automatic differentiation** (`jax.jvp`) twice:

```python
z, dz_enc   = jax.jvp(lambda uu: self._encode(uu, x), (u_t,), (u_dot,))
...
u_hat, du_dec = jax.jvp(lambda zz: self.decoder(zz, x), (z,), (dz_sindy,))
```

This is mathematically the same operation (an exact directional derivative, not a finite-difference approximation) but is architecture-agnostic: no per-activation formula needs to be re-derived when the encoder/decoder are swapped. The trade-off is that **only first-order derivatives are implemented** — there is no analogue anywhere in this codebase of Champion's `z_derivative_order2`/`sindy_library_order2`/`ddz`/`ddx` machinery, so the pendulum's second-order SINDy formulation (§4.3 of the paper) is architecturally unsupported at present, not merely unconfigured.

### 3.3 SINDy library: `sindy_utils.py` (Champion) ↔ `theta_jax` + external `pysindy` (this repo)

Champion hand-rolls the polynomial (and optional sine) feature map twice — once in plain numpy (`sindy_utils.sindy_library`, used for post-hoc simulation/analysis) and once in raw TensorFlow ops (`autoencoder.sindy_library_tf`, used inside the trainable graph) — with matching nested-loop term orderings.

This repo's `src/pysindy/__init__.py` and `src/pysindy/sindy.py` are both literally just `import pysindy as ps` — i.e., this project adopted the actual `pysindy` PyPI package (a real, independent library authored by the SINDy group) for the *library metadata* (`ps.PolynomialLibrary(degree=..., include_bias=True)`, used to get `n_output_features_` and human-readable feature names) rather than reimplementing it. But `pysindy`'s library isn't JAX-differentiable, so `training.py:theta_jax` **re-implements the same polynomial map by hand a third time**, in JAX, using `itertools.combinations_with_replacement` — and then defensively cross-checks the two implementations agree on a shared sample (`sr3_refit`'s `AssertionError` guard) rather than assuming it. No sine terms are implemented in `theta_jax` at all (`include_sine` is a Champion-only concept here) — consistent with the fact that only the Lorenz experiment, which needs no sine terms, is actually wired up.

`experiments/lorenz63/training_data.py` also independently defines its own `library_size()` "to replace the imported `sindy_utils.library_size`" — but this function differs from Champion's in how it counts sine terms (`+= 2*n`, i.e. both sin *and* cos, vs. Champion's `+= n`, sin only) and, more importantly, **is never actually called from `training.py`** (`N_FEATURES` there comes from `ps.PolynomialLibrary` directly) — it's dead code, only used internally by that file's own `lorenz_coefficients()` ground-truth-Xi helper.

### 3.4 Sparse regression / thresholding: `sindy_fit` (Champion, unused in training) ↔ hard threshold + optional SR3 (this repo)

Champion's actual **training-time** sparsification is pure hard-thresholding of the gradient-descent-fit `Ξ` (`params['coefficient_mask'] = |Ξ| > threshold`, every 500 epochs, monotonically shrinking); `sindy_utils.sindy_fit` (an STLSQ solver) exists in the repo but is used only for classical (non-autoencoder) SINDy fits in analysis, not inside `train_network`.

This repo's `training.py` implements the same hard-threshold rule by default (`cfg.loss.SPARSITY_METHOD = 'relative_threshold'`) **and** adds a second, switchable mechanism: `sr3_refit`, which freezes the encoder, re-encodes a large sample via the same `jax.jvp` path, and re-fits `Ξ` from scratch using `pysindy.SR3` (an L0-regularized sparse-relaxation solver, Zheng et al. 2019 — the same SR3 method the original paper cites as the *theoretical* justification for hard-thresholding as an L0 proxy, but never itself uses in code). This SR3 path is a genuine methodological addition with no counterpart in Champion's code, including bookkeeping Champion's training loop has no equivalent of: resetting the Adam moment estimates for just the `xi` leaf after an external overwrite (`_reset_xi_adam_moments`).

### 3.5 Loss function: `define_loss` (Champion) ↔ `sindy_ae_loss` (this repo)

| Term | Champion | This repo | Notes |
|---|---|---|---|
| Reconstruction | `mean((x - x̂)²)` | `mean((û - u)²)` | same |
| Latent consistency (`dz/dt`) | `mean((dz - dz_predict)²)` | `mean((dz_enc - dz_sindy)²)` | same intent; `dz` comes from the hand-derived chain rule vs. `jax.jvp` |
| Decoder/field consistency (`dx/dt`) | `mean((dx - dx_decode)²)` | `mean((du_dec - u_dot)²)` | same intent |
| Sparsity | `mean(abs(mask·Ξ))` | `mean(abs(mask·xi))` | same (both `mean`, not `sum`, despite an earlier draft of `MODEL_EXPLANATION.md` describing it as `sum`-based — that description is now stale, see §6) |
| **Gauge-fixing / variance term** | **none** | `mean((Cov(z) - I)²)` — full batch covariance vs. identity | **No equivalent exists anywhere in Champion et al.'s model or paper.** This is a genuinely new loss term (see §4, item 2, for why it was needed). |

### 3.6 Training loop: `training.py` (Champion) ↔ `training.py` (this repo)

| | Champion | This repo |
|---|---|---|
| Framework | TF1: `tf.Session`, `feed_dict`, `tf.placeholder` for `x`,`dx`,`mask`,`learning_rate` | JAX/Flax: `jax.jit`-compiled `train_step`, `flax.training.train_state.TrainState`, `optax` |
| Batching | **Sequential, non-shuffled** minibatches per epoch: `batch_idxs = arange(j*bs, (j+1)*bs)` | **i.i.d. random** batch without replacement each step (`rng.choice(...)`) — a bootstrap scheme, no epoch structure at all |
| Loop structure | `for epoch in max_epochs: for batch in epoch: ...` (explicit epoch/epoch-size bookkeeping) | flat `for step in 1..MAX_STEPS` (no epochs) — comments in `config.py` translate epoch counts to step counts (`5001 epochs * 250 steps/epoch`) for continuity with Champion's numbers, but the number is now stale for `lorenz63/config.py` after `BATCH_SIZE`/`N_ICS` were changed (see §6) |
| Learning rate | **constant** throughout a run (`params['learning_rate']`, fed every step, never decayed) | `optax.exponential_decay(lr, transition_steps, decay_rate)` — an LR **schedule that Champion's training code does not have at all** |
| Thresholding cadence | every `threshold_frequency` **epochs** | every `THRESH_EVERY` **steps** (in step-space, matched to be numerically equivalent to Champion's epoch cadence for the default relative-threshold path) |
| Refinement phase | separate `AdamOptimizer` instance/graph node (`train_op_refinement`), same `sess`, mask frozen, L1 term dropped from `loss_refinement` | same *optax* optimizer *state* is carried over (not reset) into a phase built with `LAMBDA_SP=0.0` via `dataclasses.replace`; mask frozen (`do_threshold=False`) — same intent, but Champion's separate-optimizer-object refinement effectively starts with fresh (zeroed) Adam moments, while this repo's continues accumulated momentum from the main phase |
| Checkpointing | `tf.train.Saver` + a separate `pickle.dump` of `params` dict, once, at the very end | `save_checkpoint`/`load_checkpoint`: numpy-ified `{params, opt_state, step, mask, loss_hist}` pickled periodically (`checkpoint_every`) *and* at the end, with resume logic that reconstructs `state.step` and skips already-completed steps/thresholding events — built for interruption-prone (e.g. Colab/remote GPU) training that Champion's original single long-running local session didn't need to worry about |
| Point-cloud subsampling | not applicable (fixed-vector model) | `sample_batch_subsampled` / `cfg.model.SUBSAMPLE_POINTS` / `N_SUB`: each *row* of a batch can get its own independently-random subset of spatial points — a capability with literally no analogue in Champion's formalism, and the most direct test of the "neural operator" (mesh-invariance) claim |

### 3.7 Example / data-generation files

**Lorenz.** `experiments/lorenz63/training_data.py`'s `generate_lorenz_data`/`simulate_lorenz`/`lorenz_coefficients` are a near line-for-line port of Champion's `examples/lorenz/example_lorenz.py` (same Legendre-mode lifting, same cubic-term option, same variable names). Two things were added on top:
- a `sindy_simulate(z0, t, Xi, sindy_layer)` function that expects a **torch** `nn.Module`-like `sindy_layer` object with `.power_matrix`/`.add_sine` attributes — nothing in this repository defines such an object (the live pipeline uses `theta_jax`, a plain JAX function, not a class), and this function is **never called anywhere in the repo**. Combined with `torch==2.12.1` sitting in `requirements.txt` unused by any live import, this looks like leftover code from a different (earlier, torch-based) prototype that was copy-pasted in and never adapted or removed.
- `get_lorenz_data(..., linear=True)` takes a `linear` flag piped through from `cfg.model.LINEAR_OBS`, controlling whether the cubic Legendre modes are added — Champion's own `get_lorenz_data` hard-codes `linear=False` (always includes the cubic modes); the *default* config here instead sets `LINEAR_OBS=True`, i.e. it trains on a **simpler, purely-linear lifting of the Lorenz state** by default, not on Champion's original nonlinear high-dimensional observation.

**"Reaction-diffusion."** `experiments/reaction_diffusion/training_data.py` contains `get_pendulum_data`, `generate_pendulum_data`, and `pendulum_to_movie` — this is a verbatim copy of Champion's `examples/pendulum/example_pendulum.py`, not any reaction-diffusion code. There is no data-loading counterpart anywhere to Champion's `example_reactiondiffusion.py` (`get_rd_data`, which loads a MATLAB-solved spiral-wave `.mat` file). `experiments/reaction_diffusion/config.py` sets fields (`LATENT_DIM=3`, `LINEAR_OBS`, Legendre-lifting-flavored comments) that match the Lorenz experiment's conventions, not the true reaction-diffusion system's (`d=2`, no Legendre lifting, spatial 2D grid, needs `include_sine`). `experiments/reaction_diffusion/analysis.ipynb` is a **0-byte file** (created, never populated). Net effect: **the reaction-diffusion experiment does not exist yet** — the folder is a stub that inherited an unrelated file by mistake or as an unfinished placeholder, and no pendulum experiment folder exists in this repo at all. Of Champion's three worked examples, only Lorenz-63 is actually implemented end-to-end here.

### 3.8 Notebooks

Champion's `train_*.ipynb` are thin orchestration (build a `params` dict, call `train_network`, pickle results over `num_experiments` random restarts); `analyze_*.ipynb` load a saved model and produce the paper's figures (attractor plots, coefficient-matrix heatmaps, an explicit affine coordinate-transform recovering the textbook Lorenz form, in/out-of-distribution error tables).

`experiments/lorenz63/analysis.ipynb` (36 cells) reproduces the analogous checks — loss curves, reconstruction quality, latent-vs-true-state correlation, an affine coordinate-recovery fit explicitly modeled on "Champion et al.'s own analysis notebook... Fig. 4" (this project's own comment), forecasting via `odeint` on the learned SINDy RHS, and a 3-panel attractor comparison styled after `analyze_lorenz_model1.ipynb` — **and then goes further** with checks that have no counterpart in Champion's repository at all:
- an inference-time **sparse-in/dense-out** test (encode from a random `N_SUB`-point subset, decode at the full 128-point grid),
- a **2×-resolution query test** (decode at 256 points when only ever trained querying 128),
- a **discretization-consistency sweep**: encode the same underlying trajectory relifted at resolutions `{32,64,128,256,512}` against a `1024`-point reference and plot `‖z(N) − z(N_ref)‖/‖z(N_ref)‖` vs. `N` — a genuine, quantitative test of the mesh-invariance claim that motivates the whole neural-operator extension, with no equivalent question even expressible in Champion's fixed-`n=128`-vector formalism,
- a **1D loss-landscape interpolation** (Goodfellow-et-al.-style) between random init and the trained parameters.

These are legitimate novel contributions of this extension, not present in the original paper or code in any form.

---

## 4. Summary: strong theoretical deviations

Ranked roughly by how much they change what the model *can represent or claim*, not just how it's implemented:

1. **Point-cloud / operator formulation replacing fixed-vector autoencoding.** Champion's `φ`, `ψ` are functions `R^n → R^d → R^n` tied to one specific `n=128` ordering. This repo's encoder/decoder are functions of `(coordinate, value)` pairs, invariant to point order and (in principle) to point count — this is the actual "neural operator" extension, and it is real, not just a relabeling: the mesh-subsampling training mode and the resolution-convergence analysis exist specifically to exercise this property.
2. **The default decoder is exactly linear in `z`**, vs. Champion's genuinely nonlinear sigmoid MLP decoder. This is a strictly weaker function class for the decoder alone, and it changes the model's symmetries: an exactly-linear, no-bias decoder makes the latent rescaling `z → cz`, `Ξ → Ξ/c` a much easier degeneracy for gradient descent to exploit (via the L1 sparsity pressure shrinking `z`'s scale rather than genuinely zeroing terms) than it would be for a saturating nonlinear MLP. This is very likely *why* the loss needed a term Champion's never did (`LAMBDA_VAR`, the batch-covariance gauge-fix) — a new inductive bias introduced to compensate for an architectural choice made elsewhere in the same change. A `NonlinearDecoder` alternative exists and is closer in spirit to Champion's, but is not the default.
3. **Automatic (forward-mode) vs. hand-derived-analytic differentiation** for the `dz/dt` and `dx̂/dt` consistency terms. Mathematically equivalent to first order, and the entire point of the switch (it's what makes plugging in a different encoder/decoder possible without re-deriving formulas) — but it is a real capability loss in one specific respect: Champion's second-order machinery (`z_derivative_order2`, needed for the pendulum) has no counterpart, so second-order SINDy models are currently out of scope architecturally, not just unconfigured.
4. **A new gauge-fixing/whitening loss term (`LAMBDA_VAR`) with no equivalent anywhere in Champion et al.** — not a re-derivation of an existing term, a new regularizer addressing a symmetry problem that is more severe in this architecture than in the original (see #2).
5. **Sequential (epoch, non-shuffled) batching → i.i.d. bootstrap batching**, and **constant LR → exponential-decay LR schedule**. Both are genuine changes to the optimization dynamics/statistics of training, independent of the architecture swap itself.
6. **An additional, qualitatively different sparsity mechanism (SR3 sparse regression)** alongside (not replacing) the original hard-thresholding rule — a real methodological addition, closer to the L0-proxy justification the original paper cites but never itself implements in code.
7. **Scope regression on the worked examples**: Champion's method is demonstrated on three systems (Lorenz, reaction-diffusion, pendulum), each exercising a different aspect of the framework (first-order/no-sine, first-order/with-sine, second-order/with-sine). This repo currently implements only the first-order, no-sine Lorenz case end-to-end; the reaction-diffusion folder contains an unrelated (pendulum) data generator and an empty analysis notebook, and there is no pendulum folder at all.

## 5. Notable non-theoretical / hygiene findings (asked for completeness)

- `torch` is a listed dependency and is imported at the top of `experiments/lorenz63/training_data.py`, but the only thing that uses it (`sindy_simulate`'s `sindy_layer.power_matrix`) is dead code — no such object exists in the JAX pipeline. Likely a carry-over from an earlier prototype.
- `src/autoencoder/Autoencoder` (a more generic, variational-ready, `batch_stats`-aware autoencoder wrapper) is fully implemented but never imported outside its own file — the actual trained model (`SINDyAE` in `training.py`) is a separate, purpose-built class that doesn't use it.
- Several alternative building blocks are implemented but never selected by any current config: `NonlinearDecoder` is reachable via `cfg.model.DECODER` but not the default; `MLPKernelPooling`/`MultiheadAttentionPooling` (vs. the live `DeepSetPooling`); `FourierEncoding1D`/`RandomFourierEncoding` (vs. the live `IdentityEncoding`); the FNO-oriented grid-reshaping helpers `_apply_grid_encoder_operator`/`_apply_grid_decoder_operator` reference an `FNODecoder` that doesn't exist anywhere in the repo.
- `experiments/lorenz63/training_data.py` defines its own `library_size()` that disagrees with Champion's sine-counting convention and is, in any case, never called by `training.py` (which sizes the library from `pysindy.PolynomialLibrary` directly) — dead code with an inconsistent docstring claim ("replace the imported `sindy_utils.library_size`").

## 6. A note on `MODEL_EXPLANATION.md`

This file (in `experiments/lorenz63/`) is this project's own detailed derivation of the model, dated 2026-07-03, and it is excellent — largely consistent with what direct reading of the current code confirms. But `config.py` has moved on since it was written, and a few of its stated facts are now stale:
- It documents `BATCH_SIZE=128`, `LAMBDA_DZ=0.0`, `LAMBDA_VAR=0.0`, and a per-axis-only variance loss; the current `config.py`/`losses.py` have `BATCH_SIZE=8000`, `LAMBDA_DZ=1.0`, `LAMBDA_VAR=1.0`, and a full-covariance (variance **and** decorrelation) gauge-fix.
- It has no mention of the SR3 sparsity path or `SUBSAMPLE_POINTS`/mesh-subsampling, both of which exist in the current `training.py` and are exercised in `analysis.ipynb`.

Worth a refresh if it's meant to stay the canonical reference document, since it now understates how far the loss function and training loop have diverged from the state it describes.
