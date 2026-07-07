# NO-SINDy Model: Complete Mathematical Reference

This document is a from-scratch, line-by-line account of what `src/training.py`,
`src/losses.py`, `src/encoder/pooling_encoder.py`, `src/decoder/linear_decoder.py`,
and `src/utils/networks/pooling.py` currently compute, with exact tensor shapes at
every step, the full forward math, the loss, and how gradients propagate back to
every trainable parameter. It reflects the code and `experiments/lorenz63/config.py`
as of 2026-07-03. Config values used throughout:

```
POLY_ORDER = 3     LATENT_DIM = 3      INPUT_DIM = 128
BATCH_SIZE = 128    N_ICS = 1024        N_VAL = 20
LAMBDA_REC = 1.0    LAMBDA_DZ = 0.0     LAMBDA_DX = 1e-3
LAMBDA_SP  = 0.01   LAMBDA_VAR = 0.0
THRESHOLD  = 0.1    THRESH_START = 2000 THRESH_EVERY = 500
LEARNING_RATE = 1e-3 (exp decay, transition=500, rate=0.99)   MAX_STEPS = 10000
```

---

## 1. High-level picture

This is a SINDy autoencoder (Champion et al. 2019) rebuilt on top of a
*neural-operator* encoder/decoder pair instead of Champion's fixed-grid MLPs. The
defining architectural choice is that **the encoder and decoder are permutation-
invariant / resolution-independent functions of a (coordinate, value) point cloud**,
rather than functions of a fixed 128-vector. Concretely:

```
 true Lorenz state          high-dim field                latent code            SINDy library
   z(t) in R^3    -- lift -->  u(y,t) in R^128  -- encode -->  ẑ(t) in R^3  --Θ(ẑ)-->  R^20
                                    |                              |
                                    | decode                       | SINDy: dẑ/dt = Θ(ẑ) (mask⊙ξ)
                                    v                              v
                              û(y,t) reconstruction          dẑ/dt predicted
```

Two chain rules are evaluated with **forward-mode automatic differentiation**
(`jax.jvp`) instead of the hand-derived analytic derivative formulas Champion et
al. use in their TF1 code (`sindy_utils.py`'s `z_derivative` functions, which
hard-code the derivative rule for each activation function layer by layer). This
is the central mechanical idea of the whole codebase and is explained in detail in
§5 and §7.

The model has exactly three trainable *groups* of parameters:

- `θ_enc` — weights of `PoolingEncoder` (a DeepSets-style point-cloud encoder)
- `θ_dec` — weights of `LinearDecoder` (a DeepONet-style trunk network)
- `ξ` ("xi") — the SINDy coefficient matrix, `[20, 3]`

and one **non-trainable** array, `mask` (`[20, 3]`, binary), which gates `ξ` and is
updated by hard thresholding between training steps, never by gradient descent.

---

## 2. Data: what a "sample" is

### 2.1 Ground-truth generation (`experiments/lorenz63/training_data.py`, a direct
port of Champion's `example_lorenz.py`)

For each of `N_ICS = 1024` random initial conditions `z0 ∈ R^3`, the Lorenz-63 ODE

```
dz0/dt = σ(z1 − z0)
dz1/dt = z0(ρ − z2) − z1
dz2/dt = z0 z1 − β z2         (σ=10, β=8/3, ρ=28)
```

is integrated with `scipy.odeint` over `t = 0, 0.02, ..., 4.98` (250 steps, `DT=0.02`,
`N_SAMPLES=250`), producing trajectories `z(t) ∈ R^{250×3}`, `dz/dt ∈ R^{250×3}`
(the latter computed by evaluating the ODE's RHS at each `z(t)`, not by finite
differencing — it is *exact*). Both are rescaled by `normalization = [1/40,1/40,1/40]`.

### 2.2 Lifting to a "field" (the neural-operator's function domain)

A spatial grid `y ∈ [-1,1]^128` is fixed once (`y_spatial`), and six Legendre
polynomials `P_0, ..., P_5` are evaluated on it (`modes`, shape `[6, 128]`). With
`LINEAR_OBS = True` (current default), the field is a **purely linear** combination
of the first three Legendre modes weighted by the Lorenz state:

```
u(y, t) = P_0(y)·z_0(t) + P_1(y)·z_1(t) + P_2(y)·z_2(t)                    (linear=True)
du/dt   = P_0(y)·ż_0(t) + P_1(y)·ż_1(t) + P_2(y)·ż_2(t)
```

(If `LINEAR_OBS=False`, cubic terms `P_3·z_0^3 + P_4·z_1^3 + P_5·z_2^3` and their
correct chain-rule time-derivatives are added too — not used currently.) Because the
modes `P_k(y)` don't depend on `t`, `du/dt` is obtained by literally substituting
`ż` for `z` in the same linear combination — no numerical differentiation is needed
for the field derivative either; it is exact by construction.

Flattening over `(ic, timestep)` gives the on-disk arrays actually used:

| array | shape | meaning |
|---|---|---|
| `data['x']` | `[1024·250, 128] = [256000, 128]` | field values `u(y,t)` |
| `data['dx']` | `[256000, 128]` | field time-derivatives `du/dt` |
| `data['y_spatial']` | `[128]` | shared spatial grid, identical for every row |

`NOISE_STRENGTH = 1e-6` adds negligible Gaussian noise on top. This is cached to
`experiments/lorenz63/training_data.npz` / `validation_data.npz` by
`load_or_create_lorenz_data` so it's generated once.

### 2.3 Batch sampling (`sample_batch`, replaces the torch `DataLoader`)

Each training step draws `BATCH_SIZE=128` **i.i.d. random rows** (without
replacement *within* the batch, but with replacement *across* steps — a bootstrap
scheme, not epoch/shuffle-based) from the flattened `[256000, 128]` pool:

```python
idx   = rng.choice(256000, size=128, replace=False)
u     = x[idx][..., None]                     # [128, 128, 1]   (B, n_points, channel)
u_dot = dx[idx][..., None]                    # [128, 128, 1]
x     = broadcast(y_spatial[None,:,None], (128, 128, 1))   # [128, 128, 1] (B, n_points, coord_dim)
```

Note the two different "128"s: `BATCH_SIZE=128` (batch axis) and `INPUT_DIM=128`
(number of spatial grid points). Both happen to equal 128 in the current config,
which is a config coincidence, not a structural requirement.

This differs from Champion et al., who batch **sequentially** without shuffling
(`batch_idxs = arange(j*batch_size, (j+1)*batch_size)`), and from the "3 - NO-AE"
reference notebook, which uses a real shuffled epoch-based `torch.utils.data.DataLoader`.

---

## 3. Encoder: `PoolingEncoder` + `DeepSetPooling`

Input: `u ∈ R^{B×128×1}` (field values), `x ∈ R^{B×128×1}` (grid coordinates,
identical across the batch). `B=128` (batch size) in what follows; `n=128` is the
number of spatial points; both numeric values coincide in this config but are
conceptually distinct axes and are written out below.

**Step 1 — positional encoding.** `positional_encoding = IdentityEncoding()`, so
`x_pos = x`, shape `[B, n, 1]` unchanged. (The framework supports Fourier-feature
encodings here; none are used currently.)

**Step 2 — concatenate coordinate and value channels.**
```
u_cat = concat([x_pos, u], axis=-1)     # [B, n, 2]
```
Each spatial point is now represented as the pair `(y, u(y,t))` — this is what
makes the encoder a genuine *operator*: it consumes `(coordinate, value)` pairs
rather than a fixed-order vector, so it is well-defined for any set of query points,
not just this particular 128-point grid.

**Step 3 — per-point MLP (`DeepSetPooling`, `mlp_dim=64`, `mlp_n_hidden_layers=2`).**
Applied identically (shared weights) to every point independently:
```
h = Dense(64)(u_cat);  h = gelu(h)          # [B, n, 64]
h = Dense(64)(h)                            # [B, n, 64]   (no activation on last MLP layer)
```
(`MLP.__call__` applies `gelu` after every layer except the last.)

**Step 4 — permutation-invariant pooling.**
```
z_pool = mean(h, axis=1)                    # [B, 64]   — mean over the 128 points
```
This is the DeepSets construction (Zaheer et al. 2017): summarizing an unordered
point set by an elementwise-shared transform followed by a symmetric reduction
(here `mean`). It is invariant to the order of the 128 grid points and would
accept a different number of points without any architecture change — the "neural
operator" property.

**Step 5 — linear readout to latent space.**
```
z = Dense(3)(z_pool)                        # [B, 3]     (latent_dim = 3, use_bias=True)
```

**Encoder parameter count:** `Dense(2→64)` = 2·64+64=192, `Dense(64→64)` =
64·64+64=4160, `Dense(64→3)` = 64·3+3=195. Total `θ_enc` ≈ 4547 scalars.

**Summary as one formula:**
```
z = W_3 · mean_{k=1..128} [ GELU(W_1·(y_k, u_k) + b_1) · W_2 + b_2 ]  +  b_3
```
(schematically; `W_2` here folds in the second Dense layer, no activation before the
mean).

---

## 4. The SINDy library `Θ(z)`

`theta_jax(z, poly_order=3, latent_dim=3)` builds, for each row of `z ∈ R^{B×3}`,
every monomial in `z_0, z_1, z_2` up to total degree 3, **matching
`pysindy.PolynomialLibrary(degree=3, include_bias=True)` exactly** but implemented
by hand in JAX so it is differentiable end-to-end:

| degree | terms | count |
|---|---|---|
| 0 | `1` | 1 |
| 1 | `z0, z1, z2` | 3 |
| 2 | `z0², z0z1, z0z2, z1², z1z2, z2²` | 6 |
| 3 | `z0³, z0²z1, z0²z2, z0z1², z0z1z2, z0z2², z1³, z1²z2, z1z2², z2³` | 10 |

Total `p = 20` columns → `Θ(z) ∈ R^{B×20}`. This matches
`library_size(latent_dim=3, poly_order=3) = 20` used to size `ξ`.

**SINDy coefficient matrix**: `ξ ∈ R^{20×3}`, a Flax parameter,
`nn.initializers.constant(1.0)` (every entry starts at exactly 1.0 — matching
Champion et al., who use `tf.constant_initializer(1.0)`; this was recently changed
from `nn.initializers.zeros`).

**Mask**: `mask ∈ {0,1}^{20×3}`, passed as a function argument (not a Flax
parameter — it is never touched by `apply_gradients`). All-ones until thresholding
begins at step 2000.

**Predicted latent velocity:**
```
dz_sindy = Θ(z) @ (mask ⊙ ξ)          # [B,20] @ [20,3] = [B,3]
```
i.e. `ż_k ≈ Σ_{j=1}^{20} Θ_j(z) · mask_{jk} · ξ_{jk}` for each latent channel `k`.
Entries with `mask_jk = 0` contribute exactly zero to both the forward value *and*
(see §7) the gradient — thresholding is a one-way, permanent gate in this
implementation, terms cannot be reinstated later in training.

---

## 5. First JVP: encoder-side latent velocity `dz_enc`

Both `u_t` (the field) and its exact time-derivative `u_dot` are available from the
data (§2.2). The spatial grid `x` does **not** depend on time. Since
`z = enc(u_t, x; θ_enc)`, the exact chain rule for its time-derivative is:

```
dz/dt = (∂enc/∂u)(u_t, x) · (∂u_t/∂t) = J_enc(u_t) · u_dot
```

`J_enc(u_t)` is the (huge, `[3, 128]`) Jacobian of the encoder's output with respect
to its field input, evaluated at `u_t`. We never form this matrix explicitly.
Instead:

```python
enc = lambda uu: encoder(uu, x)
z, dz_enc = jax.jvp(enc, (u_t,), (u_dot,))
```

`jax.jvp` performs **forward-mode automatic differentiation**: it propagates a
"tangent" value alongside the ordinary ("primal") forward pass, through every
operation of the encoder (every `Dense`, `gelu`, `mean`), applying that operation's
local derivative rule to the tangent at each step. The result, `dz_enc`, is
*exactly* `J_enc(u_t) · u_dot` — the directional derivative of the encoder's output
along the direction the input is actually moving in time — computed at the same
asymptotic cost as one extra forward pass, with no finite-difference approximation.

This is doing precisely what Champion et al.'s `sindy_utils.z_derivative` does by
hand for their fixed sigmoid-MLP encoder (chaining `dphi/dx` through each layer
symbolically); `jax.jvp` is the generic, architecture-agnostic version of the same
operation, which is what makes it possible to swap in a totally different encoder
(`PoolingEncoder` here, an MLP there) without re-deriving any derivative formulas.

**Shapes:** primal `z: [B,3]`, tangent `dz_enc: [B,3]` (identical shape to `z`,
since a JVP's output tangent always has the same shape as the output it shadows).

---

## 6. Decoder: `LinearDecoder` (DeepONet-style trunk network)

Inputs: `z ∈ R^{B×3}` (used as basis *coefficients*, not passed through any
further nonlinearity), `x ∈ R^{B×n×1}` (query grid, `n=128`).

**Trunk network** (`basis(x)`, shared across the batch structurally, though
evaluated once per batch row since `x` is batched):
```
h = Dense(64)(x_pos);  h = gelu(h)         # [B, n, 64]
h = Dense(64)(h);      h = gelu(h)         # [B, n, 64]
h = Dense(64)(h);      h = gelu(h)         # [B, n, 64]
h = Dense(n_basis·out_dim = 3)(h)          # [B, n, 3]   (no activation, last MLP layer)
basis = reshape(h, [B, n, 3, 1])           # [B, n, n_basis=3, out_dim=1]
```
So the trunk net produces **3 learned scalar basis functions**
`φ_1(y), φ_2(y), φ_3(y)` (one per latent channel), evaluated at every query point
`y` in the batch's grid.

**Combination (`_forward`):**
```python
u_hat = jnp.einsum("ij,...ikjl->ikl", z, basis)     # [B,n,1]
```
i.e. `û(y_k) = Σ_{j=1}^{3} z_j · φ_j(y_k)` for each batch row. This is exactly a
DeepONet: a "trunk" net producing basis functions of the query coordinate, combined
*linearly* by "branch" coefficients — except here there is no separate branch
network; the SINDy latent state `z` itself is directly identified with the branch
output / basis coefficients (forced by `n_basis = latent_dim = 3` in `config.py`).

**Structural consequence worth flagging explicitly:** because the combination step
is a plain linear contraction with no nonlinearity applied to `z`, **the decoder is
an exactly linear function of `z`** (for fixed `x`, fixed `θ_dec`):
`decoder(z, x) = M(x; θ_dec) · z` for some (batch-and-point-dependent) matrix `M`.
This is a strictly weaker function class than Champion et al.'s decoder, which is a
nonlinear MLP `R^3 → R^{64} → R^{32} → R^{128}` with sigmoid activations throughout
— their decoder can represent nonlinear maps from latent state to field; this one
cannot, by construction. (This was the subject of the earlier `NonlinearDecoder`
discussion.)

**Decoder parameter count:** `Dense(1→64)`=128, `Dense(64→64)`=4160 (×2 more such
layers) , `Dense(64→3)`=195. Total `θ_dec` ≈ 128+4160+4160+195 ≈ 8643 scalars.

---

## 7. Second JVP: decoder-side field velocity `du_dec`

We now ask: *if the latent state actually moved according to the SINDy model's
prediction `dz_sindy` (§4), what field time-derivative would the decoder produce?*
This is again an exact directional derivative, now with respect to the decoder's
first argument:

```python
dec = lambda zz: decoder(zz, x)
u_hat, du_dec = jax.jvp(dec, (z,), (dz_sindy,))
```

```
du_dec = J_dec(z) · dz_sindy
```

Because §6 established `decoder(z,x) = M(x;θ_dec)·z` is **exactly linear in `z`**,
its Jacobian `J_dec(z) = M(x;θ_dec)` does not depend on `z` at all. Consequently:

```
du_dec(y_k) = Σ_{j=1}^{3} φ_j(y_k) · dz_sindy_j
```

— literally the *same* basis-combination formula as the reconstruction `û`, just
fed `dz_sindy` instead of `z`. (`jax.jvp` computes this generically without
"knowing" the decoder is linear; the fact that the tangent computation reduces to
re-running the same linear combination is a property of this specific decoder, not
of the JVP mechanism — it would look different for a nonlinear decoder.)

**Shapes:** `u_hat: [B, n, 1]`, `du_dec: [B, n, 1]` (same shape as `u_hat`).

---

## 8. Full forward pass — shape trace

| step | operation | output shape |
|---|---|---|
| input | `u_t`, `u_dot` | `[128, 128, 1]` each (B, n_points, 1) |
| input | `x` (grid) | `[128, 128, 1]` |
| encoder | `z, dz_enc = jvp(enc, u_t, u_dot)` | `[128, 3]` each |
| SINDy library | `Θ(z)` | `[128, 20]` |
| SINDy prediction | `dz_sindy = Θ(z) @ (mask⊙ξ)` | `[128, 3]` |
| decoder | `u_hat, du_dec = jvp(dec, z, dz_sindy)` | `[128, 128, 1]` each |
| output | `(u_hat, z, dz_enc, dz_sindy, du_dec, ξ)` | — |

The model's `__call__` returns all six of these; `sindy_ae_loss` consumes them.

---

## 9. Loss function (`src/losses.py`)

```
loss_rec = mean( (û − u_t)² )                      over all 128·128·1 elements
loss_dz  = mean( (dz_enc − dz_sindy)² )             over all 128·3 elements
loss_dx  = mean( (du_dec − u_dot)² )                over all 128·128·1 elements
loss_sp  = sum( |mask ⊙ ξ| )                        L1, sum not mean, over 20·3=60 entries
loss_var = mean( (Var_batch(z_k) − 1)² )            over k=1..3, batch-variance per latent channel

total = λ_rec·loss_rec + λ_dz·loss_dz + λ_dx·loss_dx + λ_sp·loss_sp + λ_var·loss_var
```

**Current weights:** `λ_rec=1.0, λ_dz=0.0, λ_dx=1e-3, λ_sp=0.01, λ_var=0.0`.

Interpretation of each term:
- **`loss_rec`** — standard autoencoder reconstruction: can the encoder→decoder
  pipeline reproduce the input field at all?
- **`loss_dz`** — *latent* consistency: does the encoder's own time-derivative
  `dz_enc` (computed from the true field derivative, §5) agree with what the SINDy
  model predicts (`dz_sindy`, §4)? Currently weighted **zero** — disabled.
- **`loss_dx`** — *field* consistency: if you trust the SINDy model's latent
  velocity and push it back through the decoder (§7), does it reproduce the true
  field derivative? This is currently the *only* term connecting `ξ` to a
  correctness signal (since `loss_dz` is off).
- **`loss_sp`** — L1 sparsity on the (masked) coefficients — the differentiable
  proxy for "most of these 60 coefficients should be exactly zero."
- **`loss_var`** — gauge-fix: pins each latent channel's batch variance near 1.
  Necessary because the model has an exact continuous symmetry `z → c·z, ξ → ξ/c`
  (verify: `Θ(cz)` scales degree-`d` monomials by `c^d`, not uniformly by `c`, so
  this symmetry is *not* exact for `poly_order>1` — but for the degree-1 part it
  is, and empirically the model can still trade off scale between `z` and `ξ`
  under gradient descent). **Currently weighted zero — inert.**

---

## 10. Backpropagation: how gradients reach every parameter

`make_train_step` calls `jax.value_and_grad(loss_fn, has_aux=True)(state.params)`
— ordinary **reverse-mode** automatic differentiation over the *entire* forward
pass in §3–§9, including both `jax.jvp` calls. JAX supports differentiating
through a `jvp` call natively (this composition — reverse-mode wrapped around
forward-mode — is sometimes called "reverse-over-forward"): each `jvp` is
implemented internally as a linearization of the primal computation plus its
tangent; `grad`/`vjp` differentiates through that linearized tangent computation by
transposing it. Practically, each gradient step costs roughly the equivalent of
2 forward passes (primal + tangent, for each of the two JVPs) plus one backward
pass through the whole graph — more expensive than a plain autoencoder step, exact
(no finite differences), and fully generic to whatever `encoder`/`decoder` modules
are plugged in.

### 10.1 Gradient path for `θ_enc` (encoder weights)

`θ_enc` affects the loss through:
1. `loss_rec`, directly: `z = enc(u_t,x;θ_enc)` feeds `û = decoder(z,x)`.
2. `loss_dz`, through **two** routes simultaneously: `dz_sindy` depends on
   `z=enc(u_t;θ_enc)` via `Θ(z)`, *and* `dz_enc` is itself
   `J_enc(u_t;θ_enc)·u_dot` — a quantity that depends on `θ_enc` through the
   encoder's *Jacobian*, not just its value. Computing `∂loss_dz/∂θ_enc` is
   therefore a genuine second-order-in-effect differentiation: you are
   differentiating (w.r.t. `θ_enc`) an expression that already contains one
   derivative (w.r.t. `u_t`) of the same network. This is exactly the
   "reverse-over-forward" composition described above — JAX computes it correctly
   and automatically, but it is mathematically a heavier object than a first
   derivative. **Currently `λ_dz=0`, so this whole path (both halves) is zeroed out
   and contributes nothing.**
3. `loss_dx`, through `z` → `Θ(z)` → `dz_sindy` → `du_dec` (first order in `θ_enc`,
   since — unlike `dz_enc` — `dz_sindy` does not involve any derivative of the
   *encoder*, only of `Θ`, which is a fixed closed-form function). **Active.**
4. `loss_var`, directly through `z`. **Currently inert (`λ_var=0`).**

**Net effect under the current config: `θ_enc` receives gradient only from
`loss_rec` (direct) and `loss_dx` (indirect, via `z→Θ(z)→dz_sindy→du_dec`).** The
expensive, encoder-Jacobian-aware path (`loss_dz`) is fully disabled.

### 10.2 Gradient path for `θ_dec` (decoder / trunk-net weights)

1. `loss_rec`, directly: `û = decoder(z,x;θ_dec)`.
2. `loss_dx`, directly: `du_dec = decoder(dz_sindy,x;θ_dec)` — and because the
   decoder is *exactly linear* in its first argument (§6–§7), this is **not** a
   second-order path despite flowing through a `jvp`: `J_dec` doesn't depend on
   `θ_dec`'s effect on `z` at all, only on `θ_dec` directly, through the same
   trunk-net weights that also produce `û`. So `∂loss_dx/∂θ_dec` is an ordinary
   first derivative, unlike the analogous `∂loss_dz/∂θ_enc` path in §10.1 — the
   asymmetry comes entirely from the decoder being linear in `z` while the encoder
   is nonlinear in `u`.

**Net effect: `θ_dec` receives gradient from both `loss_rec` and `loss_dx`, always
first-order, currently both active.**

### 10.3 Gradient path for `ξ`

`ξ` appears in exactly two places: `dz_sindy = Θ(z)·(mask⊙ξ)` (linear, feeding
`loss_dz` and, via `du_dec`, `loss_dx`) and directly in `loss_sp`. So:

```
∂total/∂ξ = λ_dz · ∂loss_dz/∂ξ  +  λ_dx · ∂loss_dx/∂ξ  +  λ_sp · sign(ξ) ⊙ mask
          =    0  · (...)       +  1e-3 · ∂loss_dx/∂ξ  +  0.01 · sign(ξ) ⊙ mask     [current config]
```

Because `dz_sindy` is *linear* in `mask⊙ξ`, and `mask_jk=0 ⇒ ∂(mask⊙ξ)_jk/∂ξ_jk = 0`,
masked-out entries receive **exactly zero gradient** — once an entry is
thresholded to zero it can never be revived by further training; thresholding is a
one-way ratchet in this implementation.

**This is the mechanistically important part for the "0 active terms" symptom.**
The L1 term's gradient magnitude on any active entry is a *constant* `λ_sp = 0.01`
(the derivative of `|ξ|` is `±1`, independent of how large `ξ` is), whereas the
consistency term's pull is scaled by `λ_dx = 1e-3` *and* further attenuated by
however small `∂loss_dx/∂ξ` actually is in practice (bounded by how well the
decoder-side reconstruction error responds to each individual coefficient, which
for a mostly-linear decoder over a 3-dim library can be quite small per entry).
With `λ_sp` roughly 10–100× larger in raw weight than `λ_dx`, and with `λ_var=0` so
nothing stops `z`'s scale (and thus the natural operating scale of `ξ`) from
shrinking too, sustained L1 pressure with comparatively weak opposing signal is
expected to drive most or all of `ξ` under the `THRESHOLD=0.1` cutoff by the time
thresholding starts at step 2000 — consistent with the observed failure mode.

---

## 11. Training loop (`train`, `src/training.py`)

```
tx = optax.adam(exponential_decay(lr=1e-3, transition_steps=500, decay_rate=0.99))
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

for step in 1..MAX_STEPS (=10000):
    u_t, u_dot, x = sample_batch(training_data, BATCH_SIZE=128, rng)   # §2.3
    state, total, aux = train_step(state, u_t, u_dot, x, mask)         # one Adam update
    if step % 1000 == 0: print max|ξ|
    if step >= THRESH_START(=2000) and step % THRESH_EVERY(=500) == 0:
        mask = mask * (|ξ| >= THRESHOLD(=0.1))     # hard, permanent, sequential threshold
        print active-term count
```

No epoch structure at all — every step is an independent random batch draw; the
loop is a flat step budget, not `epochs × steps_per_epoch`. Thresholding events
happen at steps 2000, 2500, 3000, ..., 10000 (16 events over the run).

At LR-decay step 10000 (20 completed transition periods of 500 steps),
`lr ≈ 1e-3 · 0.99^20 ≈ 8.2e-4` — a mild decay, not aggressive.

---

## 12. Summary of currently-known structural gaps vs. Champion et al. (2019)

These are documented for completeness; none of them have been changed yet, pending
the "grounding" decision discussed separately.

| aspect | this codebase | Champion et al. |
|---|---|---|
| encoder/decoder | DeepSets pooling encoder / linear DeepONet-trunk decoder, operates on point clouds | fixed-width sigmoid MLPs, operate on fixed 128-vectors |
| decoder nonlinearity | **exactly linear** in `z` | nonlinear (sigmoid MLP) |
| derivative computation | `jax.jvp` (automatic, generic) | hand-derived analytic per-activation formulas |
| `λ_dz` (latent consistency) | 0.0 (disabled) | 0.0 (also disabled — this one matches) |
| `λ_dx` (field consistency) | 1e-3 | 1e-4 |
| `λ_sp` (sparsity) | 0.01 (`sum`-based) | 1e-5 (`mean`-based ⇒ ≈1.7e-7 in `sum`-based terms) |
| `λ_var` (gauge fix) | 0.0 (disabled) | no equivalent term exists in Champion's model |
| batching | i.i.d. random bootstrap sampling every step | sequential, non-shuffled per-epoch batches |
| training budget | flat 10,000 steps, no refinement phase | thousands of epochs + a dedicated sparsity-off "refinement" phase |
| `ξ` init | `constant(1.0)` | `constant(1.0)` (matches) |
| thresholding rule | hard `|ξ|≥0.1` mask, permanent | hard `|ξ|>0.1` mask, permanent (matches) |

The `λ_sp`/`λ_var` mismatch (row 6–7) is the most directly implicated in the "0
active terms" failure per the gradient analysis in §10.3.
