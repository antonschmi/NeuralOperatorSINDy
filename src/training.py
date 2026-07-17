import dataclasses
import pickle
import time
import pysindy as ps
import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from itertools import combinations_with_replacement
from pathlib import Path


def save_checkpoint(path, state, mask, loss_hist):
    """
    Dump {params, mask, loss_hist} to `path` as plain-numpy pickle, so a run can be
    resumed/inspected even if the process is killed mid-training (e.g. a Colab
    disconnect) -- JAX device arrays aren't reliably picklable across sessions, so
    everything is converted to numpy first.
    """
    payload = {
        'params': jax.tree_util.tree_map(np.array, state.params),
        'opt_state': jax.tree_util.tree_map(np.array, state.opt_state),
        'step': int(state.step),
        'mask': np.array(mask),
        'loss_hist': loss_hist,
    }
    tmp_path = Path(str(path) + '.tmp')
    with open(tmp_path, 'wb') as f:
        pickle.dump(payload, f)
    tmp_path.replace(path)  # atomic-ish: never leaves a half-written checkpoint at `path`


def load_checkpoint(path):
    """Load a `save_checkpoint` payload, or return None if `path` doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


from src.encoder.pooling_encoder import PoolingEncoder
from src.decoder.linear_decoder import LinearDecoder
from src.decoder.non_linear_decoder import NonlinearDecoder
from src.decoder.deeponet_decoder import DeepONetDecoder
from src.utils.networks.pooling import DeepSetPooling
from experiments.lorenz63.training_data import (get_lorenz_data, library_size)
from experiments.lorenz63.config import Config
from src.losses import sindy_ae_loss


def load_or_create_lorenz_data(path, n_ics, noise_strength, linear=True):
    """
    Load a cached get_lorenz_data dict from `path` if it exists, otherwise
    generate it and cache it there for next time.
    """
    if path.exists():
        with np.load(path) as npz:
            return {k: npz[k] for k in npz.files}
    data = get_lorenz_data(n_ics, noise_strength=noise_strength, linear=linear)
    np.savez(path, **data)
    return data


def _coords(data):
    """
    Normalize `data['y_spatial']` to shape [n_points, in_dim]. Lorenz's cached data
    stores a 1D grid ([n_points], in_dim=1 implicit); reaction-diffusion's 2D (y1,y2)
    grid is already [n_points, 2]. Keeping this check here (rather than forcing every
    experiment's data loader to agree on a convention) means the existing Lorenz .npz
    cache never needs regenerating just to satisfy a shape contract.
    """
    y = jnp.asarray(data['y_spatial'])
    return y[:, None] if y.ndim == 1 else y


def sample_batch(data, batch_size, rng):
    """
    Draw a random batch of (u, u_dot, x) from a get_lorenz_data-style dict, in place
    of a torch DataLoader. `data['x']`/`data['dx']` hold one row per (ic, timestep)
    snapshot, each defined on the shared spatial grid `data['y_spatial']`.
    """
    idx = rng.choice(data['x'].shape[0], size=batch_size, replace=False)
    u = jnp.asarray(data['x'][idx])[..., None]        # [B, n_points, 1]
    u_dot = jnp.asarray(data['dx'][idx])[..., None]    # [B, n_points, 1]
    y = _coords(data)                                  # [n_points, in_dim]
    x = jnp.broadcast_to(y[None, :, :], (batch_size, *y.shape))
    return u, u_dot, x


def sample_batch_subsampled(data, batch_size, n_sub, rng, key):
    """
    Like `sample_batch`, but each row in the batch gets its OWN independently-random
    subset of `n_sub` (out of the full n_points_full) spatial points -- a genuinely
    different point-cloud configuration per sample, not just per step. `sample_batch`
    always hands every row the exact same fixed grid, so nothing before this ever
    forced the encoder/decoder to handle heterogeneous point clouds within a single
    gradient step -- this is the actual test of the mesh-invariance claim.

    `key` is an ordinary jax.random.PRNGKey, split fresh by the caller every call; kept
    separate from `rng` (the numpy Generator used for the outer ic/timestep sampling,
    unchanged from `sample_batch`) since the per-point subsampling runs through
    jax.random so it can be vmapped across the batch.
    """
    idx = rng.choice(data['x'].shape[0], size=batch_size, replace=False)
    u_full = jnp.asarray(data['x'][idx])          # [B, n_points_full]
    udot_full = jnp.asarray(data['dx'][idx])      # [B, n_points_full]
    y_full = _coords(data)                        # [n_points_full, in_dim]
    n_points_full = y_full.shape[0]

    keys = jax.random.split(key, batch_size)

    def pick_and_gather(k, u_row, udot_row):
        sub_idx = jax.random.choice(k, n_points_full, shape=(n_sub,), replace=False)
        return u_row[sub_idx], udot_row[sub_idx], y_full[sub_idx]

    u, udot, x = jax.vmap(pick_and_gather)(keys, u_full, udot_full)
    return u[..., None], udot[..., None], x   # x is already [B, n_sub, in_dim]


def theta_jax(z, poly_order, latent_dim, include_sine=False):
    """
    JAX polynomial (+ optional sine) library, for clean JAX derivative computation.
    Matches `build_feature_library`'s column order exactly: polynomial terms up to
    `poly_order` (with bias) first, then, if `include_sine`, sin(z_i) for each i
    appended last -- Champion et al.'s convention (`sindy_library`/`sindy_library_tf`),
    needed for reaction-diffusion and the pendulum but not Lorenz.
    """
    B = z.shape[0]
    cols = [jnp.ones(B)]
    for deg in range(1, poly_order + 1):
        for idx in combinations_with_replacement(range(latent_dim), deg):
            feat = jnp.ones(B)
            for k in idx:
                feat = feat * z[:, k]
            cols.append(feat)
    if include_sine:
        for i in range(latent_dim):
            cols.append(jnp.sin(z[:, i]))
    return jnp.stack(cols, axis=-1)


def build_feature_library(poly_order, latent_dim, include_sine=False):
    """
    Build the pysindy library used to size Theta(z) (`N_FEATURES`) and to get
    human-readable feature names -- matching `theta_jax`'s column order exactly.

    Returns a fitted library exposing `.n_output_features_` and `.transform(z)`.
    Feature names are read through `.get_feature_names(...)`, which both
    `PolynomialLibrary` and `ConcatLibrary` support (unlike the sklearn-style
    `get_feature_names_out` alias, which `ConcatLibrary` does not implement).
    """
    poly = ps.PolynomialLibrary(degree=poly_order, include_bias=True)
    if not include_sine:
        lib = poly
    else:
        fourier = ps.FourierLibrary(n_frequencies=1, include_sin=True, include_cos=False)
        lib = ps.ConcatLibrary([poly, fourier])
    lib.fit([np.zeros((2, latent_dim))])
    return lib


def sr3_refit(model, params, cfg, training_data, rng, key):
    """
    Path A sparsity mechanism: instead of the co-trained `xi` being thresholded
    directly (absolute |xi| >= THRESHOLD), freeze the encoder at its current `params`
    and refit `xi` from scratch via pysindy's SR3 optimizer -- an L0-regularized sparse
    regression solve on a precomputed library -- rather than gradient descent + hard
    threshold. Called at the same periodic threshold-step cadence `run_phase` already
    uses; the caller is responsible for AND-ing the returned support into the running
    mask (monotone: support may only shrink) and writing `mask * new_Xi` back into
    `state.params['xi']`.

    Computes (z, dz_enc) over `cfg.loss.SR3_N_SAMPLES` points using the SAME
    diff-then-encode jax.jvp path SINDyAE.__call__ uses for the consistency loss
    (`_encode` is just `self.encoder(u, x)`, so calling `model.encoder.apply(...)`
    wrapped in jax.jvp directly is bit-identical), batched and concatenated to numpy
    on the host. Builds Theta(z) via the existing `theta_jax` (not a fresh
    ps.PolynomialLibrary) so ordering/degree match the co-trained loss exactly -- and
    verifies that claim against a numpy ps.PolynomialLibrary build on a small shared
    sample via an assert, since a silent library-ordering mismatch would make every
    downstream coefficient meaningless without ever raising an error.

    Returns (new_Xi, new_support) as jax arrays shaped (n_library, latent_dim) --
    matching the `xi` param leaf's orientation. pysindy's SR3.coef_ is
    (latent_dim, n_library), the opposite convention (verified empirically, not
    assumed); this transposes before returning, and asserts the result matches the
    `xi` leaf's shape.
    """
    encoder_params = params['encoder']
    y_full = _coords(training_data)
    n_points_full = y_full.shape[0]
    include_sine = getattr(cfg.model, 'INCLUDE_SINE', False)

    n_samples = min(cfg.loss.SR3_N_SAMPLES, training_data['x'].shape[0])
    idx = rng.choice(training_data['x'].shape[0], size=n_samples, replace=False)

    batch_size = cfg.training.BATCH_SIZE
    z_chunks, dz_chunks = [], []
    for start in range(0, n_samples, batch_size):
        chunk_idx = idx[start:start + batch_size]
        b = chunk_idx.shape[0]

        if cfg.model.SUBSAMPLE_POINTS:
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, b)
            u_full = jnp.asarray(training_data['x'][chunk_idx])
            udot_full = jnp.asarray(training_data['dx'][chunk_idx])

            def pick(k, u_row, udot_row):
                sub_idx = jax.random.choice(k, n_points_full, shape=(cfg.model.N_SUB,), replace=False)
                return u_row[sub_idx], udot_row[sub_idx], y_full[sub_idx]

            u_b, udot_b, x_b = jax.vmap(pick)(keys, u_full, udot_full)
            u_b, udot_b = u_b[..., None], udot_b[..., None]   # x_b is already [b, N_SUB, in_dim]
        else:
            u_b = jnp.asarray(training_data['x'][chunk_idx])[..., None]
            udot_b = jnp.asarray(training_data['dx'][chunk_idx])[..., None]
            x_b = jnp.broadcast_to(y_full[None, :, :], (b, *y_full.shape))

        enc = lambda uu: model.encoder.apply({'params': encoder_params}, uu, x_b)
        z_b, dz_b = jax.jvp(enc, (u_b,), (udot_b,))
        z_chunks.append(np.array(z_b))
        dz_chunks.append(np.array(dz_b))

    z_np = np.concatenate(z_chunks, axis=0)
    dz_np = np.concatenate(dz_chunks, axis=0)

    # library identity guard -- the #1 silent failure mode: JAX builder vs numpy
    # library must agree on a shared sample before we trust either.
    z_probe = z_np[:min(64, z_np.shape[0])]
    theta_jax_probe = np.array(theta_jax(jnp.asarray(z_probe), cfg.model.POLY_ORDER, cfg.model.LATENT_DIM, include_sine))
    _lib_probe = build_feature_library(cfg.model.POLY_ORDER, cfg.model.LATENT_DIM, include_sine)
    theta_np_probe = _lib_probe.transform(z_probe)
    if not np.allclose(theta_jax_probe, theta_np_probe, atol=1e-5, rtol=1e-5):
        raise AssertionError(
            "sr3_refit: theta_jax(...) and build_feature_library(...) disagree on this z "
            "sample -- library ordering/degree mismatch. Refusing to fit SR3 on an "
            "unverified library."
        )

    Theta_np = np.array(theta_jax(jnp.asarray(z_np), cfg.model.POLY_ORDER, cfg.model.LATENT_DIM, include_sine))

    opt = ps.SR3(
        reg_weight_lam=cfg.loss.SR3_LAM,
        regularizer='L0',
        relax_coeff_nu=cfg.loss.SR3_NU,
    )
    opt.fit(Theta_np, dz_np)
    coef = np.asarray(opt.coef_)  # pysindy convention: (latent_dim, n_library)

    new_Xi_np = coef.T  # -> (n_library, latent_dim), our convention
    if new_Xi_np.shape != params['xi'].shape:
        raise AssertionError(
            f"sr3_refit: new_Xi shape {new_Xi_np.shape} does not match the xi leaf's "
            f"shape {params['xi'].shape} after transpose -- orientation assumption is wrong."
        )

    new_Xi = jnp.asarray(new_Xi_np)
    new_support = (new_Xi != 0).astype(jnp.float32)

    active = int(new_support.sum())
    residual = float(np.mean((Theta_np @ new_Xi_np - dz_np) ** 2))
    print(f'SR3 refit: active {active}/{new_support.size}  residual(mean sq) {residual:.3e}  '
          f'max|Xi| {np.abs(new_Xi_np).max():.3f}  n_samples {n_samples}')
    print('SR3 support (rows=library terms, cols=latent dims):')
    print(np.array(new_support).astype(int))
    print('SR3 coefficients:')
    print(new_Xi_np)

    return new_Xi, new_support


def _reset_xi_adam_moments(opt_state):
    """
    Zero the Adam (mu, nu) moment estimates for the `xi` leaf only, leaving the shared
    `count` (global step, used for every leaf's bias correction) and every other leaf
    (encoder, decoder) untouched. Used after an SR3 refit overwrites `xi` directly,
    since the old moments would otherwise be stale relative to the just-overwritten
    value -- a transient that would fight the new value for the first several steps.
    """
    adam_state, *rest = opt_state
    new_mu = dict(adam_state.mu)
    new_nu = dict(adam_state.nu)
    new_mu['xi'] = jnp.zeros_like(adam_state.mu['xi'])
    new_nu['xi'] = jnp.zeros_like(adam_state.nu['xi'])
    new_adam_state = adam_state._replace(mu=new_mu, nu=new_nu)
    return (new_adam_state, *rest)


def _encode_correlation(model, params, training_data, rng, n_samples=512):
    """
    Encode a small, fresh random sample of `training_data` and correlate each latent
    channel against each column of the known true state `training_data['z']` -- the
    same check `analysis.ipynb`'s coordinate-recovery cell does post-hoc, just live.

    Only meaningful for experiments that ship a ground-truth low-dimensional state to
    compare against (Lorenz does -- the field is a synthetic lift of a known 3-number
    state, kept purely for validation). Returns None for experiments like
    reaction-diffusion, where there is no known low-dimensional ground truth at all;
    callers must handle that rather than assuming a matrix comes back.

    Uses its own dedicated `rng`, independent of the training batch sampler's `rng`, so
    this diagnostic never perturbs the actual sequence of training batches drawn.
    """
    if 'z' not in training_data:
        return None

    # `x`/`dx` are flattened by get_lorenz_data to one row per (ic, timestep) snapshot
    # -- but `z` (straight out of generate_lorenz_data) keeps its original
    # (n_ics, n_steps, latent_dim) shape and is never reshaped to match. Flatten it the
    # same way (row-major, identical convention to x's own `.reshape((-1, input_dim))`)
    # before indexing with the same `idx` used for x, so z_true[k] and x[k] refer to
    # the same (ic, timestep) snapshot rather than indexing into the wrong axis.
    z_all = np.asarray(training_data['z'])
    if z_all.ndim == 1:
        z_all = z_all[:, None]
    elif z_all.ndim > 2:
        z_all = z_all.reshape(-1, z_all.shape[-1])

    n_pool = min(training_data['x'].shape[0], z_all.shape[0])
    n = min(n_samples, n_pool)
    idx = rng.choice(n_pool, size=n, replace=False)
    u = jnp.asarray(training_data['x'][idx])[..., None]
    y = _coords(training_data)
    x = jnp.broadcast_to(y[None, :, :], (n, *y.shape))
    z_enc = np.array(model.encoder.apply({'params': params['encoder']}, u, x))

    z_true = z_all[idx]

    latent_dim = z_enc.shape[-1]
    true_dim = z_true.shape[-1]
    return np.array([
        [np.corrcoef(z_enc[:, i], z_true[:, j])[0, 1] for j in range(true_dim)]
        for i in range(latent_dim)
    ])


def _log_diagnostics(step, total_steps, aux, mask, t_start, steps_already_done,
                      model, params, training_data, diag_rng):
    """
    Print the full loss breakdown and, if the experiment ships a known ground-truth
    state (Lorenz does; reaction-diffusion doesn't), a correlation matrix between each
    encoded latent channel and each true state variable. Meant to run far less often
    than the per-step loss_hist bookkeeping (every `log_every` steps, and whenever a
    checkpoint is saved) -- this is for human monitoring at the console, not analysis,
    so unlike loss_hist none of it is persisted.

    The correlation matrix is a direct answer to "has a latent variable been turned
    off, and if not, which true state variable (if any) does it actually track" -- a
    plain per-axis mean/std can only hint at collapse (std -> 0) and says nothing about
    whether a *non*-collapsed axis is tracking anything physically meaningful at all. A
    genuinely collapsed (constant) axis shows up here as NaN (undefined correlation),
    which is a more obvious signal than a small-but-nonzero std buried in a row of six
    numbers -- and a non-collapsed but physically meaningless axis (e.g. tracking noise)
    shows up as a row with no large entries anywhere, which mean/std could never reveal.
    """
    elapsed = time.time() - t_start
    done_this_run = step - steps_already_done
    rate = done_this_run / elapsed if elapsed > 0 else float('nan')
    eta_min = (total_steps - step) / rate / 60 if rate > 0 else float('nan')
    n_active = int(np.array(mask).sum())

    print(f'[log {step}/{total_steps}]  ({rate:.1f} steps/s, elapsed {elapsed/60:.1f} min, ETA {eta_min:.1f} min)')
    print(f'  loss {float(aux["loss"]):.3e}  rec {float(aux["loss_rec"]):.3e}  '
          f'dz {float(aux["loss_dz"]):.3e}  dx {float(aux["loss_dx"]):.3e}  '
          f'sp {float(aux["loss_sp"]):.3e}  var {float(aux["loss_var"]):.3e}  '
          f'active {n_active}/{mask.size}')

    corr = _encode_correlation(model, params, training_data, diag_rng)
    if corr is None:
        print('  (no ground-truth z available for this experiment -- skipping correlation check)')
    else:
        with np.printoptions(precision=3, suppress=True):
            print(f'  corr(z_enc, z_true) [rows=z_enc_i, cols=z_true_j]:\n{corr}')
        best_match = np.argmax(np.abs(corr), axis=1)
        max_abs = np.abs(corr).max(axis=1)
        print(f'  best true-state match per z_enc row: {best_match.tolist()}   '
              f'max|corr|: {np.round(max_abs, 3).tolist()}')


def make_decoder(cfg):
    
    """
    Build the model's decoder from `cfg.model.DECODER`, so switching between decoders
    is a single config-file change:
      - "linear":    LinearDecoder -- DeepONet-style, exactly linear in z (no bias),
                     no branch net at all (z itself is the branch output)
      - "nonlinear": NonlinearDecoder -- MLP over concat(tile(z), x), nonlinear in z,
                     but z and x are entangled in one shared network
      - "deeponet":  DeepONetDecoder -- full branch(z) + trunk(x) DeepONet, nonlinear
                     in z like NonlinearDecoder, but z and x never share a network like
                     LinearDecoder's trunk -- see DeepONetDecoder's docstring
    All three share the same (z, x) -> u call signature, so nothing else in the model
    needs to change when switching.
    """
    if cfg.model.DECODER == "linear":
        return LinearDecoder(
            out_dim=1,
            n_basis=cfg.model.LATENT_DIM,  # z's basis coefficients are the SINDy latent states themselves
            features=(64, 64, 64),
        )
    elif cfg.model.DECODER == "nonlinear":
        return NonlinearDecoder(
            out_dim=1,
            features=(64, 64, 64),
        )
    elif cfg.model.DECODER == "deeponet":
        return DeepONetDecoder(
            out_dim=1,
            n_basis=20,  # decoupled from LATENT_DIM -- gives the branch net room to represent
                         # non-affine (e.g. z_i and z_i^3) dependencies without inflating the
                         # SINDy latent dimension itself
            branch_features=(64, 64),
            trunk_features=(64, 64, 64),
        )
    else:
        raise ValueError(
            f"Unknown cfg.model.DECODER: {cfg.model.DECODER!r} (expected 'linear', 'nonlinear', or 'deeponet')"
        )


class SINDyAE(nn.Module):
    """
        Autoencoder with SINDy latent dynamics.

    """

    encoder:    nn.Module
    decoder:    nn.Module
    n_features: int            # p = theta library size
    latent_dim: int
    poly_order: int
    include_sine: bool = False  # reaction-diffusion needs sin(z_i) terms; Lorenz doesn't

    def _encode(self, u, x):
        # encode
        z = self.encoder(u, x)
        return z
        # #
        # s = jax.lax.stop_gradient(jnp.std(z, axis=0, keepdims=True))
        # return z / (s + 1e-4)                      # gauge fix: pin per-axis scale

    @nn.compact
    def __call__(self, u_t, u_dot, x, mask):
        xi = self.param('xi', nn.initializers.constant(1.0), (self.n_features, self.latent_dim))

        # encode the input
        enc = lambda uu: self._encode(uu, x)
        # compute the forward jacobian-vector product to get the latent representation and its derivative
        z, dz_enc = jax.jvp(enc, (u_t,), (u_dot,))

        # SINDy prediction of the latent velocity (masked coefficients)
        dz_sindy = theta_jax(z, self.poly_order, self.latent_dim, self.include_sine) @ (mask * xi)

        # decoder-side: reconstruction and predicted field
        dec = lambda zz: self.decoder(zz, x)
        u_hat, du_dec = jax.jvp(dec, (z,), (dz_sindy,))

        return u_hat, z, dz_enc, dz_sindy, du_dec, xi  
    


def make_train_step(cfg):
    """
    Build a jitted training step. `cfg` (and its weighted loss terms) is closed over
    rather than passed in, since it never changes over the course of training and
    dataclasses aren't valid jit arguments.
    """

    @jax.jit
    def train_step(state, u_t, u_dot, x, mask):
        def loss_fn(params):
            bound_model = lambda *a: state.apply_fn({'params': params}, *a)
            return sindy_ae_loss(cfg, bound_model, (u_t, u_dot, x), mask)

        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, total, aux

    return train_step


def train(cfg, model, params, mask, training_data, rng, checkpoint_path=None, checkpoint_every=50_000,
          resume_from=None, key=None, log_every=10_000):
    """
    Train `model` with hard sequential thresholding of the SINDy coefficients `xi`
    (STLSQ-style: |xi| < cfg.loss.THRESHOLD -> 0), drawing random batches straight
    from `training_data` each step, for a fixed cfg.training.MAX_STEPS gradient steps
    (no epoch bookkeeping - each step is an independent random batch). Then runs a
    refinement phase (Champion et al.'s `refinement_epochs`): continues training with
    `mask` frozen and the sparsity loss dropped entirely, letting the surviving
    coefficients settle to their best-fit values without the L1 term's residual
    shrinkage bias.

    If `cfg.model.SUBSAMPLE_POINTS` is set, batches are drawn with `sample_batch_subsampled`
    instead of `sample_batch` -- every row gets its own independently-random `cfg.model.N_SUB`-point
    subset of the grid, rather than every row sharing the same fixed full grid -- and
    `key` (a jax.random.PRNGKey) must be given; it's split fresh every step.

    If `checkpoint_path` is given, {params, opt_state, mask, loss_hist} are pickled
    there every `checkpoint_every` steps and again at the end -- with step budgets
    in the hundreds of thousands to millions, this is what makes a run recoverable
    after a Colab disconnect or any other mid-training interruption.

    Every `log_every` steps, and again whenever a checkpoint is saved, the full loss
    breakdown (not just `max|xi|`) and, when the experiment ships a ground-truth `z`
    (Lorenz does), a correlation matrix between encoded and true latent state are
    printed to the console (see `_log_diagnostics`) -- this is console-only monitoring,
    not persisted, and separate from the per-step scalar loss terms that get appended
    to `loss_hist` every step regardless.

    `resume_from`, if given, is a `load_checkpoint(...)` payload: training picks up
    from the saved params/opt_state/mask/loss_hist and `state.step` rather than
    starting fresh from `params`/`mask` -- `state.step` (an ordinary optax counter,
    incremented once per `apply_gradients` call and never reset between phases) is
    exactly the number of gradient updates already done, so it doubles as "how far
    into main-phase-then-refinement-phase are we", used below to skip whichever
    steps (and threshold events) already happened before the checkpoint was written.

    Returns the final TrainState, the final (pruned) mask, and a history of the
    per-step loss terms.
    """
    if cfg.model.SUBSAMPLE_POINTS and key is None:
        raise ValueError("cfg.model.SUBSAMPLE_POINTS is True but no `key` (jax.random.PRNGKey) was given")
    if cfg.loss.SPARSITY_METHOD == 'sr3' and key is None:
        raise ValueError("cfg.loss.SPARSITY_METHOD is 'sr3' but no `key` (jax.random.PRNGKey) was given")

    tx = optax.adam(learning_rate=optax.exponential_decay(
        cfg.training.LEARNING_RATE,
        transition_steps=cfg.training.LR_TRANSITION_STEPS,
        decay_rate=cfg.training.LR_DECAY_RATE,
    ))

    if resume_from is not None:
        state = train_state.TrainState(
            step=resume_from['step'],
            apply_fn=model.apply,
            params=jax.tree_util.tree_map(jnp.asarray, resume_from['params']),
            tx=tx,
            opt_state=jax.tree_util.tree_map(jnp.asarray, resume_from['opt_state']),
        )
        mask = jnp.asarray(resume_from['mask'])
        loss_hist = list(resume_from['loss_hist'])
        print(f'resuming from step {int(state.step)}  (loss_hist has {len(loss_hist)} entries)')
    else:
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        mask = mask
        loss_hist = []

    t_start = time.time()
    total_steps = cfg.training.MAX_STEPS + cfg.training.REFINEMENT_STEPS
    steps_already_done = int(state.step)
    # Dedicated, independent RNG for the periodic diagnostic correlation check --
    # deliberately separate from `rng` (used for actual training batches) so this
    # monitoring-only computation never perturbs the training batch sequence.
    diag_rng = np.random.default_rng(getattr(cfg.model, 'SEED', 0) + 1)

    def run_phase(state, mask, n_steps, train_step, do_threshold, step_offset, start_i=0):
        nonlocal key
        for i in range(start_i + 1, n_steps + 1):
            step = step_offset + i
            if cfg.model.SUBSAMPLE_POINTS:
                key, subkey = jax.random.split(key)
                u_t, u_dot, x = sample_batch_subsampled(
                    training_data, cfg.training.BATCH_SIZE, cfg.model.N_SUB, rng, subkey,
                )
            else:
                u_t, u_dot, x = sample_batch(training_data, cfg.training.BATCH_SIZE, rng)
            state, total, aux = train_step(state, u_t, u_dot, x, mask)
            loss_hist.append({k: float(v) for k, v in aux.items() if k != 'z'})

            if step % log_every == 0:
                _log_diagnostics(step, total_steps, aux, mask, t_start, steps_already_done,
                                  model, state.params, training_data, diag_rng)

            if i % 1000 == 0 or i == start_i + 1:
                elapsed = time.time() - t_start
                done_this_run = step - steps_already_done
                rate = done_this_run / elapsed if elapsed > 0 else float('nan')
                eta_min = (total_steps - step) / rate / 60 if rate > 0 else float('nan')
                print(f'max|xi| at {step}/{total_steps}:', float(np.abs(np.array(state.params["xi"])).max()),
                      f'  ({rate:.1f} steps/s, elapsed {elapsed/60:.1f} min, ETA {eta_min:.1f} min)')

            if do_threshold and i >= cfg.loss.THRESH_START and i % cfg.loss.THRESH_EVERY == 0:
                if cfg.loss.SPARSITY_METHOD == 'sr3':
                    key, sr3_key = jax.random.split(key)
                    new_Xi, new_support = sr3_refit(model, state.params, cfg, training_data, rng, sr3_key)
                    mask = mask * new_support  # monotone: support may only shrink, never resurrect
                    new_params = dict(state.params)
                    new_params['xi'] = mask * new_Xi
                    new_opt_state = _reset_xi_adam_moments(state.opt_state)
                    state = state.replace(params=new_params, opt_state=new_opt_state)
                    print(f'SR3 refit @ {step}: active {int(np.array(mask).sum())}/{mask.size}')
                else:
                    # Flat threshold, matching Champion et al.'s actual pipeline (no degree
                    # correction) -- see the docstring above _threshold_scale for why the
                    # theoretically-derived correction isn't safe to apply here: it assumes
                    # z_enc sits at the same scale as the true (1/40-normalized) state, which
                    # nothing in training actually guarantees, especially without the
                    # variance gauge-fix (LAMBDA_VAR) pinning it. Applying it caused real
                    # terms to be pruned when the encoder's actual latent scale drifted from
                    # that assumption -- observed: down to 3/60 active terms, too sparse.
                    xi_np = np.abs(np.array(state.params['xi']))
                    mask = mask * jnp.array((xi_np >= cfg.loss.THRESHOLD).astype(np.float32))
                    print(f'prune @ {step}: active {int(np.array(mask).sum())}/{mask.size}')

            if checkpoint_path is not None and step % checkpoint_every == 0:
                save_checkpoint(checkpoint_path, state, mask, loss_hist)
                print(f'checkpoint saved @ {step} -> {checkpoint_path}')
                if step % log_every != 0:  # avoid printing the same diagnostics twice if the cadences coincide
                    _log_diagnostics(step, total_steps, aux, mask, t_start, steps_already_done,
                                      model, state.params, training_data, diag_rng)
        return state, mask

    train_step = make_train_step(cfg)
    main_start_i = min(steps_already_done, cfg.training.MAX_STEPS)
    if main_start_i < cfg.training.MAX_STEPS:
        state, mask = run_phase(state, mask, cfg.training.MAX_STEPS, train_step, do_threshold=True,
                                 step_offset=0, start_i=main_start_i)
    else:
        print(f'main phase already complete ({main_start_i}/{cfg.training.MAX_STEPS}), skipping to refinement')

    # ── refinement phase (Champion et al.): freeze mask, drop the sparsity loss ──
    if cfg.training.REFINEMENT_STEPS > 0:
        refinement_start_i = min(max(0, steps_already_done - cfg.training.MAX_STEPS), cfg.training.REFINEMENT_STEPS)
        if refinement_start_i < cfg.training.REFINEMENT_STEPS:
            print('REFINEMENT')
            refinement_cfg = dataclasses.replace(cfg, loss=dataclasses.replace(cfg.loss, LAMBDA_SP=0.0))
            refinement_train_step = make_train_step(refinement_cfg)
            state, mask = run_phase(
                state, mask, cfg.training.REFINEMENT_STEPS, refinement_train_step,
                do_threshold=False, step_offset=cfg.training.MAX_STEPS, start_i=refinement_start_i,
            )

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, state, mask, loss_hist)
        print(f'final checkpoint saved -> {checkpoint_path}')

    last = loss_hist[-1]
    n_active = int(np.array(mask).sum())
    print(f'final  loss {last["loss"]:.3e}  rec {last["loss_rec"]:.3e}  '
          f'dz {last["loss_dz"]:.3e}  dx {last["loss_dx"]:.3e}  sp {last["loss_sp"]:.3e}  '
          f'var {last["loss_var"]:.3e}  active {n_active}/{mask.size}')

    return state, mask, loss_hist


if __name__ == "__main__":
    # load configurations
    cfg = Config()

    # Data creation for lifted Lorenz63 System from Champion et al. (2019) "Data-driven discovery of coordinates and governing equations"
    # cached under experiments/lorenz63 so repeated runs reuse the same data instead of regenerating it
    lorenz63_dir = Path(__file__).resolve().parents[1] / "experiments" / "lorenz63"
    training_data = load_or_create_lorenz_data(
        lorenz63_dir / "training_data.npz", cfg.model.N_ICS, cfg.model.NOISE_STRENGTH,
        linear=cfg.model.LINEAR_OBS,
    )
    validation_data = load_or_create_lorenz_data(
        lorenz63_dir / "validation_data.npz", cfg.model.N_VAL, cfg.model.NOISE_STRENGTH,
        linear=cfg.model.LINEAR_OBS,
    )

    # Create a polynomial library for the SINDy model
    _ps_lib = ps.PolynomialLibrary(degree=cfg.model.POLY_ORDER, include_bias=True)
    _ps_lib.fit([np.zeros((2, cfg.model.LATENT_DIM))])
    N_FEATURES = _ps_lib.n_output_features_
    FEAT_NAMES = list(_ps_lib.get_feature_names_out([f'z{i}' for i in range(cfg.model.LATENT_DIM)]))
    print(f'Library: {N_FEATURES} features  {FEAT_NAMES}')

    # initialze model 
    model = SINDyAE(
        # encooder
        encoder=PoolingEncoder(
            latent_dim=cfg.model.LATENT_DIM,
            is_variational=False,
            pooling_fn=DeepSetPooling(mlp_dim=64, mlp_n_hidden_layers=2),
        ),
        # decoder
        decoder=make_decoder(cfg),

        n_features=N_FEATURES,
        latent_dim=cfg.model.LATENT_DIM,
        poly_order=cfg.model.POLY_ORDER,
    )

    key = jax.random.PRNGKey(cfg.model.SEED)
    key, init_key, subsample_key = jax.random.split(key, 3)
    rng = np.random.default_rng(cfg.model.SEED)
    if cfg.model.SUBSAMPLE_POINTS:
        u0, udot0, x0 = sample_batch_subsampled(
            training_data, cfg.training.BATCH_SIZE, cfg.model.N_SUB, rng, subsample_key,
        )
    else:
        u0, udot0, x0 = sample_batch(training_data, cfg.training.BATCH_SIZE, rng)

    mask = jnp.ones((N_FEATURES, cfg.model.LATENT_DIM))
    params = model.init(init_key, u0, udot0, x0, mask)['params']

    checkpoint_path = lorenz63_dir / "checkpoint.pkl"
    resume_from = load_checkpoint(checkpoint_path)
    if resume_from is not None:
        print(f'found existing checkpoint at {checkpoint_path}, resuming')

    state, mask, loss_hist = train(
        cfg, model, params, mask, training_data, rng,
        checkpoint_path=checkpoint_path,
        checkpoint_every=50_000,
        resume_from=resume_from,
        key=key,
    )

