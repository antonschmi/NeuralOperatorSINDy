import dataclasses
import pysindy as ps
import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from itertools import combinations_with_replacement
from pathlib import Path


from src.encoder.pooling_encoder import PoolingEncoder
from src.decoder.linear_decoder import LinearDecoder
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


def sample_batch(data, batch_size, rng):
    """
    Draw a random batch of (u, u_dot, x) from a get_lorenz_data-style dict, in place
    of a torch DataLoader. `data['x']`/`data['dx']` hold one row per (ic, timestep)
    snapshot, each defined on the shared spatial grid `data['y_spatial']`.
    """
    idx = rng.choice(data['x'].shape[0], size=batch_size, replace=False)
    u = jnp.asarray(data['x'][idx])[..., None]        # [B, n_points, 1]
    u_dot = jnp.asarray(data['dx'][idx])[..., None]    # [B, n_points, 1]
    x = jnp.broadcast_to(
        jnp.asarray(data['y_spatial'])[None, :, None],
        (batch_size, data['y_spatial'].shape[0], 1),
    )
    return u, u_dot, x


def theta_jax(z, poly_order, latent_dim):
    """
    JAX polynomial library matching ps.PolynomialLibrary(degree=poly_order, include_bias=True).

    For clean JAX derivative computiation.

    """
    B = z.shape[0]
    cols = [jnp.ones(B)]
    for deg in range(1, poly_order + 1):
        for idx in combinations_with_replacement(range(latent_dim), deg):
            feat = jnp.ones(B)
            for k in idx:
                feat = feat * z[:, k]
            cols.append(feat)
    return jnp.stack(cols, axis=-1)

class SINDyAE(nn.Module):
    """
        Autoencoder with SINDy latent dynamics.

    """

    encoder:    nn.Module
    decoder:    nn.Module
    n_features: int            # p = theta library size
    latent_dim: int
    poly_order: int

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
        dz_sindy = theta_jax(z, self.poly_order, self.latent_dim) @ (mask * xi)

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


def train(cfg, model, params, mask, training_data, rng):
    """
    Train `model` with hard sequential thresholding of the SINDy coefficients `xi`
    (STLSQ-style: |xi| < cfg.loss.THRESHOLD -> 0), drawing random batches straight
    from `training_data` each step, for a fixed cfg.training.MAX_STEPS gradient steps
    (no epoch bookkeeping - each step is an independent random batch). Then runs a
    refinement phase (Champion et al.'s `refinement_epochs`): continues training with
    `mask` frozen and the sparsity loss dropped entirely, letting the surviving
    coefficients settle to their best-fit values without the L1 term's residual
    shrinkage bias.

    Returns the final TrainState, the final (pruned) mask, and a history of the
    per-step loss terms.
    """
    tx = optax.adam(learning_rate=optax.exponential_decay(
        cfg.training.LEARNING_RATE,
        transition_steps=cfg.training.LR_TRANSITION_STEPS,
        decay_rate=cfg.training.LR_DECAY_RATE,
    ))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    loss_hist = []

    def run_phase(state, mask, n_steps, train_step, do_threshold, step_offset):
        for i in range(1, n_steps + 1):
            step = step_offset + i
            u_t, u_dot, x = sample_batch(training_data, cfg.training.BATCH_SIZE, rng)
            state, total, aux = train_step(state, u_t, u_dot, x, mask)
            loss_hist.append({k: float(v) for k, v in aux.items()})

            if i % 1000 == 0 or i == 1:
                print(f'max|xi| at {step}:', float(np.abs(np.array(state.params['xi'])).max()))

            if do_threshold and i >= cfg.loss.THRESH_START and i % cfg.loss.THRESH_EVERY == 0:
                xi_np = np.abs(np.array(state.params['xi']))
                mask = mask * jnp.array((xi_np >= cfg.loss.THRESHOLD).astype(np.float32))
                print(f'prune @ {step}: active {int(np.array(mask).sum())}/{mask.size}')
        return state, mask

    train_step = make_train_step(cfg)
    state, mask = run_phase(state, mask, cfg.training.MAX_STEPS, train_step, do_threshold=True, step_offset=0)

    # ── refinement phase (Champion et al.): freeze mask, drop the sparsity loss ──
    if cfg.training.REFINEMENT_STEPS > 0:
        print('REFINEMENT')
        refinement_cfg = dataclasses.replace(cfg, loss=dataclasses.replace(cfg.loss, LAMBDA_SP=0.0))
        refinement_train_step = make_train_step(refinement_cfg)
        state, mask = run_phase(
            state, mask, cfg.training.REFINEMENT_STEPS, refinement_train_step,
            do_threshold=False, step_offset=cfg.training.MAX_STEPS,
        )

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
        decoder=LinearDecoder(
            out_dim=1,
            n_basis=cfg.model.LATENT_DIM,  # z's basis coefficients are the SINDy latent states themselves
            features=(64, 64, 64)
        ),

        n_features=N_FEATURES,
        latent_dim=cfg.model.LATENT_DIM,
        poly_order=cfg.model.POLY_ORDER,
    )

    key = jax.random.PRNGKey(cfg.model.SEED)
    key, init_key = jax.random.split(key)
    rng = np.random.default_rng(cfg.model.SEED)
    u0, udot0, x0 = sample_batch(training_data, cfg.training.BATCH_SIZE, rng)

    mask = jnp.ones((N_FEATURES, cfg.model.LATENT_DIM))
    params = model.init(init_key, u0, udot0, x0, mask)['params']

    state, mask, loss_hist = train(cfg, model, params, mask, training_data, rng)

