import numpy as np
import scipy.io as sio
from pathlib import Path


def get_rd_data(mat_path, noise_strength=1e-6, random_split=True, seed=None):
    """
    Load Champion et al.'s lambda-omega reaction-diffusion dataset, produced by
    `rd_solver/reaction_diffusion.m` (run once in MATLAB -- see that script's header
    comment; it saves `reaction_diffusion.mat` directly into this experiment's folder).

    Only the `u` channel is used as the autoencoder's field -- `v`/`dv` are loaded but
    discarded, exactly matching Champion's own `get_rd_data` (`example_reactiondiffusion.py`),
    which never feeds `v` into the autoencoder either.

    Returns `(training_data, val_data, test_data)`, each a dict with:
      't'         : (n_samples,) time stamps
      'x'         : (n_samples, N) field values u(y,t), N = n_grid**2 = 10_000
      'dx'        : (n_samples, N) field time-derivatives u_t(y,t)
      'y_spatial' : (N, 2) shared (y1, y2) spatial grid, identical for every row

    matching the (x, dx, y_spatial) contract `sample_batch`/`sample_batch_subsampled`
    (`src/training.py`) already expect from the Lorenz experiment -- so both experiments
    can share the same batching code unmodified.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(
            f"{mat_path} not found -- run rd_solver/reaction_diffusion.m in MATLAB first "
            f"(it saves reaction_diffusion.mat directly into this folder)."
        )
    data = sio.loadmat(mat_path)
    rng = np.random.default_rng(seed)

    n_samples = data['t'].size
    n = data['x'].size          # grid points per axis (100)
    N = n * n                  # total field points (10_000)

    uf = data['uf'] + noise_strength * rng.standard_normal(data['uf'].shape)
    duf = data['duf'] + noise_strength * rng.standard_normal(data['duf'].shape)

    # Shared (y1, y2) grid. The flattening convention here must match uf.reshape((N, -1))
    # below: numpy's default meshgrid indexing ('xy') reproduces MATLAB's [X,Y]=meshgrid(x,y),
    # and both use the same default row-major (C-order) reshape, so flat index k = i*n+j
    # corresponds to (x[j], y[i]) consistently in both the field values and this coordinate array.
    x_1d = data['x'].ravel()
    y_1d = data['y'].ravel()
    Xg, Yg = np.meshgrid(x_1d, y_1d)
    y_spatial = np.stack([Xg.reshape(-1), Yg.reshape(-1)], axis=-1).astype(np.float32)

    if not random_split:
        # sequential blocks: first 80% train, next 10% val, last 10% test
        training_samples = np.arange(int(.8 * n_samples))
        val_samples = np.arange(int(.8 * n_samples), int(.9 * n_samples))
    else:
        # Champion's default: random 80/10 split of the first 90% of samples
        perm = rng.permutation(int(.9 * n_samples))
        training_samples = perm[:int(.8 * n_samples)]
        val_samples = perm[int(.8 * n_samples):]
    # last 10% of samples always strictly held out as test, in both cases -- matches
    # Champion's get_rd_data (tests temporal extrapolation, not interpolation)
    test_samples = np.arange(int(.9 * n_samples), n_samples)

    def _subset(idx):
        return {
            't': data['t'].ravel()[idx],
            'x': uf[:, :, idx].reshape((N, -1)).T.astype(np.float32),
            'dx': duf[:, :, idx].reshape((N, -1)).T.astype(np.float32),
            'y_spatial': y_spatial,
        }

    return _subset(training_samples), _subset(val_samples), _subset(test_samples)
