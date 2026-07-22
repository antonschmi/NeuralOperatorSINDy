"""
Training entry point for the reaction-diffusion experiment.

Mirrors `src/training.py`'s `__main__` block (same SINDyAE / train() / checkpointing
machinery), but points at this experiment's config/data instead of Lorenz's, and sizes
the encoder/decoder per Champion et al.'s Table S2 (hidden width 256, since the RD
field is 10_000-dimensional vs Lorenz's 128) rather than reusing `make_decoder`'s
Lorenz-sized (64,64,64) hardcoded widths.

Prerequisite: run `rd_solver/reaction_diffusion.m` once in MATLAB -- it saves
`reaction_diffusion.mat` directly into this folder. This script will raise
`FileNotFoundError` (via `get_rd_data`) until that file exists.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `src.*` imports

from src.encoder.pooling_encoder import PoolingEncoder
from src.decoder.linear_decoder import LinearDecoder
from src.decoder.non_linear_decoder import NonlinearDecoder
from src.decoder.deeponet_decoder import DeepONetDecoder
from src.utils.networks.pooling import DeepSetPooling
from src.training import (
    SINDyAE, train, build_feature_library, load_checkpoint,
    sample_batch, sample_batch_subsampled,
)
from experiments.reaction_diffusion.config import Config
from experiments.reaction_diffusion.training_data import get_rd_data

RD_HIDDEN_WIDTH = 256  # Champion's Table S2 decoder width for reaction-diffusion (encoder
# width/depth now comes from cfg.model.ENCODER_FEATURES instead -- see DeepSetPooling)


def make_rd_decoder(cfg):
    if cfg.model.DECODER == "linear":
        return LinearDecoder(out_dim=1, n_basis=cfg.model.LATENT_DIM, features=(RD_HIDDEN_WIDTH,))
    elif cfg.model.DECODER == "nonlinear":
        return NonlinearDecoder(out_dim=1, features=(RD_HIDDEN_WIDTH,))
    elif cfg.model.DECODER == "deeponet":
        return DeepONetDecoder(
            out_dim=1,
            n_basis=20,  # decoupled from LATENT_DIM, same rationale as make_decoder (src/training.py)
            branch_features=(RD_HIDDEN_WIDTH, RD_HIDDEN_WIDTH),
            trunk_features=(RD_HIDDEN_WIDTH,),
        )
    else:
        raise ValueError(
            f"Unknown cfg.model.DECODER: {cfg.model.DECODER!r} (expected 'linear', 'nonlinear', or 'deeponet')"
        )


if __name__ == "__main__":
    cfg = Config()
    rd_dir = Path(__file__).resolve().parent

    training_data, validation_data, test_data = get_rd_data(
        rd_dir / cfg.model.MAT_PATH,
        noise_strength=cfg.model.NOISE_STRENGTH,
        random_split=cfg.model.RANDOM_SPLIT,
        seed=cfg.model.SEED,
    )
    print(f"loaded {training_data['x'].shape[0]} train / {validation_data['x'].shape[0]} val / "
          f"{test_data['x'].shape[0]} test samples, N={training_data['x'].shape[1]} field points")

    lib = build_feature_library(cfg.model.POLY_ORDER, cfg.model.LATENT_DIM, cfg.model.INCLUDE_SINE)
    N_FEATURES = lib.n_output_features_
    FEAT_NAMES = list(lib.get_feature_names([f'z{i}' for i in range(cfg.model.LATENT_DIM)]))
    print(f'Library: {N_FEATURES} features  {FEAT_NAMES}')

    model = SINDyAE(
        encoder=PoolingEncoder(
            latent_dim=cfg.model.LATENT_DIM,
            is_variational=False,
            pooling_fn=DeepSetPooling(features=cfg.model.ENCODER_FEATURES),
        ),
        decoder=make_rd_decoder(cfg),
        n_features=N_FEATURES,
        latent_dim=cfg.model.LATENT_DIM,
        poly_order=cfg.model.POLY_ORDER,
        include_sine=cfg.model.INCLUDE_SINE,
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

    checkpoint_path = rd_dir / "checkpoint.pkl"
    resume_from = load_checkpoint(checkpoint_path)
    if resume_from is not None:
        print(f'found existing checkpoint at {checkpoint_path}, resuming')

    state, mask, loss_hist = train(
        cfg, model, params, mask, training_data, rng,
        checkpoint_path=checkpoint_path,
        checkpoint_every=5_000,
        resume_from=resume_from,
        key=key,
    )
