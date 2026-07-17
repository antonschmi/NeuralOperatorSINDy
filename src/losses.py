import jax.numpy as jnp


def sindy_ae_loss(cfg, model, batch, mask):
    """
    Combine the SINDy-autoencoder loss terms (Champion et al. 2019):
      - reconstruction:      u_hat matches the input field u_t
      - latent consistency:  the encoder-side dz/dt (dz_enc) matches the SINDy-predicted dz/dt
      - decoder consistency: the decoder-side du/dt (du_dec, driven by the SINDy latent
                              dynamics) matches the true du/dt (u_dot)
      - sparsity:            L1 regularization on the (masked) SINDy coefficients xi
      - gauge fix:           pins Cov(z) ~= I across the batch. The diagonal breaks the latent
                              scale symmetry (z -> c*z, xi -> xi/c leaves rec/dz/dx unchanged, so
                              without it the coefficients can be pushed toward zero purely by
                              shrinking z, which prunes terms that are actually needed); the
                              off-diagonal penalizes correlated latent channels, which otherwise
                              can collapse into redundant copies of the same direction (each
                              individually satisfying Var(z_k)=1) without the SINDy library ever
                              seeing an actually 3-dimensional latent state.

    Each term is only computed if its `cfg.loss.LAMBDA_*` weight is nonzero -- a weight of
    0.0 means "not used" (e.g. lorenz63's LAMBDA_DZ/LAMBDA_VAR right now), so the term
    contributes nothing to `total` either way, but computing it regardless would waste
    work for no observable effect (loss_var's Cov(z) matmul is the one real cost here;
    the others are cheap, but the same guard is applied uniformly for consistency and so
    aux's reported value for a disabled term reads as an explicit 0.0, not a stray
    nonzero number an unfamiliar reader might mistake for something being enforced).
    `cfg.loss.LAMBDA_*` are plain Python floats on `cfg` (not JAX tracers), so this `if`
    is resolved at trace time -- a disabled term's branch never even gets compiled into
    the jitted train_step, not merely skipped at runtime.
    """
    u_t, u_dot, x = batch
    u_hat, z, dz_enc, dz_sindy, du_dec, xi = model(u_t, u_dot, x, mask)

    zero = jnp.asarray(0.0)

    loss_rec = jnp.mean((u_hat - u_t) ** 2) if cfg.loss.LAMBDA_REC > 0 else zero
    loss_dz = jnp.mean((dz_enc - dz_sindy) ** 2) if cfg.loss.LAMBDA_DZ > 0 else zero
    loss_dx = jnp.mean((du_dec - u_dot) ** 2) if cfg.loss.LAMBDA_DX > 0 else zero
    loss_sp = jnp.mean(jnp.abs(mask * xi)) if cfg.loss.LAMBDA_SP > 0 else zero

    if cfg.loss.LAMBDA_VAR > 0:
        z_centered = z - jnp.mean(z, axis=0, keepdims=True)
        cov = (z_centered.T @ z_centered) / z.shape[0]
        loss_var = jnp.mean((cov - jnp.eye(cov.shape[0])) ** 2)
    else:
        loss_var = zero

    total = (
        cfg.loss.LAMBDA_REC * loss_rec
        + cfg.loss.LAMBDA_DZ * loss_dz
        + cfg.loss.LAMBDA_DX * loss_dx
        + cfg.loss.LAMBDA_SP * loss_sp
        + cfg.loss.LAMBDA_VAR * loss_var
    )

    return total, {
        "loss": total,
        "loss_rec": loss_rec,
        "loss_dz": loss_dz,
        "loss_dx": loss_dx,
        "loss_sp": loss_sp,
        "loss_var": loss_var,
        "z": z,  # per-batch latent codes -- for diagnostic logging only, never persisted to loss_hist
    }
