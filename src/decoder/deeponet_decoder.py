import jax.numpy as jnp
from typing import Sequence
from dataclasses import field
from src.decoder import Decoder
from src.positional_encodings import (
    PositionalEncoding,
    IdentityEncoding,
)
from src.utils.networks import MLP


class DeepONetDecoder(Decoder):
    r"""
    A full (branch + trunk) DeepONet, unlike the other two decoders in this package:

    - `LinearDecoder` has a trunk net (nonlinear in `x`) but no branch net at all --
      `z` itself serves as the branch coefficients directly, so the overall decoder is
      exactly affine in `z`.
    - `NonlinearDecoder` has no separation at all -- `z` and `x` are concatenated and
      jointly processed by one entangled MLP, which lets the optimizer explain
      reconstruction error via `x` (which varies per query point) while barely
      depending on `z` (which is broadcast identically to every query point within a
      sample), since nothing forces it to do otherwise.

    This decoder keeps a genuine branch net `B(z)` (sees only `z`, never `x`) and a
    trunk net `T(x)` (sees only `x`, never `z`, identical in spirit to `LinearDecoder`'s
    trunk), combined via the same linear contraction `LinearDecoder` uses:

        output(z, x) = sum_k B(z)_k * T(x)_k

    Because the branch net never sees `x`, it has nowhere else to route information
    except through `z` -- removing `NonlinearDecoder`'s specific escape hatch. Because
    `B` is a genuinely nonlinear function of `z` (unlike `LinearDecoder`'s implicit
    identity branch), the overall map is no longer restricted to being affine in `z`,
    which matters whenever the true observation depends on both `z` and some nonlinear
    function of it (e.g. `z_i` and `z_i^3` as independent directions) with independent
    coefficients -- something an exactly-affine decoder cannot represent regardless of
    how expressive its trunk is.

    This does *not* restore `LinearDecoder`'s free, automatic "value-accuracy implies
    derivative-accuracy" guarantee (`B`'s Jacobian w.r.t. `z` is genuinely z-dependent,
    just like `NonlinearDecoder`'s) -- it puts this decoder in the same general
    category as Champion et al.'s original (also nonlinear, also branch-only-sees-its-
    own-input) decoder: no automatic guarantee, but no competing input to hide behind
    either.

    `n_basis` is decoupled from `latent_dim` here, unlike `LinearDecoder` (where
    `n_basis` is conventionally set equal to `latent_dim`, since `z` serves as the
    branch output directly) -- the branch net can map a `latent_dim`-dimensional `z` to
    any number of basis coefficients, giving room to represent non-affine dependencies
    on `z` without inflating the SINDy latent dimension (and therefore the SINDy
    library size) itself.

    Inputs:

    `z` : [batch, latent_dim]
        the SINDy latent state (encoder output), fed only to the branch net

    `x` : [batch, n_evals, in_dim]
        tensor of query points, fed only to the trunk net
    """

    out_dim: int
    n_basis: int = 64
    branch_features: Sequence[int] = (64, 64)
    trunk_features: Sequence[int] = (128, 128, 128)
    positional_encoding: PositionalEncoding = IdentityEncoding()
    branch_mlp_args: dict = field(default_factory=dict)
    trunk_mlp_args: dict = field(default_factory=dict)

    def setup(self):
        self.branch_net = MLP([*self.branch_features, self.n_basis], **self.branch_mlp_args)
        self.trunk_net = MLP([*self.trunk_features, self.n_basis * self.out_dim], **self.trunk_mlp_args)

    def _forward(self, z, x, train=False):
        b = self.branch(z)
        t = self.trunk(x)
        return jnp.einsum("ij,...ikjl->ikl", b, t)

    def branch(self, z):
        return self.branch_net(z)

    def trunk(self, x):
        x = self.positional_encoding(x)
        t = self.trunk_net(x)
        return jnp.reshape(t, (x.shape[0], x.shape[1], self.n_basis, self.out_dim))
