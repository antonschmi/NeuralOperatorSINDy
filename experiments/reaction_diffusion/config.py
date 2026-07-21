from dataclasses import dataclass, field

@dataclass
class ExperimentalConfig:
    POLY_ORDER: int = 3
    LATENT_DIM: int = 2
    INCLUDE_SINE: bool = True   # Champion's RD SINDy library needs sin(z1), sin(z2) -- unlike Lorenz
    N_GRID: int = 100          # spatial grid points per axis -> N_GRID**2 = 10_000 field points, matching Champion
    NOISE_STRENGTH: float = 1e-6  # matches Champion's added Gaussian noise on uf/duf
    SEED: int = 42
    DECODER: str = "deeponet"    # "linear" (DeepONet-style, linear in z, no branch net) | "nonlinear" (MLP over
                               # concat(z, x), z and x entangled) | "deeponet" (full branch(z)+trunk(x) DeepONet)
    SUBSAMPLE_POINTS: bool = True  # mesh-invariance test -- no Champion equivalent, this project's own extension
    N_SUB: int = 3000          # points per row when SUBSAMPLE_POINTS is True (10% of the 10_000-point grid; tune freely)
    MAT_PATH: str = "reaction_diffusion.mat"  # produced by rd_solver/reaction_diffusion.m (run once in MATLAB)
    RANDOM_SPLIT: bool = True  # True: Champion's random 80/10 split of the first 90% of samples (get_rd_data(random=True));
                               # False: sequential blocks. Last 10% of samples always held out as test either way.
    LINEAR_OBS: bool = False  # True -> linear Legendre observation; False -> + cubic terms
    INCLUDE_SINE: bool = True  # Lorenz's SINDy library needs no sine terms; reaction-diffusion's does


@dataclass
class TrainingConfig:
    BATCH_SIZE: int = 1000     # matches Champion's Table S2
    LEARNING_RATE: float = 1e-3
    LR_TRANSITION_STEPS: int = 500
    LR_DECAY_RATE: float = 0.99
    MAX_STEPS: int = 24_008        # 3001 epochs * 8 steps/epoch (8000 training samples // batch_size=1000), matches Champion
    REFINEMENT_STEPS: int = 8_008  # 1001 epochs * 8 steps/epoch -- Champion et al.'s refinement phase: mask frozen, sparsity loss dropped


@dataclass
class LossConfig:
    LAMBDA_REC: float = 1.0
    LAMBDA_DZ: float = 0.01   # Champion's loss_weight_sindy_z for reaction-diffusion (NOT Lorenz's weight)
    LAMBDA_DX: float = 0.5    # Champion's loss_weight_sindy_x for reaction-diffusion
    LAMBDA_SP: float = 0.1    # Champion's loss_weight_sindy_regularization for reaction-diffusion
    LAMBDA_VAR: float = 0.1   # gauge fix: pins Cov(z) ~= I -- no Champion equivalent, kept on as in lorenz63
    THRESHOLD: float = 0.1
    THRESH_START: int = 4_000  # = Champion's threshold_frequency=500 epochs * 8 steps/epoch (first prune fires here)
    THRESH_EVERY: int = 4_000

    SPARSITY_METHOD: str = 'sr3'  # 'relative_threshold' (default, existing behavior) | 'sr3'
    SR3_LAM: float = 0.05        # pysindy SR3 reg_weight_lam (L0 regularization weight)
    SR3_NU: float = 1.0          # pysindy SR3 relax_coeff_nu
    SR3_N_SAMPLES: int = 8_000   # fixed subsample size (rows of training_data) used for each SR3 solve

@dataclass
class Config:
    model: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
