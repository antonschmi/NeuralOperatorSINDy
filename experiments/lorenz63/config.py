from dataclasses import dataclass, field

@dataclass
class ExperimentalConfig: 
    POLY_ORDER: int = 3
    INPUT_DIM: int = 128
    LATENT_DIM: int = 3
    DT: float = 0.02
    N_SAMPLES: int = 250
    ALPHA      = 0.1    
    BETA       = 8 / 3
    RHO        = 28.
    NOISE_STRENGTH: float = 1e-6
    N_ICS: int = 2048
    N_VAL: int = 20
    SEED: int = 42
    LINEAR_OBS: bool = False  # True -> linear Legendre observation; False -> + cubic terms
    INCLUDE_SINE: bool = False  # Lorenz's SINDy library needs no sine terms; reaction-diffusion's does
    DECODER: str = "deeponet"  # "linear" (DeepONet-style, linear in z, no branch net) | "nonlinear" (MLP over
                                # concat(z, x), z and x entangled) | "deeponet" (full branch(z)+trunk(x) DeepONet)
    SUBSAMPLE_POINTS: bool = False # if True, each batch row gets its own random n_sub-point subset of the grid (mesh-invariance test) instead of the fixed full grid
    N_SUB: int = 32       # points per row when SUBSAMPLE_POINTS is True; ignored otherwise
    ENCODER_FEATURES: list = field(default_factory=lambda: [64, 64])  # DeepSetPooling per-point MLP layer widths


@dataclass
class TrainingConfig:
    BATCH_SIZE: int = 8000
    LEARNING_RATE: float = 1e-3
    LR_TRANSITION_STEPS: int = 62500  # rescaled x125 to preserve the ~20-period mild decay shape over the new step budget
    LR_DECAY_RATE: float = 0.99
    MAX_STEPS: int = 1_250_250        # 5001 epochs * 250 steps/epoch
    REFINEMENT_STEPS: int = 250_250   # 1001 epochs * 250 steps/epoch -- Champion et al.'s refinement phase: mask frozen, sparsity loss dropped


@dataclass
class LossConfig:
    LAMBDA_REC: float = 1.0
    LAMBDA_DZ: float = 0.0
    LAMBDA_DX: float = 1e-2
    LAMBDA_SP: float = 1e-3
    LAMBDA_VAR: float = 1.0 # gauge fix: pins Var(z_k) ~= 1, breaks the latent scale symmetry
    THRESHOLD: float = 0.1
    THRESH_START: int = 32_000  # = Champion's threshold_frequency=500 epochs * 250 steps/epoch (first prune fires here, not before)
    THRESH_EVERY: int = 32_000  # = Champion's threshold_frequency=500 epochs * 250 steps/epoch
    
    SPARSITY_METHOD: str = 'sr3'  # 'relative_threshold' (default, existing behavior) | 'sr3'
    SR3_LAM: float = 0.1       # pysindy SR3 reg_weight_lam (L0 regularization weight)
    SR3_NU: float = 1.0          # pysindy SR3 relax_coeff_nu
    SR3_N_SAMPLES: int = 20_000  # fixed subsample size (rows of training_data) used for each SR3 solve

@dataclass
class Config:
    model: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
