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
    N_ICS: int = 1024
    N_VAL: int = 20
    SEED: int = 42
    LINEAR_OBS: bool = True   # True -> linear Legendre observation; False -> + cubic terms
    DECODER: str = "linear"  # "linear" (DeepONet-style, linear in z) | "nonlinear" (MLP over concat(z, x))


@dataclass
class TrainingConfig:
    BATCH_SIZE: int = 1024
    LEARNING_RATE: float = 1e-3
    LR_TRANSITION_STEPS: int = 62500  # rescaled x125 to preserve the ~20-period mild decay shape over the new step budget
    LR_DECAY_RATE: float = 0.99
    MAX_STEPS: int = 1_250_250        # 5001 epochs * 250 steps/epoch
    REFINEMENT_STEPS: int = 250_250   # 1001 epochs * 250 steps/epoch -- Champion et al.'s refinement phase: mask frozen, sparsity loss dropped


@dataclass
class LossConfig:
    LAMBDA_REC: float = 1.0
    LAMBDA_DZ: float = 1.0
    LAMBDA_DX: float = 1e-4
    LAMBDA_SP: float = 1e-5
    LAMBDA_VAR: float = 1.0  # gauge fix: pins Var(z_k) ~= 1, breaks the latent scale symmetry
    THRESHOLD: float = 0.1
    THRESH_START: int = 125_000  # = Champion's threshold_frequency=500 epochs * 250 steps/epoch (first prune fires here, not before)
    THRESH_EVERY: int = 125_000  # = Champion's threshold_frequency=500 epochs * 250 steps/epoch

@dataclass
class Config:
    model: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
