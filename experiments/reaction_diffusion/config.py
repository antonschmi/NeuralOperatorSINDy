from dataclasses import dataclass, field

@dataclass
class ExperimentalConfig: 
    POLY_ORDER: int = 2
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
    LINEAR_OBS: bool = True   # True -> linear Legendre observation; False -> + cubic terms
    

@dataclass
class TrainingConfig:
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 1e-3
    LR_TRANSITION_STEPS: int = 500
    LR_DECAY_RATE: float = 0.99
    MAX_STEPS: int = 10000
    REFINEMENT_STEPS: int = 1000  # Champion et al.'s refinement phase: mask frozen, sparsity loss dropped

@dataclass
class LossConfig:
    LAMBDA_REC: float = 1.0
    LAMBDA_DZ: float = 1.0
    LAMBDA_DX: float = 1e-4
    LAMBDA_SP: float = 1e-5
    LAMBDA_VAR: float = 1.0  # gauge fix: pins Var(z_k) ~= 1, breaks the latent scale symmetry
    THRESHOLD: float = 0.1
    THRESH_START: int = 2000
    THRESH_EVERY: int = 500

@dataclass
class Config:
    model: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
