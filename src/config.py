"""Reference values transcribed from training notebook cells 7 and 17."""
from dataclasses import dataclass, field
from dataclasses import asdict
import math


@dataclass
class DataConfig:
    data_dir: str = ""  # or PTBXL_DATA_DIR
    sampling_rate: int = 100
    duration: int = 10
    layout: str = "channels_first"  # or time_first
    apply_filter: bool = True
    augment: bool = True
    unlabeled: str = "drop"  # or error; never treat unknown diagnoses as negatives
    num_workers: int = 0
    dataset_version: str = "unknown"  # operator-declared; never inferred as verified
    normalization_mode: str = "sampled"  # sampled or full, both training-only
    normalization_samples: int = 1000
    normalization_seed: int = 42
    cache_dir: str = "outputs/ptbxl_cache"
    cache_signals: bool = True

    def __post_init__(self):
        if self.sampling_rate not in (100, 500):
            raise ValueError("sampling_rate must be 100 or 500")
        if self.duration <= 0 or self.layout not in {"channels_first", "time_first"}:
            raise ValueError("Invalid duration or tensor layout")
        if self.unlabeled not in {"drop", "error"} or self.num_workers < 0:
            raise ValueError("Invalid unlabeled policy or worker count")
        if self.normalization_mode not in {"sampled", "full"} or self.normalization_samples < 1:
            raise ValueError("Invalid normalization mode or sample count")


@dataclass
class ModelConfig:
    embed_dim: int = 384
    hidden_dim: int = 384
    n_layers: int = 8
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.45
    bidirectional: bool = True
    pooling: str = "mean_attention"


@dataclass
class ExperimentConfig:
    model_name: str = "mamba"
    task: str = "superclass"
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 5e-4
    warmup_epochs: int = 3
    min_lr: float = 1e-6
    label_smoothing: float = 0.05
    use_pos_weight: bool = True
    pos_weight_max: float = 8.0
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True
    numerical_stability_mode: str = "strict"  # conservative also requires FP32 / lr <= 1e-4
    initial_checkpoint: str | None = None  # explicit weights-only start; not exact optimizer resume
    patience: int = 15
    min_delta: float = 1e-4
    seed: int = 42
    device: str = "auto"
    output_root: str = "outputs"
    diagnostic: bool = False
    diagnostic_batches: int = 2
    log_every_batches: int = 20
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        if self.task not in {"superclass", "subclass"}:
            raise ValueError("task must be superclass or subclass")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size, patience must be positive")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.diagnostic_batches < 1 or self.log_every_batches < 1:
            raise ValueError("diagnostic_batches and log_every_batches must be positive")
        if self.numerical_stability_mode not in {"strict", "conservative"}:
            raise ValueError("numerical_stability_mode must be strict or conservative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be finite and positive")
        if not math.isfinite(self.pos_weight_max) or self.pos_weight_max <= 0:
            raise ValueError("pos_weight_max must be finite and positive")
        if self.numerical_stability_mode == "conservative" and (self.mixed_precision or self.learning_rate > 1e-4):
            raise ValueError("Conservative mode requires mixed_precision=False and learning_rate <= 1e-4; settings are never silently changed")


def checkpoint_config_matches(saved, current):
    """Add only new-field legacy defaults; do not relax training/data comparisons."""
    saved = dict(saved)
    saved.setdefault("numerical_stability_mode", "strict")
    saved.setdefault("initial_checkpoint", None)
    return saved == (asdict(current) if isinstance(current, ExperimentConfig) else current)
