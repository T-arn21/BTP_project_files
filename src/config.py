"""Reference values transcribed from training notebook cells 7 and 17."""
from dataclasses import dataclass, field


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

    def __post_init__(self):
        if self.sampling_rate not in (100, 500):
            raise ValueError("sampling_rate must be 100 or 500")
        if self.duration <= 0 or self.layout not in {"channels_first", "time_first"}:
            raise ValueError("Invalid duration or tensor layout")
        if self.unlabeled not in {"drop", "error"} or self.num_workers < 0:
            raise ValueError("Invalid unlabeled policy or worker count")


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
    patience: int = 15
    min_delta: float = 1e-4
    seed: int = 42
    device: str = "auto"
    output_root: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        if self.task not in {"superclass", "subclass"}:
            raise ValueError("task must be superclass or subclass")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size, patience must be positive")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
