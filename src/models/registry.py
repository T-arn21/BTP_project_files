"""Adapters accept (B, 12, T) and return (B, C) logits, never probabilities."""
from dataclasses import asdict

_MODELS = {}


def register_model(name, factory):
    if name == "mamba" or name in _MODELS:
        raise ValueError(f"Model already registered: {name}")
    _MODELS[name] = factory


def create_model(name, n_classes, config):
    if name == "mamba":
        from .mamba import MambaClassifier
        return MambaClassifier(n_leads=12, n_classes=n_classes, **asdict(config))
    if name not in _MODELS:
        raise ValueError(f"Unknown model {name!r}; register a lightweight adapter first")
    return _MODELS[name](n_classes=n_classes, config=config)
