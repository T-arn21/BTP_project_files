"""Small, standard-library-only selectors for the existing unattended worker.

The JSON snapshot retains the unified runner's settings. Profiles select stages;
they do not implement models, data processing, provenance exceptions or training.
"""
from copy import deepcopy
import json
from pathlib import Path


PROFILES = ("supervised_baseline", "ssl_pretraining", "ssl_finetuning",
            "ssl_extended_finetuning", "asl_finetuning")


def workflow_settings(request):
    profile = request["WORKFLOW_PROFILE"]
    mode = request["WORKFLOW_MODE"]
    if profile not in PROFILES or mode not in ("smoke", "full"):
        raise ValueError("Unknown workflow or mode; choose a documented profile and smoke/full")
    if profile in PROFILES[3:] and mode != "full":
        raise ValueError("Extension and ASL profiles have no separate smoke mode; execution is opt-in")
    settings = json.loads((Path(__file__).resolve().parents[1] / "configs/kaggle_defaults.json").read_text())
    for key in settings:
        if key.startswith("RUN_"):
            settings[key] = False
    settings.update(CONTINUATION_SESSION_ID="", INITIAL_CHECKPOINTS={"superclass": "", "subclass": ""},
        RESUME_RUN_DIRS={"superclass": "", "subclass": ""}, SSL_FINETUNE_PROVENANCE_ONLY=False,
        DIAGNOSTIC_MODE=False)
    settings["CACHE_DIR"] = str(Path(request["OUTPUT_ROOT"]) / "cache")
    overrides = deepcopy(request.get("WORKFLOW_OVERRIDES", {}))
    forbidden = {key for key in overrides if key not in settings or key.startswith("RUN_") or
        key in {"CHECKOUT_DIR", "GIT_BRANCH", "GIT_TIMEOUT_SECONDS", "OUTPUT_ROOT", "SSL_FINETUNE_PROVENANCE_ONLY", "DIAGNOSTIC_MODE", "CONTINUATION_SESSION_ID", "INITIAL_CHECKPOINTS", "RESUME_RUN_DIRS"}}
    if forbidden:
        raise ValueError(f"Unknown or competing workflow overrides: {sorted(forbidden)}")
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(settings[key], dict):
            settings[key].update(value)
        else:
            settings[key] = value
    for key in ("CHECKOUT_DIR", "GIT_BRANCH", "GIT_TIMEOUT_SECONDS", "OUTPUT_ROOT"):
        settings[key] = request[key]
    if profile == "supervised_baseline":
        settings.update(RUN_SMOKE=True, RUN_METADATA_AUDIT=True,
                        RUN_SUPERCLASS=mode == "full", RUN_SUBCLASS=mode == "full")
    elif profile == "ssl_pretraining":
        settings.update(RUN_SSL_SMOKE=True, RUN_SSL_PRETRAIN=mode == "full")
    elif profile == "ssl_finetuning":
        settings.update(RUN_SSL_FINETUNE_SMOKE=True, RUN_SSL_FINETUNE_SUPERCLASS=mode == "full",
                        RUN_SSL_FINETUNE_SUBCLASS=mode == "full")
    elif profile == "ssl_extended_finetuning":
        settings["RUN_SSL_FINETUNE_RESUME"] = True
    else:
        settings.update(RUN_ASL_SUPERCLASS=True, RUN_ASL_SUBCLASS=True)
    settings.update(WORKFLOW_PROFILE=profile, WORKFLOW_MODE=mode)
    return settings
