from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.config import DataConfig


@pytest.fixture
def official_statements():
    return pd.read_csv(Path(__file__).parent / "fixtures" / "scp_statements.csv", index_col=0)


@pytest.fixture
def synthetic_root(tmp_path, official_statements):
    official_statements.to_csv(tmp_path / "scp_statements.csv")
    rows = []
    for i in range(36):
        rows.append(dict(ecg_id=i + 1, patient_id=i + 1,
                         strat_fold=(i % 8) + 1 if i < 24 else 9 if i < 30 else 10,
                         scp_codes=str({"NORM": 100} if i % 2 else {"IMI": 100, "CRBBB": 0}),
                         filename_lr=f"record_{i + 1}", filename_hr=f"record_{i + 1}"))
    pd.DataFrame(rows).set_index("ecg_id").to_csv(tmp_path / "ptbxl_database.csv")
    return tmp_path


def synthetic_reader(path, sampling_rate):
    number = int(Path(path).name.rsplit("_", 1)[-1])
    rng = np.random.default_rng(number)
    t = np.arange(sampling_rate * 10) / sampling_rate
    return (np.sin(2 * np.pi * 1.2 * t)[None, :] * np.linspace(0.5, 2, 12)[:, None]
            + rng.normal(0, 0.05, (12, len(t))) + number / 100)


@pytest.fixture
def data_config(synthetic_root):
    return DataConfig(data_dir=str(synthetic_root), augment=False, cache_dir=str(synthetic_root / "cache"))
