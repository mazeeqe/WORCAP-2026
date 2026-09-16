import numpy as np
import pandas as pd
import xarray as xr

from src.data import ALL_VARS, FEATURE_VARS, build_examples
from src.submit import build_submission


def test_training_uses_previous_month_atmosphere() -> None:
    series = {
        name: np.arange(8, dtype=np.float32).reshape(-1, 1)
        for name in ALL_VARS
    }

    _, atmosphere, _, _, target, _, target_index = build_examples(
        series,
        origin_start_idx=2,
        origin_end_idx=2,
        last_valid_idx=7,
        hindcast_len=2,
        lags=range(1, 2),
    )

    assert target_index.tolist() == [3]
    assert atmosphere.shape == (1, len(FEATURE_VARS))
    assert np.all(atmosphere == 2)
    assert target.item() == 3


def test_submission_preserves_official_ids(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    output = tmp_path / "submission.csv"
    official_ids = ["id-c", "id-a", "id-d", "id-b"]
    pd.DataFrame({"id": official_ids, "tp_mm_day": 0.0}).to_csv(sample, index=False)
    predictions = xr.DataArray(
        np.array([[[1.0, 2.0], [3.0, 4.0]]]),
        dims=("time", "lat", "lon"),
    )

    result = build_submission(predictions, str(sample), str(output))

    assert result["id"].tolist() == official_ids
    assert result["tp_mm_day"].tolist() == [1.0, 2.0, 3.0, 4.0]
