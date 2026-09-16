"""Geracao do arquivo de submissao usando os ids oficiais do Kaggle (dono: P1)."""

from __future__ import annotations

import pandas as pd
import xarray as xr


def build_submission(
    predictions: xr.DataArray,
    sample_submission_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Copia os ids oficiais e preenche as previsoes na ordem time/lat/lon."""
    ordered = predictions.transpose("time", "lat", "lon").values.reshape(-1)
    sample = pd.read_csv(sample_submission_path, usecols=["id", "tp_mm_day"])
    if len(ordered) != len(sample):
        raise ValueError(
            f"grade tem {len(ordered)} valores, mas sample_submission tem {len(sample)} ids"
        )
    df = sample[["id"]].copy()
    df["tp_mm_day"] = ordered
    df.to_csv(output_path, index=False)
    return df
