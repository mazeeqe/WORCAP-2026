"""Retreina o esquema escolhido pela CV (cv_ensemble.py --report) com TODO o historico
rotulado (1940-2022) e gera a submissao: ensemble de seeds do LSTM (opcionalmente
empilhado com o ridge) encolhido em direcao a climatologia.

    pred = clim + sum_k alfa_k * (p_k - clim)      (em espaco de componentes PCA de tp)

Reaproveita a mesma reducao PLS(defasado) ajustada ate 2018-12 do run promovido
(models/_reduction_cache/pls_lagged_shift1.joblib) - a mesma que a dobra "2018" da CV validou.
A climatologia (em componentes) usa todos os meses de 1940-2022. Nao aplica a correcao ENSO
pos-hoc (a CV dela mostrou que piora: ver models/pls_lagged_lstm_run1/enso_correction/report.json).

Uso (a partir da raiz do repositorio), com os valores lidos do relatorio da CV:
    python3 -m scripts.final_ensemble --config anom_wd --epochs 3 --seeds 0 1 2 3 4 --alphas 0.8
    python3 -m scripts.final_ensemble --config anom_wd --epochs 3 --seeds 0 1 2 3 4 \
        --ridge-config ridge_a1000 --alphas 0.7 0.2
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

from scripts.cv_ensemble import CONFIGS, N_JOBS_REDUCTION_CV, _fit_ridge, _train_lstm
from scripts.download_data import download_competition_data
from src.data import (
    ALL_VARS,
    FEATURE_VARS,
    HINDCAST_LEN,
    LAGS,
    TP_VAR,
    TRAIN_END,
    build_examples,
    compute_normalization_stats,
    load_all_datasets,
    stack_features,
)
from src.models.pca_lstm.train import PLS_LAG_SHIFT, load_or_fit_reduction, reconstruct_tp
from src.submit import build_submission


def main(config: str, epochs: int, seeds: list[int], ridge_config: str | None, alphas: list[float], tag: str):
    n_modelos = 1 + (ridge_config is not None)
    assert len(alphas) == n_modelos, f"--alphas precisa de {n_modelos} valor(es) (LSTM{' + ridge' if ridge_config else ''})"
    cfg = CONFIGS[config]
    assert "ridge_alpha" not in cfg, "--config deve ser um LSTM; use --ridge-config para o ridge"

    print("=== 1. Carregando dados e reducao (ajustada ate 2018-12, como no run promovido) ===")
    datasets = load_all_datasets()
    ds = stack_features(datasets)
    teste_ds = datasets["teste_features"]
    time_index = pd.DatetimeIndex(ds["time"].values)
    last_valid_idx = len(time_index) - 1
    train_end_idx = time_index.get_indexer([pd.Timestamp(TRAIN_END)])[0]
    stats = compute_normalization_stats(ds, train_end=TRAIN_END)
    reduction_objects, component_series = load_or_fit_reduction(
        ds, stats, train_end_idx, method="pls_lagged", pls_lag_shift=PLS_LAG_SHIFT, n_jobs=N_JOBS_REDUCTION_CV
    )
    tp_pca, stats_tp = reduction_objects[TP_VAR], stats[TP_VAR]
    n_tp = tp_pca.n_components_
    n_atm = sum(reduction_objects[v].n_components_ for v in FEATURE_VARS)
    n_hind = n_atm + n_tp

    print("\n=== 2. Exemplos de treino: todas as origens 1940-2022 ===")
    hind, atm, tpf, lag, y, _, alvo = build_examples(component_series, 0, last_valid_idx, last_valid_idx=last_valid_idx)
    meses = time_index.month.values
    tp_c = component_series[TP_VAR]
    clim_comp = np.stack([tp_c[meses == m].mean(axis=0) for m in range(1, 13)]).astype("float32")
    clim_tr = clim_comp[meses[alvo] - 1]
    print(f"  {len(y)} exemplos")

    print("\n=== 3. Entradas do teste real (2023-2024) ===")
    hindcast_teste = np.concatenate(
        [component_series[v][last_valid_idx - HINDCAST_LEN + 1 : last_valid_idx + 1] for v in ALL_VARS], axis=1
    )
    n_teste = teste_ds.sizes["time"]
    atm_teste = np.concatenate(
        [
            reduction_objects[v].transform((teste_ds[v].values.astype("float32") - stats[v][0]) / stats[v][1])
            for v in FEATURE_VARS
        ],
        axis=1,
    ).astype("float32")
    hind_te = np.repeat(hindcast_teste[None], n_teste, axis=0).astype("float32")
    tpf_te = np.repeat(component_series[TP_VAR][last_valid_idx][None], n_teste, axis=0).astype("float32")
    lag_te = (teste_ds["lag_meses"].values / max(LAGS)).astype("float32")
    clim_te = clim_comp[pd.DatetimeIndex(teste_ds["time"].values).month.values - 1]
    tr, te = (hind, atm, tpf, lag), (hind_te, atm_teste, tpf_te, lag_te)

    print(f"\n=== 4. LSTM {config}: {len(seeds)} seeds x {epochs} epocas ===")
    shift_tr = clim_tr if cfg["anomaly"] else np.zeros_like(clim_tr)
    shift_te = clim_te if cfg["anomaly"] else np.zeros_like(clim_te)
    preds = []
    for seed in seeds:
        print(f"  -- seed {seed}", flush=True)
        p, _, _ = _train_lstm(cfg, seed, n_hind, n_atm, n_tp, tr, y - shift_tr, te, shift_te, None, epochs)
        preds.append(p[-1])
    p_lstm = np.mean(preds, axis=0)
    modelos = [p_lstm]

    if ridge_config:
        print(f"\n=== 4b. Ridge {ridge_config} ===")
        modelos.append(_fit_ridge(CONFIGS[ridge_config]["ridge_alpha"], tr, y - clim_tr, te, clim_te)[0])

    pred_comp = clim_te + sum(a * (p - clim_te) for a, p in zip(alphas, modelos))

    print("\n=== 5. Reconstruindo a grade e gerando a submissao ===")
    pred_grid = np.clip(reconstruct_tp(tp_pca, stats_tp, pred_comp), 0, None)  # precipitacao nao e negativa
    predictions_da = xr.DataArray(
        pred_grid,
        dims=("time", "lat", "lon"),
        coords={"time": teste_ds["time"].values, "lat": teste_ds["lat"].values, "lon": teste_ds["lon"].values},
    )
    sample_path = os.path.join(download_competition_data(), "sample_submission.csv")
    os.makedirs("submissions", exist_ok=True)
    output_path = f"submissions/submission_{tag}.csv"
    df = build_submission(predictions_da, sample_path, output_path)
    print(f"  salvo em {output_path} ({len(df)} linhas)")

    # diagnostico: distancia para a climatologia e para o run promovido (nao ha rotulo do teste)
    clim_grid = np.clip(reconstruct_tp(tp_pca, stats_tp, clim_te), 0, None)
    print(f"  RMS(pred - clim) = {np.sqrt(np.mean((pred_grid - clim_grid) ** 2)):.4f} mm/dia | "
          f"media da previsao {pred_grid.mean():.4f} vs clim {clim_grid.mean():.4f}")
    anterior = "submissions/submission_pls_lagged_lstm.csv"
    if os.path.exists(anterior):
        ant = pd.read_csv(anterior).set_index("id")["tp_mm_day"]
        atual = df.set_index("id")["tp_mm_day"].reindex(ant.index)
        print(f"  RMS(pred - submissao promovida anterior) = {np.sqrt(np.mean((atual - ant) ** 2)):.4f} mm/dia")

    with open(f"models/_cv/final_{tag}.json", "w") as f:
        json.dump(
            {"config": config, "cfg": cfg, "epochs": epochs, "seeds": seeds, "ridge_config": ridge_config,
             "alphas": alphas, "output": output_path}, f, indent=2,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, choices=[c for c in CONFIGS if "ridge_alpha" not in CONFIGS[c]])
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--ridge-config", choices=[c for c in CONFIGS if "ridge_alpha" in CONFIGS[c]])
    p.add_argument("--alphas", type=float, nargs="+", required=True, help="pesos de encolhimento: LSTM [, ridge]")
    p.add_argument("--tag", default="pls_lagged_ensemble_lstm")
    a = p.parse_args()
    main(a.config, a.epochs, a.seeds, a.ridge_config, a.alphas, a.tag)
