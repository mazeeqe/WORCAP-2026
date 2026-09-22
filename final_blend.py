"""Reproduz, para a submissao final, o melhor esquema de "encolhimento" (shrinkage em
direcao a climatologia) encontrado por `cv_ensemble.py --report` (models/_cv/report.json):
hoje um blend de 3 LSTMs (anom_wd_do3 + anom_wd_do3_oni + anom_wd), RMSE-CV
(leave-one-fold-out) 1.778, contra 1.783 do melhor modelo unico e 1.858 do run promovido
anterior (models/pls_lagged_lstm_run1, sem esta CV de 5 dobras).

Le o relatorio para descobrir automaticamente qual combinacao venceu (menor `rmse_lofo`
entre as chaves `encolhimento_*`) e qual epoca/seeds usar por config (secao `configs` do
mesmo relatorio) - nao hardcoda os hiperparametros vencedores, entao acompanha se
`cv_ensemble.py --report` for rodado de novo com mais dados/configs. Retreina cada modelo
do blend com TODO o historico rotulado (1940-2022, como final_ensemble.py) e combina com
os `alfa` do relatorio antes de reconstruir a grade e gerar a submissao. Nao aplica a
correcao ENSO pos-hoc (a CV dela mostrou que piora: ver
models/pls_lagged_lstm_run1/enso_correction/report.json).

Uso (a partir da raiz do repositorio):
    python3 final_blend.py
    python3 final_blend.py --report models/_cv/report.json --tag meu_blend
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

from cv_ensemble import CONFIGS, N_JOBS_REDUCTION_CV, _fit_ridge, _train_lstm
from download_data import download_competition_data
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
from src.oni import oni_para_datas, oni_series_or_nan
from src.submit import build_submission


def _melhor_combo(relatorio: dict) -> tuple[str, list[str], list[float]]:
    chaves = {k: v for k, v in relatorio.items() if k.startswith("encolhimento_")}
    if not chaves:
        raise ValueError("relatorio sem nenhuma chave 'encolhimento_*' - rode cv_ensemble.py --report primeiro")
    melhor_chave = min(chaves, key=lambda k: chaves[k]["rmse_lofo"])
    tokens = melhor_chave[len("encolhimento_"):].split("+")
    return melhor_chave, tokens, chaves[melhor_chave]["alfa"]


def _resolve_token(token: str, configs_info: dict) -> tuple[str, bool]:
    """Devolve (nome_da_config, e_ridge), replicando a resolucao de candidatos de
    cv_ensemble.report()."""
    if token.startswith("m_"):
        config = token[2:]
        return config, config.startswith("ridge")
    if token == "lstm":
        lstm = {c: v for c, v in configs_info.items() if not c.startswith("ridge")}
        return min(lstm, key=lambda c: lstm[c]["rmse_media"]), False
    if token == "ridge":
        ridge = {c: v for c, v in configs_info.items() if c.startswith("ridge")}
        return min(ridge, key=lambda c: ridge[c]["rmse_media"]), True
    raise ValueError(f"token de candidato desconhecido: {token!r}")


def main(report_path: str, tag: str | None):
    relatorio = json.load(open(report_path))
    melhor_chave, tokens, alfas = _melhor_combo(relatorio)
    configs_info = relatorio["configs"]
    modelos = [_resolve_token(t, configs_info) for t in tokens]
    print(f"=== Melhor esquema no relatorio: {melhor_chave} (RMSE-CV LOFO {relatorio[melhor_chave]['rmse_lofo']:.4f}) ===")
    for (config, e_ridge), alfa in zip(modelos, alfas):
        detalhe = "ridge" if e_ridge else f"epoca {configs_info[config]['epoca']}, seeds {configs_info[config]['seeds']}"
        print(f"  {config} ({detalhe}, alfa={alfa:.4f})")
    tag = tag or melhor_chave.replace("encolhimento_", "").replace("+", "_")

    print("\n=== 1. Carregando dados e reducao (pls_lagged, ajustada ate 2018-12, como no run promovido) ===")
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
    tr = (hind, atm, tpf, lag)
    print(f"  {len(y)} exemplos")

    precisa_oni = any(CONFIGS[c].get("use_oni_feature") for c, _ in modelos)
    tr_oni = clim_tr_oni = y_oni = oni_tr = None
    if precisa_oni:
        oni_full = oni_series_or_nan(time_index)
        hind_o, atm_o, tpf_o, lag_o, y_oni, _, alvo_o, oni_tr = build_examples(
            component_series, 0, last_valid_idx, last_valid_idx=last_valid_idx, oni_series=oni_full
        )
        tr_oni = (hind_o, atm_o, tpf_o, lag_o)
        clim_tr_oni = clim_comp[meses[alvo_o] - 1]
        print(f"  treino com ONI: {len(y_oni)} exemplos (vs. {len(y)} sem - origens antes de 1950 descartadas)")

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
    te = (hind_te, atm_teste, tpf_te, lag_te)

    oni_te = None
    if precisa_oni:
        # mesmo alinhamento anti-vazamento do treino (feature_idx = alvo_idx - 1): ONI do mes
        # anterior a cada mes-alvo do teste (ver src/models/pca_lstm/train.py)
        datas_alvo_teste = pd.DatetimeIndex(teste_ds["time"].values) - pd.DateOffset(months=1)
        oni_te = oni_para_datas(datas_alvo_teste).reshape(-1, 1)

    print(f"\n=== 4. Treinando os {len(modelos)} modelos do blend ===")
    preds_por_modelo = []
    for config, e_ridge in modelos:
        cfg = CONFIGS[config]
        if e_ridge:
            print(f"  -- {config} (ridge)", flush=True)
            pred = _fit_ridge(cfg["ridge_alpha"], tr, y - clim_tr, te, clim_te)[0]
        else:
            epocas, seeds = configs_info[config]["epoca"], configs_info[config]["seeds"]
            com_oni = cfg.get("use_oni_feature", False)
            print(f"  -- {config}: {len(seeds)} seed(s) x {epocas} epocas" + (" (com ONI)" if com_oni else ""), flush=True)
            preds_seed = []
            for seed in seeds:
                print(f"     seed {seed}", flush=True)
                if com_oni:
                    p, _, _ = _train_lstm(
                        cfg, seed, n_hind, n_atm, n_tp, tr_oni, y_oni - clim_tr_oni, te, clim_te, None, epocas,
                        oni_tr=oni_tr, oni_va=oni_te,
                    )
                else:
                    p, _, _ = _train_lstm(cfg, seed, n_hind, n_atm, n_tp, tr, y - clim_tr, te, clim_te, None, epocas)
                preds_seed.append(p[-1])
            pred = np.mean(preds_seed, axis=0)
        preds_por_modelo.append(pred)

    pred_comp = clim_te + sum(a * (p - clim_te) for a, p in zip(alfas, preds_por_modelo))

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
            {
                "melhor_chave": melhor_chave,
                "rmse_lofo_cv": relatorio[melhor_chave]["rmse_lofo"],
                "modelos": [c for c, _ in modelos],
                "alfas": alfas,
                "output": output_path,
            },
            f, indent=2,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", default="models/_cv/report.json")
    p.add_argument("--tag", help="default: derivado da chave vencedora no relatorio")
    a = p.parse_args()
    main(a.report, a.tag)
