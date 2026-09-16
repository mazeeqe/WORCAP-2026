"""Modelo C: XGBoost sobre a mesma representacao PCA do Modelo A (LSTM).

Ideia (ver PESQUISA_LSTM_XGBOOST.md, secao 6): reaproveitar a MESMA reducao
espacial (SpatialPCA, `fit_reduction_per_variable` de src/train_pca_lstm.py --
literalmente a funcao do Tomaz) e o MESMO contrato de exemplos origem+lag
(src/data.py build_examples), mas trocar a LSTM por XGBoost como modelo de
regressao sobre os coeficientes PCA de tp. Como o XGBoost nao tem estrutura
recorrente, o hindcast (H meses x F variaveis) e achatado num vetor so em vez
de processado sequencialmente -- perde a nocao de "ordem" explicita dentro da
janela, mas cada mes/variavel ainda vira uma feature que o XGBoost pode usar.

Isso permite comparar os dois modelos de forma justa (mesma reducao, mesmo
split, mesma metrica -- ver src/evaluate.py) e, depois, combinar as previsoes
dos dois via um meta-modelo (Ridge) treinado so na validacao -- ver
`ensemble_with_lstm()`, que so roda se voce tiver um run do Modelo A salvo
localmente (os pesos do modelo nao sao versionados, ver .gitignore).

Uso (a partir da raiz do repositorio):
    python3 -m src.train_xgboost
    python3 -m src.train_xgboost --ensemble-with experiments/pca_lstm_run1
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import xarray as xr
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from download_data import download_competition_data
from src.baseline import climatology_baseline, persistence_baseline
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
from src.evaluate import evaluate_predictions
from src.models.pca_lstm import HindcastForecastLSTM
from src.submit import build_submission
from src.train_pca_lstm import fit_reduction_per_variable, reconstruct_tp

N_COMPONENTS = 20  # igual ao train_pca_lstm.py, pra reducao/comparacao ficarem no mesmo pe
RUN_DIR = "experiments/xgboost_run1"

XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)


def flatten_features(hindcast: np.ndarray, alvo_atm: np.ndarray, tp_congelado: np.ndarray,
                      lag: np.ndarray) -> np.ndarray:
    """XGBoost nao processa sequencia -- achata o hindcast (N, H, F) em (N, H*F) e
    concatena com as features do mes alvo, tp congelado (origem) e o lag."""
    n = hindcast.shape[0]
    hindcast_flat = hindcast.reshape(n, -1)
    return np.concatenate([hindcast_flat, alvo_atm, tp_congelado, lag.reshape(-1, 1)], axis=1).astype("float32")


def train_xgboost(X_tr: np.ndarray, y_tr: np.ndarray) -> MultiOutputRegressor:
    """Um XGBRegressor por componente PCA de tp (MultiOutputRegressor) -- mais simples e
    portavel entre versoes do xgboost do que a API nativa de multi-output."""
    modelo = MultiOutputRegressor(XGBRegressor(**XGB_PARAMS), n_jobs=-1)
    modelo.fit(X_tr, y_tr)
    return modelo


def ensemble_with_lstm(run_dir_lstm: str, hindcast, atm, tp_froz, lag, X_flat,
                        modelo_xgb, stats, reduction_objects, tp_raw, origin_idx, alvo_idx,
                        time_index) -> dict | None:
    """Combina as previsoes do XGBoost com as de um Modelo A (LSTM) ja treinado e salvo
    localmente em `run_dir_lstm` (ver train_pca_lstm.py) via um meta-modelo Ridge,
    treinado so na validacao (nunca no teste - mesma regra do stacking descrito em
    PESQUISA_LSTM_XGBOOST.md).

    Retorna None se os artefatos do LSTM nao existirem localmente (nao sao versionados
    no git -- cada pessoa precisa ter rodado `python3 -m src.train_pca_lstm` antes)."""
    caminho_modelo = os.path.join(run_dir_lstm, "model_final.pt")
    caminho_reducao = os.path.join(run_dir_lstm, "reduction_and_stats.joblib")
    if not (os.path.exists(caminho_modelo) and os.path.exists(caminho_reducao)):
        print(f"\n[ensemble] Artefatos do Modelo A nao encontrados em {run_dir_lstm} "
              f"(rode 'python3 -m src.train_pca_lstm' primeiro para gerar). Pulando ensemble.")
        return None

    artefatos_lstm = joblib.load(caminho_reducao)
    n_features_hindcast = N_COMPONENTS * len(ALL_VARS)
    n_features_atm = N_COMPONENTS * len(FEATURE_VARS)
    lstm = HindcastForecastLSTM(
        n_features_hindcast=n_features_hindcast, n_features_atm=n_features_atm,
        n_components_tp=N_COMPONENTS,
    )
    lstm.load_state_dict(torch.load(caminho_modelo, map_location="cpu"))
    lstm.eval()

    with torch.no_grad():
        pred_pca_lstm = lstm(
            torch.from_numpy(hindcast), torch.from_numpy(atm),
            torch.from_numpy(tp_froz), torch.from_numpy(lag),
        ).numpy()
    pred_pca_xgb = modelo_xgb.predict(X_flat)

    # O ajuste do meta-modelo (Ridge, treinado so na validacao) fica no chamador
    # (main()), que tem acesso ao y_va real -- aqui so geramos as duas opinioes.
    return {
        "pred_pca_xgb": pred_pca_xgb, "pred_pca_lstm": pred_pca_lstm,
    }


def main(ensemble_with: str | None = None):
    os.makedirs(RUN_DIR, exist_ok=True)

    print("=== 1. Carregando dados ===")
    datasets = load_all_datasets()
    ds = stack_features(datasets)
    time_index = pd.DatetimeIndex(ds["time"].values)
    train_end_idx = time_index.get_indexer([pd.Timestamp(TRAIN_END)])[0]
    last_valid_idx = len(time_index) - 1
    print(f"  {len(time_index)} meses ({time_index[0].date()} a {time_index[-1].date()})")

    stats = compute_normalization_stats(ds, train_end=TRAIN_END)
    tp_raw = ds[TP_VAR].values.astype("float32").copy()

    print("\n=== 2. Reduzindo dimensionalidade por variavel (PCA, so no treino interno) ===")
    print("  (reaproveitando fit_reduction_per_variable de src/train_pca_lstm.py)")
    reduction_objects, component_series = fit_reduction_per_variable(
        ds, stats, train_end_idx, method="pca"
    )

    print("\n=== 3. Montando exemplos (contrato origem + lag) ===")
    train_ex = build_examples(component_series, 0, train_end_idx, last_valid_idx=train_end_idx)
    val_ex = build_examples(component_series, train_end_idx + 1, last_valid_idx, last_valid_idx=last_valid_idx)
    hindcast_tr, atm_tr, tp_froz_tr, lag_tr, y_tr, origin_tr, alvo_tr = train_ex
    hindcast_va, atm_va, tp_froz_va, lag_va, y_va, origin_va, alvo_va = val_ex
    print(f"  treino: {len(y_tr)} exemplos | validacao: {len(y_va)} exemplos")

    X_tr = flatten_features(hindcast_tr, atm_tr, tp_froz_tr, lag_tr)
    X_va = flatten_features(hindcast_va, atm_va, tp_froz_va, lag_va)
    print(f"  features achatadas: {X_tr.shape[1]} colunas "
          f"({HINDCAST_LEN}meses x {len(ALL_VARS)}vars x {N_COMPONENTS}comp + atm + tp_congelado + lag)")

    print("\n=== 4. Treinando XGBoost (1 regressor por componente PCA de tp) ===")
    modelo = train_xgboost(X_tr, y_tr)

    print("\n=== 5. Avaliando contra baselines (grade real, mm/dia) ===")
    pred_pca_va = modelo.predict(X_va)
    pred_grid = reconstruct_tp(reduction_objects[TP_VAR], stats[TP_VAR], pred_pca_va)
    true_grid = tp_raw[alvo_va]
    persist_grid = persistence_baseline(tp_raw[origin_va])
    meses_alvo = time_index.month.values[alvo_va]
    clim_grid = climatology_baseline(ds[TP_VAR].sel(time=slice(None, TRAIN_END)), list(meses_alvo))

    lag_inteiro_va = (lag_va * max(LAGS)).round().astype(int)
    metrics_model = evaluate_predictions(true_grid, pred_grid, lags=lag_inteiro_va)
    metrics_persist = evaluate_predictions(true_grid, persist_grid, lags=lag_inteiro_va)
    metrics_clim = evaluate_predictions(true_grid, clim_grid, lags=lag_inteiro_va)

    print(f"  XGBoost       RMSE={metrics_model['rmse']:.3f}  MAE={metrics_model['mae']:.3f}  (mm/dia)")
    print(f"  Persistencia  RMSE={metrics_persist['rmse']:.3f}  MAE={metrics_persist['mae']:.3f}  (mm/dia)")
    print(f"  Climatologia  RMSE={metrics_clim['rmse']:.3f}  MAE={metrics_clim['mae']:.3f}  (mm/dia)")

    with open(f"{RUN_DIR}/metrics.json", "w") as f:
        json.dump({
            "modelo": metrics_model, "persistencia": metrics_persist, "climatologia": metrics_clim,
            "xgb_params": XGB_PARAMS, "n_components": N_COMPONENTS,
        }, f, indent=2)
    print(f"  metrics.json salvo em {RUN_DIR}/")

    ensemble_result = None
    if ensemble_with:
        ensemble_result = ensemble_with_lstm(
            ensemble_with, hindcast_va, atm_va, tp_froz_va, lag_va, X_va,
            modelo, stats, reduction_objects, tp_raw, origin_va, alvo_va, time_index,
        )
        if ensemble_result:
            pred_pca_lstm = ensemble_result["pred_pca_lstm"]
            pred_pca_xgb = ensemble_result["pred_pca_xgb"]
            meta = MultiOutputRegressor(Ridge(alpha=1.0))
            opinioes_va = np.concatenate([pred_pca_xgb, pred_pca_lstm], axis=1)  # (N, 2*N_COMPONENTS)
            meta.fit(opinioes_va, y_va)
            pred_pca_ens = meta.predict(opinioes_va)
            pred_grid_ens = reconstruct_tp(reduction_objects[TP_VAR], stats[TP_VAR], pred_pca_ens)
            metrics_ens = evaluate_predictions(true_grid, pred_grid_ens, lags=lag_inteiro_va)
            print(f"  Ensemble (XGB+LSTM, Ridge)  RMSE={metrics_ens['rmse']:.3f}  MAE={metrics_ens['mae']:.3f}  (mm/dia)")
            with open(f"{RUN_DIR}/metrics_ensemble.json", "w") as f:
                json.dump({"ensemble": metrics_ens, "xgb_sozinho": metrics_model}, f, indent=2)

    print("\n=== 6. Retreinando com todo o historico rotulado (1940-2022) ===")
    full_ex = build_examples(component_series, 0, last_valid_idx, last_valid_idx=last_valid_idx)
    hindcast_full, atm_full, tp_froz_full, lag_full, y_full, _, _ = full_ex
    X_full = flatten_features(hindcast_full, atm_full, tp_froz_full, lag_full)
    modelo_final = train_xgboost(X_full, y_full)
    joblib.dump(modelo_final, f"{RUN_DIR}/model_final.joblib")
    joblib.dump({"reduction_objects": reduction_objects, "stats": stats}, f"{RUN_DIR}/reduction_and_stats.joblib")
    print(f"  modelo final salvo em {RUN_DIR}/")

    print("\n=== 7. Prevendo o teste real (2023-2024) ===")
    teste_ds = datasets["teste_features"]
    origem_idx = last_valid_idx
    hindcast_teste = np.concatenate(
        [component_series[v][origem_idx - HINDCAST_LEN + 1: origem_idx + 1] for v in ALL_VARS], axis=1
    )
    tp_congelado_teste = component_series[TP_VAR][origem_idx]

    n_meses_teste = teste_ds.sizes["time"]
    atm_teste_list = []
    for var in FEATURE_VARS:
        media, desvio = stats[var]
        bruto = teste_ds[var].values.astype("float32")
        normalizado = (bruto - media) / desvio
        atm_teste_list.append(reduction_objects[var].transform(normalizado))
    atm_teste = np.concatenate(atm_teste_list, axis=1).astype("float32")

    hindcast_batch = np.repeat(hindcast_teste[None, :, :], n_meses_teste, axis=0)
    tp_congelado_batch = np.repeat(tp_congelado_teste[None, :], n_meses_teste, axis=0)
    lag_batch = (teste_ds["lag_meses"].values / max(LAGS)).astype("float32")

    X_teste = flatten_features(hindcast_batch, atm_teste, tp_congelado_batch, lag_batch)
    pred_pca_teste = modelo_final.predict(X_teste)
    pred_grid_teste = reconstruct_tp(reduction_objects[TP_VAR], stats[TP_VAR], pred_pca_teste)
    pred_grid_teste = np.clip(pred_grid_teste, 0, None)

    predictions_da = xr.DataArray(
        pred_grid_teste, dims=("time", "lat", "lon"),
        coords={"time": teste_ds["time"].values, "lat": teste_ds["lat"].values, "lon": teste_ds["lon"].values},
    )

    print("\n=== 8. Gerando submissao ===")
    os.makedirs("submissions", exist_ok=True)
    competition_path = download_competition_data()
    sample_path = os.path.join(competition_path, "sample_submission.csv")
    output_path = "submissions/submission_xgboost.csv"
    submission_df = build_submission(predictions_da, sample_path, output_path)
    print(f"  salvo em {output_path} ({len(submission_df)} linhas)")

    return predictions_da, submission_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ensemble-with", default=None, metavar="RUN_DIR",
        help="Pasta de um run local do Modelo A ja treinado (ex: experiments/pca_lstm_run1) "
             "para combinar as previsoes via Ridge. Requer ter rodado 'python3 -m "
             "src.train_pca_lstm' antes (os pesos nao sao versionados no git).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(ensemble_with=args.ensemble_with)
