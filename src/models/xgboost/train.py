"""Modelo C: XGBoost sobre a mesma representacao PCA usada no Modelo A (LSTM).

Ideia (ver PESQUISA_LSTM_XGBOOST.md, secao 6): reaproveita a MESMA reducao
espacial (`fit_reduction_per_variable`, de src/models/pca_lstm/train.py --
literalmente a funcao do Tomaz, agora com numero de componentes escolhido
automaticamente por variavel via criterio de 90% de variancia) e o MESMO
contrato de exemplos origem+lag (src/data.py build_examples), mas troca a
LSTM por XGBoost como regressor dos coeficientes PCA de tp. Como o XGBoost
nao tem estrutura recorrente, o hindcast (H meses x variaveis x componentes)
e achatado num vetor so em vez de processado sequencialmente.

Depois de treinado, pode ser combinado com um Modelo A ja treinado e salvo
localmente (`--ensemble-with models/pca_lstm_run1`, por exemplo) via um
meta-modelo Ridge por componente, treinado so na validacao (stacking, ver
PESQUISA_LSTM_XGBOOST.md).

Uso (a partir da raiz do repositorio):
    python3 -m src.models.xgboost.train
    python3 -m src.models.xgboost.train --ensemble-with models/pls_concurrent_lstm_run1
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
from src.models.pca_lstm.train import fit_reduction_per_variable, reconstruct_tp
from src.submit import build_submission

RUN_DIR = "models/xgboost_run1"
RUN_DIR_ONI = "models/xgboost_oni_run1"
CAMINHO_ONI = "oni_mensal.csv"  # gerado localmente por preparar_oni.py (nao versionado)

XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric="rmse", early_stopping_rounds=30,
)


# ════════════════════════════════════════════════════════════
# EXPERIMENTO: ONI (indice El Nino/La Nina) como feature de entrada
# ════════════════════════════════════════════════════════════
# Motivacao: models/pls_lagged_lstm_run1/enso_correction/ (Tomaz) corrige o
# resultado DEPOIS via uma correcao linear global calibrada so em 2019-2022
# (ONI fraco/moderado, -1.10 a +0.90) -- nao cobre o El Nino extremo real de
# 2023-2024 (+2.0) e por isso satura em vez de extrapolar. Aqui, em vez de
# corrigir depois, o ONI vira uma FEATURE DE ENTRADA do XGBoost, treinada
# com os 83 anos inteiros de historico (que incluem El Ninos fortes de
# verdade: 1982-83, 1997-98, 2015-16) -- ver discussao no grupo antes de
# decidir se isso deve entrar em src/data.py pra todo mundo (LSTM e
# ConvLSTM tambem se beneficiariam do mesmo sinal).
def carregar_oni_por_data(caminho: str = CAMINHO_ONI) -> pd.Series:
    """Serie do indice ONI (NOAA CPC, ver preparar_oni.py) indexada por mes.
    NAO tem cobertura antes de 1950 -- a fonte oficial nao publica ONI pra
    tras disso. Meses de 1940-1949 do treino ficam sem valor real (ver
    alinhar_oni, preenchido com 0.0 = neutro); e uma limitacao conhecida
    deste experimento, nao escondida."""
    df = pd.read_csv(caminho, parse_dates=["data"])
    return df.set_index("data")["oni"]


def alinhar_oni(oni_por_data: pd.Series, datas) -> np.ndarray:
    """Valor de ONI pra cada uma das `datas` (mes-a-mes); 0.0 (neutro) fora da
    cobertura da fonte (pre-1950, ver carregar_oni_por_data)."""
    return oni_por_data.reindex(pd.DatetimeIndex(datas)).fillna(0.0).values.astype("float32")


def montar_features_oni(oni_serie: np.ndarray, origin_idx: np.ndarray, alvo_idx: np.ndarray,
                         hindcast_len: int) -> np.ndarray:
    """oni_serie: array (n_meses,) alinhado aos MESMOS indices inteiros usados em
    component_series (src/data.py). Monta, por exemplo: os H meses de ONI da
    janela de hindcast (mesma janela das demais variaveis) + o ONI do mes
    o+L-1 (mesma convencao de alvo_atm em build_examples -- sem vazamento,
    ver a correcao do Tomaz em src/data.py)."""
    hindcast_cols = np.stack([oni_serie[o - hindcast_len + 1: o + 1] for o in origin_idx])
    alvo_col = oni_serie[alvo_idx - 1].reshape(-1, 1)
    return np.concatenate([hindcast_cols, alvo_col], axis=1).astype("float32")


def flatten_features(hindcast: np.ndarray, alvo_atm: np.ndarray, tp_congelado: np.ndarray,
                      lag: np.ndarray, oni_extra: np.ndarray | None = None) -> np.ndarray:
    """XGBoost nao processa sequencia -- achata o hindcast (N, H, F) em (N, H*F) e
    concatena com as features do mes alvo, tp congelado (origem), o lag, e
    opcionalmente as colunas de ONI (ver montar_features_oni, --with-oni)."""
    n = hindcast.shape[0]
    hindcast_flat = hindcast.reshape(n, -1)
    partes = [hindcast_flat, alvo_atm, tp_congelado, lag.reshape(-1, 1)]
    if oni_extra is not None:
        partes.append(oni_extra)
    return np.concatenate(partes, axis=1).astype("float32")


class MultiOutputXGB:
    """1 XGBRegressor por componente PCA de tp -- mais simples e portavel entre
    versoes do xgboost do que a API nativa de multi-output, e permite registrar a
    curva de MSE de validacao por rodada de boosting (early stopping) por componente,
    igual ao "epoch" do LSTM (ver secao 2 de resultados_pca_lstm.ipynb)."""

    def __init__(self, params: dict, n_components: int):
        self.params = params
        self.n_components = n_components
        self.estimadores: list[XGBRegressor] = []
        self.melhor_iteracao_por_componente: list[int] = []

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray) -> "MultiOutputXGB":
        curvas_treino, curvas_val = [], []
        for c in range(self.n_components):
            modelo_c = XGBRegressor(**self.params)
            modelo_c.fit(X_tr, y_tr[:, c], eval_set=[(X_tr, y_tr[:, c]), (X_va, y_va[:, c])], verbose=False)
            self.estimadores.append(modelo_c)
            self.melhor_iteracao_por_componente.append(modelo_c.best_iteration)
            resultados = modelo_c.evals_result()
            curvas_treino.append(resultados["validation_0"]["rmse"])
            curvas_val.append(resultados["validation_1"]["rmse"])
        # curvas de tamanhos diferentes (cada componente para no seu best_iteration + patience) --
        # trunca no menor comprimento comum so pra ter 1 curva agregada pra visualizacao
        min_len = min(len(c) for c in curvas_val)
        rmse_treino = np.array([c[:min_len] for c in curvas_treino])
        rmse_val = np.array([c[:min_len] for c in curvas_val])
        self.history_ = pd.DataFrame({
            "epoch": np.arange(1, min_len + 1),
            "train_mse_pca": (rmse_treino ** 2).mean(axis=0),
            "val_mse_pca": (rmse_val ** 2).mean(axis=0),
        })
        return self

    def fit_no_eval(self, X: np.ndarray, y: np.ndarray) -> "MultiOutputXGB":
        """Treino final (sem early stopping/eval_set -- usa o numero medio de
        rodadas escolhido no fit() de selecao acima, arredondado)."""
        n_estimators_final = int(round(np.mean(self.melhor_iteracao_por_componente))) + 1
        params_finais = {k: v for k, v in self.params.items() if k not in ("early_stopping_rounds", "eval_metric")}
        params_finais["n_estimators"] = n_estimators_final
        self.estimadores = []
        for c in range(self.n_components):
            modelo_c = XGBRegressor(**params_finais)
            modelo_c.fit(X, y[:, c])
            self.estimadores.append(modelo_c)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([e.predict(X) for e in self.estimadores])


def load_lstm_run(run_dir: str) -> dict | None:
    """Carrega um Modelo A ja treinado e salvo localmente (pesos nao versionados no
    git -- retorna None se o run nao existir aqui, ver .gitignore)."""
    caminho_modelo = os.path.join(run_dir, "model_final.pt")
    caminho_reducao = os.path.join(run_dir, "reduction_and_stats.joblib")
    if not (os.path.exists(caminho_modelo) and os.path.exists(caminho_reducao)):
        return None

    artefatos = joblib.load(caminho_reducao)
    reduction_objects = artefatos["reduction_objects"]
    n_components_tp = reduction_objects[TP_VAR].n_components_
    n_features_hindcast = sum(reduction_objects[v].n_components_ for v in ALL_VARS)
    n_features_atm = sum(reduction_objects[v].n_components_ for v in FEATURE_VARS)

    lstm = HindcastForecastLSTM(
        n_features_hindcast=n_features_hindcast, n_features_atm=n_features_atm,
        n_components_tp=n_components_tp,
    )
    lstm.load_state_dict(torch.load(caminho_modelo, map_location="cpu"))
    lstm.eval()
    return {"lstm": lstm, "reduction_objects": reduction_objects, "stats": artefatos["stats"]}


def main(ensemble_with: str | None = None, run_dir: str | None = None, with_oni: bool = False):
    run_dir = run_dir or (RUN_DIR_ONI if with_oni else RUN_DIR)
    os.makedirs(run_dir, exist_ok=True)

    oni_por_data = carregar_oni_por_data() if with_oni else None
    if with_oni:
        print(f"=== 0. ONI ativado -- {len(oni_por_data)} meses carregados de {CAMINHO_ONI} "
              f"({oni_por_data.index.min().date()} a {oni_por_data.index.max().date()}) ===")

    print("=== 1. Carregando dados ===")
    datasets = load_all_datasets()
    ds = stack_features(datasets)
    time_index = pd.DatetimeIndex(ds["time"].values)
    train_end_idx = time_index.get_indexer([pd.Timestamp(TRAIN_END)])[0]
    last_valid_idx = len(time_index) - 1
    print(f"  {len(time_index)} meses ({time_index[0].date()} a {time_index[-1].date()})")

    stats = compute_normalization_stats(ds, train_end=TRAIN_END)
    tp_raw = ds[TP_VAR].values.astype("float32").copy()

    oni_serie_treino = alinhar_oni(oni_por_data, time_index) if with_oni else None

    print("\n=== 2. Reduzindo dimensionalidade por variavel (PCA, so no treino interno) ===")
    print("  (reaproveitando fit_reduction_per_variable de src/models/pca_lstm/train.py)")
    reduction_objects, component_series = fit_reduction_per_variable(ds, stats, train_end_idx, method="pca")
    n_components_tp = reduction_objects[TP_VAR].n_components_
    print(f"  tp reduzido a {n_components_tp} componentes")

    print("\n=== 3. Montando exemplos (contrato origem + lag) ===")
    train_ex = build_examples(component_series, 0, train_end_idx, last_valid_idx=train_end_idx)
    val_ex = build_examples(component_series, train_end_idx + 1, last_valid_idx, last_valid_idx=last_valid_idx)
    hindcast_tr, atm_tr, tp_froz_tr, lag_tr, y_tr, origin_tr, alvo_tr = train_ex
    hindcast_va, atm_va, tp_froz_va, lag_va, y_va, origin_va, alvo_va = val_ex
    print(f"  treino: {len(y_tr)} exemplos | validacao: {len(y_va)} exemplos")

    oni_tr = oni_va = None
    if with_oni:
        oni_tr = montar_features_oni(oni_serie_treino, origin_tr, alvo_tr, HINDCAST_LEN)
        oni_va = montar_features_oni(oni_serie_treino, origin_va, alvo_va, HINDCAST_LEN)

    X_tr = flatten_features(hindcast_tr, atm_tr, tp_froz_tr, lag_tr, oni_tr)
    X_va = flatten_features(hindcast_va, atm_va, tp_froz_va, lag_va, oni_va)
    print(f"  features achatadas: {X_tr.shape[1]} colunas" + (" (incluindo ONI)" if with_oni else ""))

    print(f"\n=== 4. Treinando XGBoost ({n_components_tp} regressores, 1 por componente PCA de tp) ===")
    modelo = MultiOutputXGB(XGB_PARAMS, n_components_tp).fit(X_tr, y_tr, X_va, y_va)
    melhor_iter_media = float(np.mean(modelo.melhor_iteracao_por_componente))
    print(f"  melhor iteracao media (early stopping, por componente): {melhor_iter_media:.1f}")

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

    print("\n=== 5b. Salvando artefatos para visualizacao (models/) ===")
    modelo.history_.to_csv(f"{run_dir}/history.csv", index=False)
    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump({
            "xgb_params": {k: v for k, v in XGB_PARAMS.items()},
            "best_epoch": melhor_iter_media, "with_oni": with_oni,
            "modelo": metrics_model, "persistencia": metrics_persist, "climatologia": metrics_clim,
        }, f, indent=2)

    lags_amostra = [l for l in [1, 3, 6, 12, 18, 24] if l in lag_inteiro_va]
    amostras = {"lat": ds["lat"].values, "lon": ds["lon"].values, "lags": np.array(lags_amostra)}
    for nome, grade in [("true", true_grid), ("pred", pred_grid), ("persist", persist_grid), ("clim", clim_grid)]:
        escolhidas = [grade[np.where(lag_inteiro_va == l)[0][0]] for l in lags_amostra]
        amostras[nome] = np.stack(escolhidas)
    amostras["origin_date"] = np.array(
        [str(time_index[origin_va[np.where(lag_inteiro_va == l)[0][0]]].date()) for l in lags_amostra])
    amostras["target_date"] = np.array(
        [str(time_index[alvo_va[np.where(lag_inteiro_va == l)[0][0]]].date()) for l in lags_amostra])
    np.savez_compressed(f"{run_dir}/sample_grids.npz", **amostras)
    print(f"  history.csv, metrics.json e sample_grids.npz salvos em {run_dir}/")

    ensemble_result = None
    if ensemble_with:
        artefatos_lstm = load_lstm_run(ensemble_with)
        if artefatos_lstm is None:
            print(f"\n[ensemble] Artefatos do Modelo A nao encontrados em {ensemble_with} "
                  f"(rode 'python3 -m src.models.pca_lstm.train' primeiro). Pulando ensemble.")
        else:
            with torch.no_grad():
                pred_pca_lstm = artefatos_lstm["lstm"](
                    torch.from_numpy(hindcast_va), torch.from_numpy(atm_va),
                    torch.from_numpy(tp_froz_va), torch.from_numpy(lag_va),
                ).numpy()
            # Ridge por componente: aprende o peso de XGBoost vs LSTM pra cada
            # coeficiente PCA de tp, treinado SO na validacao (nunca no teste).
            pred_pca_ens = np.zeros_like(pred_pca_va)
            for c in range(n_components_tp):
                meta_c = Ridge(alpha=1.0)
                opinioes_c = np.column_stack([pred_pca_va[:, c], pred_pca_lstm[:, c]])
                meta_c.fit(opinioes_c, y_va[:, c])
                pred_pca_ens[:, c] = meta_c.predict(opinioes_c)
            pred_grid_ens = reconstruct_tp(reduction_objects[TP_VAR], stats[TP_VAR], pred_pca_ens)
            metrics_ens = evaluate_predictions(true_grid, pred_grid_ens, lags=lag_inteiro_va)
            print(f"\n  Ensemble (XGB+LSTM, Ridge)  RMSE={metrics_ens['rmse']:.3f}  MAE={metrics_ens['mae']:.3f}  (mm/dia)")
            with open(f"{run_dir}/metrics_ensemble.json", "w") as f:
                json.dump({"ensemble_com": ensemble_with, "ensemble": metrics_ens, "xgb_sozinho": metrics_model}, f, indent=2)
            ensemble_result = metrics_ens

    print("\n=== 6. Retreinando com todo o historico rotulado (1940-2022) ===")
    full_ex = build_examples(component_series, 0, last_valid_idx, last_valid_idx=last_valid_idx)
    hindcast_full, atm_full, tp_froz_full, lag_full, y_full, origin_full, alvo_full = full_ex
    oni_full = montar_features_oni(oni_serie_treino, origin_full, alvo_full, HINDCAST_LEN) if with_oni else None
    X_full = flatten_features(hindcast_full, atm_full, tp_froz_full, lag_full, oni_full)
    modelo_final = MultiOutputXGB(XGB_PARAMS, n_components_tp)
    modelo_final.melhor_iteracao_por_componente = modelo.melhor_iteracao_por_componente
    modelo_final.fit_no_eval(X_full, y_full)
    joblib.dump(modelo_final, f"{run_dir}/model_final.joblib")
    joblib.dump({"reduction_objects": reduction_objects, "stats": stats}, f"{run_dir}/reduction_and_stats.joblib")
    print(f"  modelo final salvo em {run_dir}/")

    print("\n=== 7. Prevendo o teste real (2023-2024) ===")
    teste_ds = datasets["teste_features"]
    origem_idx = last_valid_idx
    hindcast_teste = np.concatenate(
        [component_series[v][origem_idx - HINDCAST_LEN + 1: origem_idx + 1] for v in ALL_VARS], axis=1)
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

    oni_teste = None
    if with_oni:
        # origem fixa (dez/2022) pra todo mundo -- mesma janela de hindcast repetida
        oni_hindcast_teste = np.repeat(
            oni_serie_treino[origem_idx - HINDCAST_LEN + 1: origem_idx + 1][None, :], n_meses_teste, axis=0)
        # alvo_atm do teste ja e o mes o+L-1 (ver teste_features.nc); pega o ONI dessas
        # mesmas datas calendario (fora do intervalo de time_index, por isso por data)
        datas_feature_teste = pd.DatetimeIndex(teste_ds["time"].values) - pd.DateOffset(months=1)
        oni_alvo_teste = alinhar_oni(oni_por_data, datas_feature_teste).reshape(-1, 1)
        oni_teste = np.concatenate([oni_hindcast_teste, oni_alvo_teste], axis=1).astype("float32")

    X_teste = flatten_features(hindcast_batch, atm_teste, tp_congelado_batch, lag_batch, oni_teste)
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
    output_path = f"submissions/submission_xgboost{'_oni' if with_oni else ''}.csv"
    submission_df = build_submission(predictions_da, sample_path, output_path)
    print(f"  salvo em {output_path} ({len(submission_df)} linhas)")

    return predictions_da, submission_df, ensemble_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ensemble-with", default=None, metavar="RUN_DIR",
        help="Pasta de um run local do Modelo A ja treinado (ex: models/pls_concurrent_lstm_run1) "
             "para combinar as previsoes via Ridge. Requer ter rodado 'python3 -m "
             "src.models.pca_lstm.train' antes (os pesos nao sao versionados no git).",
    )
    parser.add_argument(
        "--run-dir", default=None,
        help=f"Pasta de saida dos artefatos (padrao: {RUN_DIR}, ou {RUN_DIR_ONI} com --with-oni).",
    )
    parser.add_argument(
        "--with-oni", action="store_true",
        help="Experimento: adiciona o indice ONI (El Nino/La Nina, NOAA CPC) como feature de "
             "entrada -- ver montar_features_oni. Requer 'python3 preparar_oni.py' rodado antes "
             "(gera oni_mensal.csv localmente, nao versionado).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(ensemble_with=args.ensemble_with, run_dir=args.run_dir, with_oni=args.with_oni)
