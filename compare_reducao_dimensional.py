"""Compara metodos de reducao dimensional (PCA, ICA, Sparse PCA - nao supervisionados;
PLS, CCA - supervisionados contra tp) SEM treinar o LSTM - so pela qualidade da propria
reducao, medida por erro de reconstrucao em dado genuinamente fora da amostra (2019-2022).

Motivacao (ver conversa): treinar o LSTM completo por metodo custa muito tempo (o retreino
so do pls_lagged com os novos tetos de componentes ja levou dezenas de minutos, e um dos
runs anteriores chegou a ser morto pelo OOM killer). Este script e um filtro rapido: mede
o RMSE de reconstrucao (inverse_transform) de cada metodo tanto no proprio treino interno
(ate 2018-12, referencia) quanto na validacao 2019-2022 (fora da amostra) - a DIFERENCA
entre os dois e o sinal de overfitting da propria reducao (um metodo que reconstroi bem o
treino mas mal a validacao esta capturando ruido amostral, nao estrutura espacial real -
exatamente a preocupacao que motivou reduzir PCA_MAX_COMPONENTS/PLS_MAX_COMPONENTS, ver
src/models/pca_lstm/train.py).

Isso NAO substitui uma avaliacao fim-a-fim (RMSE do LSTM completo, mm/dia) - e um triagem
rapida pra decidir qual(is) metodo(s) vale a pena promover pra um teste completo depois.

Uso (a partir da raiz do repositorio):
    python3 compare_reducao_dimensional.py
    python3 compare_reducao_dimensional.py --var-nao-supervisionada shum_850 --n-components 20,40
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.decomposition import PCA, FastICA, MiniBatchSparsePCA

from src.data import TP_VAR, TRAIN_END, compute_normalization_stats, load_all_datasets, stack_features

VAL_END = "2022-12-01"
PLS_LAG_SHIFT = 1  # mesma convencao de src/models/pca_lstm/train.py (pls_lagged)


def _split_normalizado(ds, var: str, stats: dict, train_end: str, val_end: str):
    media, desvio = stats[var]
    bruto = ds[var].values.astype("float32")
    normalizado = (bruto - media) / desvio
    time_index = pd.DatetimeIndex(ds["time"].values)
    train_mask = time_index <= pd.Timestamp(train_end)
    val_mask = (time_index > pd.Timestamp(train_end)) & (time_index <= pd.Timestamp(val_end))
    flat_train = normalizado[train_mask].reshape(train_mask.sum(), -1)
    flat_val = normalizado[val_mask].reshape(val_mask.sum(), -1)
    return flat_train, flat_val


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def avaliar_nao_supervisionado(nome_var: str, flat_train: np.ndarray, flat_val: np.ndarray, n_components: int) -> list[dict]:
    metodos = {
        "PCA": PCA(n_components=n_components, svd_solver="randomized", random_state=42),
        "ICA": FastICA(n_components=n_components, whiten="unit-variance", max_iter=300, random_state=42),
        "SparsePCA": MiniBatchSparsePCA(n_components=n_components, alpha=1.0, random_state=42, max_iter=300),
    }
    resultados = []
    for nome_metodo, modelo in metodos.items():
        t0 = time.time()
        modelo.fit(flat_train)
        dt = time.time() - t0

        recon_train = modelo.inverse_transform(modelo.transform(flat_train))
        recon_val = modelo.inverse_transform(modelo.transform(flat_val))
        rmse_train = _rmse(flat_train, recon_train)
        rmse_val = _rmse(flat_val, recon_val)

        resultados.append({
            "variavel": nome_var, "metodo": nome_metodo, "n_components": n_components,
            "rmse_reconstrucao_treino": rmse_train, "rmse_reconstrucao_validacao": rmse_val,
            "gap_val_menos_treino": rmse_val - rmse_train, "tempo_fit_s": dt,
        })
        print(f"  {nome_metodo:10s} k={n_components:3d} | RMSE treino={rmse_train:.4f}  "
              f"validacao={rmse_val:.4f}  gap={rmse_val - rmse_train:+.4f}  ({dt:.1f}s)")
    return resultados


def avaliar_supervisionado(nome_var: str, flat_x_train: np.ndarray, flat_x_val: np.ndarray,
                            y_train: np.ndarray, n_components: int) -> list[dict]:
    """PLS vs CCA, ambos usando tp (defasado 1 mes, mesma convencao de pls_lagged) como alvo Y
    no fit - mas a reconstrucao avaliada e a de X (a variavel atmosferica), nao de Y, porque o
    que importa pro pipeline e quanto da variavel atmosferica sobrevive a reducao."""
    metodos = {
        "PLS": PLSRegression(n_components=n_components, scale=False, max_iter=200),
        "CCA": CCA(n_components=n_components, scale=False, max_iter=200),
    }
    resultados = []
    for nome_metodo, modelo in metodos.items():
        t0 = time.time()
        modelo.fit(flat_x_train, y_train)
        dt = time.time() - t0

        recon_train = modelo.inverse_transform(modelo.transform(flat_x_train))
        recon_val = modelo.inverse_transform(modelo.transform(flat_x_val))
        rmse_train = _rmse(flat_x_train, recon_train)
        rmse_val = _rmse(flat_x_val, recon_val)

        resultados.append({
            "variavel": nome_var, "metodo": nome_metodo, "n_components": n_components,
            "rmse_reconstrucao_treino": rmse_train, "rmse_reconstrucao_validacao": rmse_val,
            "gap_val_menos_treino": rmse_val - rmse_train, "tempo_fit_s": dt,
        })
        print(f"  {nome_metodo:10s} k={n_components:3d} | RMSE treino={rmse_train:.4f}  "
              f"validacao={rmse_val:.4f}  gap={rmse_val - rmse_train:+.4f}  ({dt:.1f}s)")
    return resultados


def main(var_nao_supervisionada: str, var_supervisionada: str, n_components_list: list[int]):
    print("=== Carregando dados ===")
    datasets = load_all_datasets()
    ds = stack_features(datasets)
    stats = compute_normalization_stats(ds, train_end=TRAIN_END)

    todos_resultados = []

    print(f"\n{'=' * 70}\nNao supervisionado ({var_nao_supervisionada}): PCA vs ICA vs SparsePCA\n{'=' * 70}")
    flat_train, flat_val = _split_normalizado(ds, var_nao_supervisionada, stats, TRAIN_END, VAL_END)
    print(f"  {flat_train.shape[0]} meses treino | {flat_val.shape[0]} meses validacao | {flat_train.shape[1]} pixels")
    for k in n_components_list:
        todos_resultados += avaliar_nao_supervisionado(var_nao_supervisionada, flat_train, flat_val, k)

    print(f"\n{'=' * 70}\nSupervisionado ({var_supervisionada} vs tp defasado {PLS_LAG_SHIFT}m): PLS vs CCA\n{'=' * 70}")
    media_tp, desvio_tp = stats[TP_VAR]
    tp_normalizado = (ds[TP_VAR].values.astype("float32") - media_tp) / desvio_tp
    time_index = pd.DatetimeIndex(ds["time"].values)
    train_mask = time_index <= pd.Timestamp(TRAIN_END)
    train_idx = np.where(train_mask)[0]
    # Y = tp no mes seguinte (mesmo pls_lag_shift=1 de pls_lagged) - reduzido via PCA(90%) so
    # como alvo de referencia, igual ao que fit_reduction_per_variable faz para o PLS de verdade
    tp_pca_ref = PCA(n_components=0.90, svd_solver="full", random_state=42)
    tp_pca_ref.fit(tp_normalizado[train_idx].reshape(len(train_idx), -1))
    tp_components_full = tp_pca_ref.transform(tp_normalizado.reshape(tp_normalizado.shape[0], -1))
    y_train = tp_components_full[train_idx + PLS_LAG_SHIFT]

    flat_x_train_full, flat_x_val = _split_normalizado(ds, var_supervisionada, stats, TRAIN_END, VAL_END)
    flat_x_train = flat_x_train_full[: len(train_idx) - PLS_LAG_SHIFT] if PLS_LAG_SHIFT else flat_x_train_full
    y_train = y_train[: flat_x_train.shape[0]]
    print(f"  {flat_x_train.shape[0]} meses treino | {flat_x_val.shape[0]} meses validacao | "
          f"Y (tp defasado) com {y_train.shape[1]} colunas")
    for k in n_components_list:
        if k >= y_train.shape[1]:
            print(f"  (pulando k={k}: >= n_colunas de Y, PLS/CCA exigem k < min(n_amostras, n_features_X, n_features_Y))")
            continue
        todos_resultados += avaliar_supervisionado(var_supervisionada, flat_x_train, flat_x_val, y_train, k)

    df = pd.DataFrame(todos_resultados)
    df.to_csv("comparacao_reducao_dimensional.csv", index=False)
    print(f"\n=== Resultado completo salvo em comparacao_reducao_dimensional.csv ===")
    print(df.to_string(index=False))
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--var-nao-supervisionada", default=TP_VAR,
                         help="Variavel para comparar PCA/ICA/SparsePCA (padrao: %(default)s, "
                         "e a que mais preocupa hoje - 40-85 componentes).")
    parser.add_argument("--var-supervisionada", default="geopotential_850",
                         help="Variavel atmosferica para comparar PLS/CCA (padrao: %(default)s, "
                         "a que pior performou no teto atual do PLS - 74.4%% de variancia com 30 componentes).")
    parser.add_argument("--n-components", default="10,20,40",
                         help="Lista de k a testar, separados por virgula (padrao: %(default)s).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_components_list = [int(k) for k in args.n_components.split(",")]
    main(args.var_nao_supervisionada, args.var_supervisionada, n_components_list)
