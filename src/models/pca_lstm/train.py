"""Treina o Modelo A (PCA/EOF ou PLS + LSTM hindcast/forecast) e gera a previsao/submissao.

Uso (a partir da raiz do repositorio):
    python3 -m src.models.pca_lstm.train                            # PCA (padrao)
    python3 -m src.models.pca_lstm.train --reduction pls_concurrent  # PLS contra tp no mesmo mes
    python3 -m src.models.pca_lstm.train --reduction pls_lagged      # PLS contra tp defasado
    python3 -m src.models.pca_lstm.train --lr 5e-4 --hidden-size 256 --dropout 0.3 --run-dir models/meu_run
        # hiperparametros do LSTM (--lr, --hidden-size, --dropout); --run-dir evita
        # sobrescrever outra execucao do mesmo --reduction (ver run_hparam_sweep.py)
    python3 -m src.models.pca_lstm.train --n-jobs 2
        # limita o paralelismo do ajuste das variaveis atmosfericas (padrao: todos os
        # nucleos) - reduza se faltar memoria (ver fit_reduction_per_variable)
    python3 -m src.models.pca_lstm.train --force-refit
        # ignora o cache do ajuste de reducao (models/_reduction_cache/) e reajusta o
        # PCA/PLS do zero - o cache e reaproveitado automaticamente entre execucoes do
        # mesmo --reduction (e --pls-lag-shift), incluindo entre hiperparametros
        # diferentes do LSTM (ver run_hparam_sweep.py), enquanto os .nc de treino e os
        # parametros de reducao nao mudarem (ver load_or_fit_reduction)
    python3 -m src.models.pca_lstm.train --reduction pls_lagged --use-oni-feature \
        --run-dir models/pls_lagged_oni_lstm_run1
        # acrescenta o indice ONI (src/oni.py) como input do decoder do LSTM (mes
        # alvo-1), em vez de so corrigir o resultado depois (ver postprocess_enso.py) -
        # use --run-dir para nao sobrescrever o run sem essa feature

Passos:
    1. Carrega os dados de treino (.nc, cache local do kagglehub) e do teste real.
    2. Normaliza (z-score, estatisticas so do periodo de treino interno) e reduz
       cada variavel espacialmente, via PCA (nao-supervisionado) ou PLS
       (supervisionado, com tp como alvo - ver `--reduction`).
    3. Treina o LSTM hindcast/forecast simulando o contrato origem+lag (Fase 0 do
       PLANO_TRABALHO.md), com split temporal interno (treino/validacao) para
       escolher o numero de epocas por early stopping.
    4. Reporta RMSE/MAE contra os baselines de persistencia e climatologia.
    5. Retreina o modelo final usando todo o historico rotulado (1940-2022) e gera
       a previsao para o teste real (2023-2024), salvando
       `submissions/submission_<reduction>_lstm.csv`.

Cada metodo de reducao salva seus artefatos numa pasta separada em `models/`
(ver RUN_DIRS), entao rodar os tres nao sobrescreve resultados anteriores - basta
comparar os `metrics.json` de cada pasta para decidir qual reducao performou melhor.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baseline import climatology_baseline, persistence_baseline
from download_data import download_competition_data
from src.data import (
    ALL_VARS,
    FEATURE_VARS,
    HINDCAST_LEN,
    LAGS,
    TP_VAR,
    TRAIN_END,
    TRAIN_FILES,
    build_examples,
    compute_normalization_stats,
    load_all_datasets,
    stack_features,
)
from src.evaluate import evaluate_predictions
from src.models.pca_lstm import HindcastForecastLSTM, SpatialPCA, SpatialPLS
from src.oni import oni_para_datas, oni_series_or_nan
from src.submit import build_submission

VARIANCE_THRESHOLD = 0.90  # criterio de contribuicao minima: mantem o menor n_components que atinja isso
# PCA_MAX_COMPONENTS e PLS_MAX_COMPONENTS reduzidos (200->40, 30->15 - ver conversa) depois de
# observar que, com os tetos antigos, tp sozinho chegava a 85 componentes pra bater 90% de
# variancia e 6 das 9 variaveis atmosfericas batiam no teto de 30 sem sequer atingir 90% - alta
# dimensionalidade total (~337 features de hindcast) que provavelmente prejudicava a generalizacao
# (o modelo promovido dava early stopping ja na epoca 1-2) alem de deixar o fit do PLS bem mais
# lento. Tetos menores tambem deixam mais folga pra somar novas variaveis atmosfericas no futuro
# sem a dimensionalidade total explodir ainda mais.
PCA_MAX_COMPONENTS = 40  # teto de busca para SpatialPCA (barato: SVD randomizada, nao iterativo)
# Teto de busca (binaria) e max_iter por fit para SpatialPLS - mantidos baixos porque o alvo Y (os
# componentes de tp) e multi-coluna (agora ate 40, com o teto do PCA acima), o que deixa cada fit
# do PLS bem mais caro que um alvo escalar (Y multi-coluna exige mais iteracoes do NIPALS para
# convergir); um teto de 100 chegou a deixar um unico fit rodando por horas - ver conversa.
PLS_MAX_COMPONENTS = 15
PLS_MAX_ITER = 100
# Paralelismo do ajuste das variaveis atmosfericas (Passo 2): -1 = usa todos os nucleos
# disponiveis (convencao do joblib). Cada worker e um PROCESSO separado (nao thread) -
# o trabalho e numpy/sklearn puro (CPU-bound), entao processos aproveitam nucleos de
# verdade em vez de esbarrar no GIL. Ajustavel via --n-jobs se a memoria for um problema
# (cada worker mantem sua propria copia da grade normalizada, ~313MB em float32).
N_JOBS_REDUCTION = -1
HIDDEN_SIZE = 128
DROPOUT = 0.1
LR = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 25
PATIENCE = 5
SEED = 42  # mesma seed do random_state do SpatialPCA, por consistencia
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REDUCTION_METHODS = ("pca", "pls_concurrent", "pls_lagged")
PLS_LAG_SHIFT = 1  # meses - so usado por "pls_lagged" (Y = tp em t+PLS_LAG_SHIFT)

RUN_DIRS = {
    "pca": "models/pca_lstm_run1",
    "pls_concurrent": "models/pls_concurrent_lstm_run1",
    "pls_lagged": "models/pls_lagged_lstm_run1",
}

# Cache do ajuste de reducao (Passo 2 - a etapa cara/propensa a OOM: SVD do PCA, NIPALS do
# PLS). O reduction_objects/component_series ajustado so depende do metodo de reducao, do
# pls_lag_shift, dos dados de treino e dos hiperparametros de reducao (VARIANCE_THRESHOLD etc.)
# - nunca dos hiperparametros do LSTM (--hidden-size/--dropout/--lr) nem do --run-dir. Isso
# significa que rodar o mesmo metodo varias vezes (ex.: run_hparam_sweep.py testando varios
# hiperparametros do LSTM) refazia esse ajuste caro do zero a cada execucao, sem necessidade -
# ver conversa. O cache fica fora de RUN_DIRS (que e por --run-dir) justamente para ser
# compartilhado entre execucoes diferentes do mesmo metodo.
REDUCTION_CACHE_DIR = "models/_reduction_cache"


def _reduction_cache_path(method: str, pls_lag_shift: int) -> str:
    nome = method if method != "pls_lagged" else f"pls_lagged_shift{pls_lag_shift}"
    return f"{REDUCTION_CACHE_DIR}/{nome}.joblib"


def _reduction_cache_config(method: str, pls_lag_shift: int, train_end_idx: int) -> dict:
    """Parametros que, se mudarem, invalidam o cache (o ajuste teria saido diferente)."""
    return {
        "method": method,
        "pls_lag_shift": pls_lag_shift if method == "pls_lagged" else None,
        "train_end_idx": train_end_idx,
        "variance_threshold": VARIANCE_THRESHOLD,
        "pca_max_components": PCA_MAX_COMPONENTS,
        "pls_max_components": PLS_MAX_COMPONENTS,
        "pls_max_iter": PLS_MAX_ITER,
    }


def _data_fingerprint(competition_path: str) -> dict[str, float]:
    """mtime de cada .nc de treino usado no ajuste - detecta dado trocado/atualizado mesmo
    se os parametros de configuracao (_reduction_cache_config) nao mudaram."""
    from src.data import TRAIN_FILES

    fingerprint = {}
    for key in TRAIN_FILES.values():
        caminho = os.path.join(competition_path, f"{key}.nc")
        if os.path.exists(caminho):
            fingerprint[key] = os.path.getmtime(caminho)
    return fingerprint


def load_or_fit_reduction(
    ds: xr.Dataset,
    stats: dict,
    train_end_idx: int,
    method: str,
    pls_lag_shift: int = PLS_LAG_SHIFT,
    n_jobs: int = N_JOBS_REDUCTION,
    force_refit: bool = False,
) -> tuple[dict, dict]:
    """Reusa o cache do ajuste de reducao (ver REDUCTION_CACHE_DIR) se ele bater com a
    configuracao atual e os dados de treino nao tiverem mudado; caso contrario, ajusta do
    zero via fit_reduction_per_variable e salva o resultado no cache para as proximas.

    O cache guarda so `reduction_objects` + `stats` (nao `component_series`): num acerto,
    so o `.transform()` (barato, sem SVD/NIPALS) roda de novo em cima do `ds` que o chamador
    ja tem em memoria (Passo 1 sempre carrega os dados antes de chegar aqui) - isso tambem
    permite popular o cache retroativamente a partir de um `reduction_and_stats.joblib` ja
    salvo por uma execucao anterior (mesmo formato), sem precisar re-carregar os .nc."""
    cache_path = _reduction_cache_path(method, pls_lag_shift)
    config_atual = _reduction_cache_config(method, pls_lag_shift, train_end_idx)
    fingerprint_atual = _data_fingerprint(download_competition_data())

    if not force_refit and os.path.exists(cache_path):
        cache = joblib.load(cache_path)
        if cache.get("config") == config_atual and cache.get("data_fingerprint") == fingerprint_atual:
            print(
                f"  cache de reducao reaproveitado ({cache_path}, calculado em "
                f"{cache.get('saved_at', '?')}) - pulando o ajuste do PCA/PLS"
            )
            reduction_objects = cache["reduction_objects"]
            # so o .transform() (barato) precisa rodar de novo - o ds ja esta em memoria
            # de qualquer forma (Passo 1), entao isso nao volta a tocar disco/rede.
            component_series = {
                var: reduction_objects[var]
                .transform(((ds[var].values.astype("float32") - cache["stats"][var][0]) / cache["stats"][var][1]))
                .astype("float32")
                for var in reduction_objects
            }
            return reduction_objects, component_series
        motivo = "config diferente" if cache.get("config") != config_atual else "dados de treino mudaram"
        print(
            f"  cache de reducao em {cache_path} invalido ({motivo}; calculado em "
            f"{cache.get('saved_at', '?')}) - recalculando"
        )
    elif force_refit and os.path.exists(cache_path):
        print(f"  --force-refit: ignorando cache existente em {cache_path}")

    reduction_objects, component_series = fit_reduction_per_variable(
        ds, stats, train_end_idx, method=method, pls_lag_shift=pls_lag_shift, n_jobs=n_jobs
    )

    os.makedirs(REDUCTION_CACHE_DIR, exist_ok=True)
    joblib.dump(
        {
            "reduction_objects": reduction_objects,
            "stats": stats,
            "config": config_atual,
            "data_fingerprint": fingerprint_atual,
            "saved_at": pd.Timestamp.now().isoformat(),
        },
        cache_path,
    )
    print(f"  cache de reducao salvo em {cache_path}")

    return reduction_objects, component_series


class _Tee:
    """Escreve simultaneamente em varios streams (usado para logar no console e em arquivo)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def _fit_one_atm_variable(
    var: str,
    normalizado: np.ndarray,
    method: str,
    train_end_idx: int,
    pls_lag_shift: int,
    tp_components_full: np.ndarray,
) -> tuple[str, SpatialPCA | SpatialPLS, np.ndarray, str]:
    """Ajusta e transforma uma variavel atmosferica - e o "worker" despachado em paralelo
    por `fit_reduction_per_variable` (um processo por variavel, via joblib).

    Recebe so arrays numpy simples (nunca o `xr.Dataset`/`ds`): objetos xarray com backend
    lazy de netCDF nao sao seguros/eficientes de serializar entre processos, entao a leitura
    do disco (`ds[var].values`) e a normalizacao continuam no processo principal - so a parte
    cara (fit do PCA/PLS) e que roda em paralelo. A mensagem de log e retornada (em vez de
    impressa aqui) porque um `print` dentro do worker nao passa pelo `_Tee`/`train.log` do
    processo principal.
    """
    if method == "pca":
        reducer = SpatialPCA(variance_threshold=VARIANCE_THRESHOLD, max_components=PCA_MAX_COMPONENTS)
        reducer.fit(normalizado[: train_end_idx + 1])
    else:
        shift = pls_lag_shift if method == "pls_lagged" else 0
        x_fit = normalizado[: train_end_idx + 1 - shift]
        y_fit = tp_components_full[shift : train_end_idx + 1]
        reducer = SpatialPLS(
            variance_threshold=VARIANCE_THRESHOLD, max_components=PLS_MAX_COMPONENTS, max_iter=PLS_MAX_ITER
        )
        reducer.fit(x_fit, y_fit)

    componentes = reducer.transform(normalizado).astype("float32")
    log_msg = (
        f"  {method.upper()}[{var}]: {reducer.n_components_} componentes explicam "
        f"{reducer.explained_variance_ratio():.1%} da variancia de X (treino interno)"
    )
    return var, reducer, componentes, log_msg


def fit_reduction_per_variable(
    ds: xr.Dataset,
    stats: dict,
    train_end_idx: int,
    method: str = "pca",
    pls_lag_shift: int = PLS_LAG_SHIFT,
    n_jobs: int = N_JOBS_REDUCTION,
) -> tuple[dict, dict]:
    """Normaliza e reduz cada variavel (ajustado so no periodo de treino interno).

    `tp` e sempre reduzido via PCA (nao-supervisionado): serve como reducao final
    quando method="pca" e, quando method comeca com "pls_", como o alvo Y (serie de
    referencia de precipitacao) usado para ajustar o PLS das variaveis atmosfericas.
    Em "pls_concurrent" o alvo e tp no mesmo mes t; em "pls_lagged" e tp em
    t+pls_lag_shift (o PLS busca a parte de cada variavel mais ligada a
    precipitacao futura, nao so a de maior variancia espacial).

    `tp` e ajustado primeiro, sequencialmente (as variantes PLS dependem dos
    componentes dele como alvo Y). As `FEATURE_VARS` sao independentes entre si -
    o ajuste de cada uma roda em paralelo (`n_jobs` processos via joblib,
    `_fit_one_atm_variable`), o que passa a valer a pena conforme mais dados
    entrarem no pipeline (mais variaveis e/ou grades maiores).

    O despacho e "preguicoso" (generator + `pre_dispatch="n_jobs"`): a leitura/
    normalizacao de cada variavel so acontece pouco antes dela ser enviada a um
    worker livre, entao no maximo ~n_jobs grades normalizadas ficam na memoria ao
    mesmo tempo (nao as 9 de uma vez) - grade tem 301x261 pontos, ~313MB em
    float32 por variavel completa.
    """
    assert method in REDUCTION_METHODS, f"method invalido: {method}"

    reduction_objects: dict[str, SpatialPCA | SpatialPLS] = {}
    component_series: dict[str, np.ndarray] = {}

    media_tp, desvio_tp = stats[TP_VAR]
    bruto_tp = ds[TP_VAR].values.astype("float32")
    normalizado_tp = (bruto_tp - media_tp) / desvio_tp
    del bruto_tp

    pca_tp = SpatialPCA(variance_threshold=VARIANCE_THRESHOLD, max_components=PCA_MAX_COMPONENTS)
    pca_tp.fit(normalizado_tp[: train_end_idx + 1])
    print(
        f"  PCA[{TP_VAR}] (referencia): {pca_tp.n_components_} componentes explicam "
        f"{pca_tp.explained_variance_ratio():.1%} da variancia (treino interno)"
    )
    tp_components_full = pca_tp.transform(normalizado_tp).astype("float32")
    del normalizado_tp

    reduction_objects[TP_VAR] = pca_tp
    component_series[TP_VAR] = tp_components_full

    def _tarefas():
        for var in FEATURE_VARS:
            media, desvio = stats[var]
            bruto = ds[var].values.astype("float32")
            normalizado = (bruto - media) / desvio
            del bruto
            yield joblib.delayed(_fit_one_atm_variable)(
                var, normalizado, method, train_end_idx, pls_lag_shift, tp_components_full
            )

    resultados = joblib.Parallel(n_jobs=n_jobs, pre_dispatch="n_jobs")(_tarefas())

    for var, reducer, componentes, log_msg in resultados:
        print(log_msg)
        reduction_objects[var] = reducer
        component_series[var] = componentes

    return reduction_objects, component_series


def make_loader(hindcast, alvo_atm, tp_congelado, lag, y, batch_size, shuffle, oni=None):
    tensors = [
        torch.from_numpy(hindcast),
        torch.from_numpy(alvo_atm),
        torch.from_numpy(tp_congelado),
        torch.from_numpy(lag),
        torch.from_numpy(y),
    ]
    if oni is not None:
        tensors.append(torch.from_numpy(oni))
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _unpack_batch(batch, n_oni_features: int, device: torch.device):
    """Desempacota um batch do DataLoader (5 tensores, ou 6 se `--use-oni-feature`
    - ver make_loader) e move tudo para `device`. `oni` volta None quando
    n_oni_features == 0 (repassado direto para HindcastForecastLSTM.forward)."""
    if n_oni_features > 0:
        hindcast, alvo_atm, tp_congelado, lag, y, oni = batch
        oni = oni.to(device)
    else:
        hindcast, alvo_atm, tp_congelado, lag, y = batch
        oni = None
    return hindcast.to(device), alvo_atm.to(device), tp_congelado.to(device), lag.to(device), y.to(device), oni


def train_model(
    train_loader,
    val_loader,
    n_features_hindcast,
    n_features_atm,
    n_components_tp,
    max_epochs,
    patience,
    run_dir,
    hidden_size=HIDDEN_SIZE,
    dropout=DROPOUT,
    lr=LR,
    n_oni_features=0,
):
    model = HindcastForecastLSTM(
        n_features_hindcast=n_features_hindcast,
        n_features_atm=n_features_atm,
        n_components_tp=n_components_tp,
        hidden_size=hidden_size,
        dropout=dropout,
        n_oni_features=n_oni_features,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_sem_melhora = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            hindcast, alvo_atm, tp_congelado, lag, y, oni = _unpack_batch(batch, n_oni_features, DEVICE)

            optimizer.zero_grad()
            pred = model(hindcast, alvo_atm, tp_congelado, lag, oni=oni)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * hindcast.size(0)
            n_train += hindcast.size(0)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                hindcast, alvo_atm, tp_congelado, lag, y, oni = _unpack_batch(batch, n_oni_features, DEVICE)
                pred = model(hindcast, alvo_atm, tp_congelado, lag, oni=oni)
                loss = criterion(pred, y)
                val_loss += loss.item() * hindcast.size(0)
                n_val += hindcast.size(0)
        val_loss /= n_val

        print(f"  epoca {epoch:2d}/{max_epochs} | treino MSE(pca) {train_loss:.4f} | val MSE(pca) {val_loss:.4f}")
        history.append({"epoch": epoch, "train_mse_pca": train_loss, "val_mse_pca": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_sem_melhora = 0
        else:
            epochs_sem_melhora += 1

        # Backup por epoca: se o processo cair, o progresso ate aqui nao se perde
        # (nao ha retomada automatica - serve so para nao ter que rodar tudo de novo do zero).
        pd.DataFrame(history).to_csv(f"{run_dir}/history.csv", index=False)
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "epochs_sem_melhora": epochs_sem_melhora,
            },
            f"{run_dir}/checkpoint_selecao_epocas.pt",
        )

        if epochs_sem_melhora >= patience:
            print(f"  early stopping na epoca {epoch} (melhor: {best_epoch})")
            break

    model.load_state_dict(best_state)
    # O loop acima sobrescreve checkpoint_selecao_epocas.pt a cada epoca com os pesos
    # ATUAIS (so um backup de recuperacao contra crash - ver comentario acima), entao ao
    # sair do loop o arquivo em disco tem os pesos da ULTIMA epoca rodada (epoch), nao da
    # melhor (best_epoch) - a unica coisa que recebe best_state e o objeto `model` em
    # memoria, via load_state_dict logo acima. Sem este save final, qualquer script que
    # carregue checkpoint_selecao_epocas.pt depois do treino (postprocess_enso.py,
    # enso_correction_cv.py) avalia a epoca errada - ver conversa (RMSE deu 1.8548 em vez
    # de 1.8405 no checkpoint promovido, porque carregava a epoca 8 em vez da 3).
    torch.save(
        {
            "epoch": best_epoch,
            "model_state": best_state,
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_sem_melhora": epochs_sem_melhora,
        },
        f"{run_dir}/checkpoint_selecao_epocas.pt",
    )
    return model, best_epoch, best_val_loss, history


def reconstruct_tp(pca_tp: SpatialPCA, stats_tp: tuple[float, float], coeffs_norm_pca: np.ndarray) -> np.ndarray:
    media, desvio = stats_tp
    grid_normalizado = pca_tp.inverse_transform(coeffs_norm_pca)
    return grid_normalizado * desvio + media


def main(
    method: str = "pca",
    pls_lag_shift: int = PLS_LAG_SHIFT,
    run_dir: str | None = None,
    hidden_size: int = HIDDEN_SIZE,
    dropout: float = DROPOUT,
    lr: float = LR,
    n_jobs: int = N_JOBS_REDUCTION,
    seed: int = SEED,
    force_refit: bool = False,
    use_oni_feature: bool = False,
):
    """Ponto de entrada publico: prepara a pasta do run e loga tudo (console + arquivo)
    em `{run_dir}/train.log`, alem de delegar o treino de fato para `_train`.

    `run_dir` e opcional: por padrao usa RUN_DIRS[method] (um run por metodo de
    reducao), mas pode ser sobrescrito para nao colidir quando varias execucoes do
    mesmo metodo rodam com hiperparametros diferentes (ver run_hparam_sweep.py).
    `n_jobs` controla o paralelismo do ajuste das variaveis atmosfericas (Passo 2) -
    ver fit_reduction_per_variable. `seed` fixa a inicializacao dos pesos do LSTM e o
    shuffle do DataLoader (nao fixados antes - cada rodada dava um RMSE levemente
    diferente mesmo com os mesmos dados/hiperparametros). `force_refit` ignora o
    cache do ajuste de reducao (ver load_or_fit_reduction/REDUCTION_CACHE_DIR) -
    use se os dados de treino mudaram sem que o mtime dos .nc tenha mudado (ex.:
    substituicao manual preservando timestamp) ou para depurar o proprio cache.
    `use_oni_feature` acrescenta o indice ONI (ver src/oni.py) como input do decoder
    (mes alvo-1, mesmo alinhamento de `alvo_atm`) - testa a hipotese de que dar ENSO
    como entrada real supera so corrigir o resultado depois (ver postprocess_enso.py).
    Origens de treino antes de 1950 (sem ONI publicado) sao descartadas nesse modo -
    ver src.data.build_examples."""
    assert method in REDUCTION_METHODS, f"method invalido: {method} (esperado um de {REDUCTION_METHODS})"
    run_dir = run_dir or RUN_DIRS[method]
    os.makedirs(run_dir, exist_ok=True)

    log_path = f"{run_dir}/train.log"
    with open(log_path, "a") as log_file, contextlib.redirect_stdout(_Tee(sys.stdout, log_file)):
        print(f"\n{'=' * 70}\nnova execucao ({method}) em {pd.Timestamp.now()}\n{'=' * 70}")
        return _train(
            method, pls_lag_shift, run_dir, hidden_size, dropout, lr, n_jobs, seed, force_refit, use_oni_feature
        )


def _train(
    method: str,
    pls_lag_shift: int,
    run_dir: str,
    hidden_size: int = HIDDEN_SIZE,
    dropout: float = DROPOUT,
    lr: float = LR,
    n_jobs: int = N_JOBS_REDUCTION,
    seed: int = SEED,
    force_refit: bool = False,
    use_oni_feature: bool = False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    label_modelo = {
        "pca": "PCA+LSTM",
        "pls_concurrent": "PLS(concorrente)+LSTM",
        "pls_lagged": "PLS(defasado)+LSTM",
    }[method]

    n_oni_features = 1 if use_oni_feature else 0
    print(
        f"=== 0. Metodo de reducao dimensional: {method} | seed: {seed} | "
        f"hiperparametros: hidden_size={hidden_size} dropout={dropout} lr={lr} | "
        f"use_oni_feature={use_oni_feature} ==="
    )

    print("=== 1. Carregando dados ===")
    datasets = load_all_datasets()
    ds = stack_features(datasets)
    time_index = pd.DatetimeIndex(ds["time"].values)
    train_end_idx = time_index.get_indexer([pd.Timestamp(TRAIN_END)])[0]
    last_valid_idx = len(time_index) - 1
    print(f"  {len(time_index)} meses ({time_index[0].date()} a {time_index[-1].date()})")
    print(f"  origens de treino interno ate indice {train_end_idx} ({time_index[train_end_idx].date()})")
    print(f"  origens de validacao interna ate indice {last_valid_idx} ({time_index[last_valid_idx].date()})")

    oni_full = oni_series_or_nan(time_index) if use_oni_feature else None
    if use_oni_feature:
        n_sem_oni = int(np.isnan(oni_full).sum())
        print(f"  ONI: {n_sem_oni} meses sem indice disponivel (antes de 1950-01) - origens nesses meses serao descartadas")

    stats = compute_normalization_stats(ds, train_end=TRAIN_END)

    print(f"\n=== 2. Reduzindo dimensionalidade por variavel ({method}, so no treino interno) ===")
    tp_raw = ds[TP_VAR].values.astype("float32").copy()
    reduction_objects, component_series = load_or_fit_reduction(
        ds, stats, train_end_idx, method=method, pls_lag_shift=pls_lag_shift, n_jobs=n_jobs,
        force_refit=force_refit,
    )

    n_components_tp = reduction_objects[TP_VAR].n_components_
    n_features_hindcast = sum(reduction_objects[v].n_components_ for v in ALL_VARS)
    n_features_atm = sum(reduction_objects[v].n_components_ for v in FEATURE_VARS)

    # As 9 variaveis atmosfericas ja viraram component_series (pequeno) - a grade bruta delas
    # nao e mais usada no resto da funcao (so tp, via ds[TP_VAR], no calculo da climatologia
    # do Passo 5). Soltar essa memoria agora (nao no fim da funcao) reduz o pico sustentado
    # durante o treino do LSTM/retreino, que sao os passos mais longos - ver conversa sobre
    # runs presos por pouca memoria.
    for var in FEATURE_VARS:
        if var in ds.data_vars:
            del ds[var]
        datasets.pop(TRAIN_FILES[var], None)
    gc.collect()

    print("\n=== 3. Montando exemplos (contrato origem + lag) ===")
    train_ex = build_examples(component_series, 0, train_end_idx, last_valid_idx=train_end_idx, oni_series=oni_full)
    val_ex = build_examples(
        component_series, train_end_idx + 1, last_valid_idx, last_valid_idx=last_valid_idx, oni_series=oni_full
    )
    if use_oni_feature:
        hindcast_tr, atm_tr, tp_froz_tr, lag_tr, y_tr, origin_tr, alvo_tr, oni_tr = train_ex
        hindcast_va, atm_va, tp_froz_va, lag_va, y_va, origin_va, alvo_va, oni_va = val_ex
    else:
        hindcast_tr, atm_tr, tp_froz_tr, lag_tr, y_tr, origin_tr, alvo_tr = train_ex
        hindcast_va, atm_va, tp_froz_va, lag_va, y_va, origin_va, alvo_va = val_ex
        oni_tr = oni_va = None
    print(f"  treino: {len(y_tr)} exemplos | validacao: {len(y_va)} exemplos")

    train_loader = make_loader(hindcast_tr, atm_tr, tp_froz_tr, lag_tr, y_tr, BATCH_SIZE, shuffle=True, oni=oni_tr)
    val_loader = make_loader(hindcast_va, atm_va, tp_froz_va, lag_va, y_va, BATCH_SIZE, shuffle=False, oni=oni_va)

    print("\n=== 4. Treinando (selecao de epocas via validacao interna) ===")
    t0 = time.time()
    model, best_epoch, best_val_loss, history = train_model(
        train_loader,
        val_loader,
        n_features_hindcast,
        n_features_atm,
        n_components_tp,
        MAX_EPOCHS,
        PATIENCE,
        run_dir,
        hidden_size=hidden_size,
        dropout=dropout,
        lr=lr,
        n_oni_features=n_oni_features,
    )
    print(f"  treino concluido em {time.time() - t0:.0f}s | melhor epoca: {best_epoch} | val MSE(pca): {best_val_loss:.4f}")

    print("\n=== 5. Avaliando contra baselines (grade real, mm/dia) ===")
    model.eval()
    with torch.no_grad():
        pred_pca_va = model(
            torch.from_numpy(hindcast_va).to(DEVICE),
            torch.from_numpy(atm_va).to(DEVICE),
            torch.from_numpy(tp_froz_va).to(DEVICE),
            torch.from_numpy(lag_va).to(DEVICE),
            oni=torch.from_numpy(oni_va).to(DEVICE) if use_oni_feature else None,
        ).cpu().numpy()

    pred_grid = reconstruct_tp(reduction_objects[TP_VAR], stats[TP_VAR], pred_pca_va)
    true_grid = tp_raw[alvo_va]
    persist_grid = persistence_baseline(tp_raw[origin_va])
    meses_alvo = time_index.month.values[alvo_va]
    clim_grid = climatology_baseline(ds[TP_VAR].sel(time=slice(None, TRAIN_END)), list(meses_alvo))

    lag_inteiro_va = (lag_va * max(LAGS)).round().astype(int)
    metrics_model = evaluate_predictions(true_grid, pred_grid, lags=lag_inteiro_va)
    metrics_persist = evaluate_predictions(true_grid, persist_grid, lags=lag_inteiro_va)
    metrics_clim = evaluate_predictions(true_grid, clim_grid, lags=lag_inteiro_va)

    print(f"  {label_modelo:<22}RMSE={metrics_model['rmse']:.3f}  MAE={metrics_model['mae']:.3f}  (mm/dia)")
    print(f"  Persistencia  RMSE={metrics_persist['rmse']:.3f}  MAE={metrics_persist['mae']:.3f}  (mm/dia)")
    print(f"  Climatologia  RMSE={metrics_clim['rmse']:.3f}  MAE={metrics_clim['mae']:.3f}  (mm/dia)")

    print("\n=== 5b. Salvando artefatos para visualizacao (models/) ===")
    os.makedirs(run_dir, exist_ok=True)

    pd.DataFrame(history).to_csv(f"{run_dir}/history.csv", index=False)

    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump(
            {
                "reduction_method": method,
                "hyperparams": {
                    "hidden_size": hidden_size,
                    "dropout": dropout,
                    "lr": lr,
                    "use_oni_feature": use_oni_feature,
                },
                "best_epoch": best_epoch,
                "best_val_loss_pca": best_val_loss,
                "modelo": metrics_model,
                "persistencia": metrics_persist,
                "climatologia": metrics_clim,
            },
            f,
            indent=2,
        )

    # Grades de amostra (para mapas espaciais) - uma por lag representativo
    lags_amostra = [l for l in [1, 3, 6, 12, 18, 24] if l in lag_inteiro_va]
    amostras = {"lat": ds["lat"].values, "lon": ds["lon"].values, "lags": np.array(lags_amostra)}
    for nome, grade in [("true", true_grid), ("pred", pred_grid), ("persist", persist_grid), ("clim", clim_grid)]:
        escolhidas = [grade[np.where(lag_inteiro_va == l)[0][0]] for l in lags_amostra]
        amostras[nome] = np.stack(escolhidas)
    amostras["origin_date"] = np.array(
        [str(time_index[origin_va[np.where(lag_inteiro_va == l)[0][0]]].date()) for l in lags_amostra]
    )
    amostras["target_date"] = np.array(
        [str(time_index[alvo_va[np.where(lag_inteiro_va == l)[0][0]]].date()) for l in lags_amostra]
    )
    np.savez_compressed(f"{run_dir}/sample_grids.npz", **amostras)
    print(f"  history.csv, metrics.json e sample_grids.npz salvos em {run_dir}/")

    print("\n=== 6. Retreinando com todo o historico rotulado (1940-2022) ===")
    full_ex = build_examples(component_series, 0, last_valid_idx, last_valid_idx=last_valid_idx, oni_series=oni_full)
    if use_oni_feature:
        hindcast_full, atm_full, tp_froz_full, lag_full, y_full, _, _, oni_full_ex = full_ex
    else:
        hindcast_full, atm_full, tp_froz_full, lag_full, y_full, _, _ = full_ex
        oni_full_ex = None
    full_loader = make_loader(
        hindcast_full, atm_full, tp_froz_full, lag_full, y_full, BATCH_SIZE, shuffle=True, oni=oni_full_ex
    )

    final_model = HindcastForecastLSTM(
        n_features_hindcast=n_features_hindcast,
        n_features_atm=n_features_atm,
        n_components_tp=n_components_tp,
        hidden_size=hidden_size,
        dropout=dropout,
        n_oni_features=n_oni_features,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    retrain_history = []
    for epoch in range(1, best_epoch + 1):
        final_model.train()
        epoch_loss, n_seen = 0.0, 0
        for batch in full_loader:
            hindcast, alvo_atm, tp_congelado, lag, y, oni = _unpack_batch(batch, n_oni_features, DEVICE)
            optimizer.zero_grad()
            pred = final_model(hindcast, alvo_atm, tp_congelado, lag, oni=oni)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * hindcast.size(0)
            n_seen += hindcast.size(0)
        epoch_loss /= n_seen
        print(f"  epoca {epoch:2d}/{best_epoch} | MSE(pca) {epoch_loss:.4f}")

        # Backup por epoca (mesma logica do checkpoint de selecao de epocas em train_model).
        retrain_history.append({"epoch": epoch, "train_mse_pca": epoch_loss})
        pd.DataFrame(retrain_history).to_csv(f"{run_dir}/retrain_history.csv", index=False)
        torch.save(
            {
                "epoch": epoch,
                "model_state": final_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            f"{run_dir}/checkpoint_retrain_final.pt",
        )

    torch.save(final_model.state_dict(), f"{run_dir}/model_final.pt")
    joblib.dump(
        {
            "reduction_objects": reduction_objects,
            "method": method,
            "stats": stats,
            "n_oni_features": n_oni_features,
        },
        f"{run_dir}/reduction_and_stats.joblib",
    )
    print(f"  modelo final e objetos de reducao salvos em {run_dir}/")

    print("\n=== 7. Prevendo o teste real (2023-2024) ===")
    teste_ds = datasets["teste_features"]
    origem_idx = last_valid_idx  # dez/2022, mesmo mes usado como "tp_ultima_obs" no teste
    hindcast_teste = np.concatenate(
        [component_series[v][origem_idx - HINDCAST_LEN + 1 : origem_idx + 1] for v in ALL_VARS], axis=1
    )
    tp_congelado_teste = component_series[TP_VAR][origem_idx]

    n_meses_teste = teste_ds.sizes["time"]
    atm_teste_list = []
    for var in FEATURE_VARS:
        media, desvio = stats[var]
        bruto = teste_ds[var].values.astype("float32")
        normalizado = (bruto - media) / desvio
        coeffs = reduction_objects[var].transform(normalizado)
        atm_teste_list.append(coeffs)
    atm_teste = np.concatenate(atm_teste_list, axis=1).astype("float32")  # (n_meses_teste, n_features_atm)

    hindcast_batch = np.repeat(hindcast_teste[None, :, :], n_meses_teste, axis=0)
    tp_congelado_batch = np.repeat(tp_congelado_teste[None, :], n_meses_teste, axis=0)
    lag_batch = (teste_ds["lag_meses"].values / max(LAGS)).astype("float32")

    oni_teste_batch = None
    if use_oni_feature:
        # mesmo alinhamento anti-vazamento do treino (feature_idx = alvo_idx - 1):
        # ONI do mes anterior a cada mes-alvo do teste. oni_para_datas (nao
        # oni_series_or_nan) porque 2023-2024 esta sempre dentro da cobertura da
        # tabela (1950-2024) - falta aqui seria um bug, nao um caso esperado.
        datas_alvo_teste = pd.DatetimeIndex(teste_ds["time"].values) - pd.DateOffset(months=1)
        oni_teste_batch = oni_para_datas(datas_alvo_teste).reshape(-1, 1)

    final_model.eval()
    with torch.no_grad():
        pred_pca_teste = final_model(
            torch.from_numpy(hindcast_batch).to(DEVICE),
            torch.from_numpy(atm_teste).to(DEVICE),
            torch.from_numpy(tp_congelado_batch).to(DEVICE),
            torch.from_numpy(lag_batch).to(DEVICE),
            oni=torch.from_numpy(oni_teste_batch).to(DEVICE) if use_oni_feature else None,
        ).cpu().numpy()

    pred_grid_teste = reconstruct_tp(reduction_objects[TP_VAR], stats[TP_VAR], pred_pca_teste)
    pred_grid_teste = np.clip(pred_grid_teste, 0, None)  # precipitacao nao e negativa

    predictions_da = xr.DataArray(
        pred_grid_teste,
        dims=("time", "lat", "lon"),
        coords={"time": teste_ds["time"].values, "lat": teste_ds["lat"].values, "lon": teste_ds["lon"].values},
    )

    print("\n=== 8. Gerando submissao ===")
    os.makedirs("submissions", exist_ok=True)
    competition_path = download_competition_data()
    sample_path = os.path.join(competition_path, "sample_submission.csv")
    # Tag da submissao: nome do metodo no caso padrao, ou o nome da pasta do run quando
    # run_dir foi sobrescrito (ex.: varios hiperparametros do mesmo metodo, ver run_hparam_sweep.py) -
    # evita colidir arquivos de submissao entre execucoes diferentes.
    run_tag = method if run_dir == RUN_DIRS.get(method) else os.path.basename(os.path.normpath(run_dir))
    output_path = f"submissions/submission_{run_tag}_lstm.csv"
    submission_df = build_submission(predictions_da, sample_path, output_path)
    print(f"  salvo em {output_path} ({len(submission_df)} linhas)")

    return predictions_da, submission_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reduction",
        choices=REDUCTION_METHODS,
        default="pca",
        help=(
            "Metodo de reducao dimensional por variavel: 'pca' (nao-supervisionado, "
            "padrao), 'pls_concurrent' (PLS contra tp no mesmo mes) ou 'pls_lagged' "
            "(PLS contra tp defasado, ver --pls-lag-shift)."
        ),
    )
    parser.add_argument(
        "--pls-lag-shift",
        type=int,
        default=PLS_LAG_SHIFT,
        help="Meses de defasagem entre X e o alvo tp para --reduction pls_lagged (padrao: %(default)s).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Pasta de saida dos artefatos (padrao: RUN_DIRS[--reduction]). Use para nao sobrescrever "
        "outra execucao do mesmo metodo com hiperparametros diferentes (ver run_hparam_sweep.py).",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=HIDDEN_SIZE, help="Tamanho do hidden state do LSTM (padrao: %(default)s)."
    )
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout do decoder (padrao: %(default)s).")
    parser.add_argument("--lr", type=float, default=LR, help="Taxa de aprendizado do Adam (padrao: %(default)s).")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS_REDUCTION,
        help="Processos paralelos para ajustar as variaveis atmosfericas no Passo 2 (padrao: "
        "%(default)s = todos os nucleos, convencao do joblib). Reduza se faltar memoria "
        "(cada worker mantem sua propria copia da grade normalizada, ~313MB).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed do torch/numpy (inicializacao dos pesos do LSTM e shuffle do DataLoader), "
        "para o treino ser reprodutivel entre execucoes (padrao: %(default)s).",
    )
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Ignora o cache do ajuste de reducao (ver REDUCTION_CACHE_DIR) e reajusta o "
        "PCA/PLS do zero mesmo se ja houver um cache valido para esse metodo/pls-lag-shift.",
    )
    parser.add_argument(
        "--use-oni-feature",
        action="store_true",
        help="Acrescenta o indice ONI (ver src/oni.py) como input do decoder do LSTM, "
        "alinhado como as demais variaveis atmosfericas (mes alvo-1). Testa dar ENSO "
        "como entrada real do modelo em vez de so corrigir o resultado depois (ver "
        "postprocess_enso.py). Origens de treino antes de 1950 (sem ONI publicado) "
        "sao descartadas.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        method=args.reduction,
        pls_lag_shift=args.pls_lag_shift,
        run_dir=args.run_dir,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        lr=args.lr,
        n_jobs=args.n_jobs,
        seed=args.seed,
        force_refit=args.force_refit,
        use_oni_feature=args.use_oni_feature,
    )
