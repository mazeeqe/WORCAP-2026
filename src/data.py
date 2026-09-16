"""Pipeline de dados compartilhado (dono: P1).

Contrato combinado na Fase 0 do PLANO_TRABALHO.md: para uma origem `o` e um lag `L`
(1 a 24), o exemplo de treino e
    X = [variaveis atmosfericas do mes o+L-1, tp do mes o (congelado), L]
    y = tp do mes o+L (equivalente a tp_alvo do mes o+L-1; ver nota abaixo)
simulando exatamente o cenario do teste real (tp_ultima_obs congelado em dez/2022).

O deslocamento `o+L-1` e essencial: em `teste_features.nc`, cada posicao do eixo
do mes-alvo guarda o estado atmosferico do mes anterior em `time_origem`.

Nota sobre o alvo: `treino_tp_alvo.nc` e so `tp` deslocado um mes para frente
(`tp_alvo[t] == tp[t+1]`), confirmado inspecionando os dados brutos. Por isso
`build_examples` usa `tp` diretamente como fonte do alvo em vez de `tp_alvo` -
evita ambiguidade de indice e permite gerar qualquer lag a partir de uma unica serie.

A grade e grande (301 lat x 261 lon = 78.561 pontos), entao este modulo nao acha
a grade em (X, y) por ponto - ele so faz o alinhamento/normalizacao no espaco
original. A reducao dimensional (PCA/EOF) e feita em `src/models/pca_lstm.py`, e
`build_examples` (abaixo) opera sobre as series de coeficientes PCA ja reduzidas,
nao sobre a grade completa.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from download_data import download_competition_data, load_dataset_files

FEATURE_VARS = [
    "t2",
    "cloud_cover",
    "shum_850",
    "surface_pressure",
    "u_850",
    "v_850",
    "temperature_850",
    "rel_hum_850",
    "geopotential_850",
]
TP_VAR = "tp"
ALL_VARS = FEATURE_VARS + [TP_VAR]

TRAIN_FILES = {
    "t2": "treino_t2",
    "cloud_cover": "treino_cloud_cover",
    "shum_850": "treino_shum_850",
    "surface_pressure": "treino_surface_pressure",
    "u_850": "treino_u_850",
    "v_850": "treino_v_850",
    "temperature_850": "treino_temperature_850",
    "rel_hum_850": "treino_rel_hum_850",
    "geopotential_850": "treino_geopotential_850",
    "tp": "treino_tp",
}

TRAIN_END = "2018-12-01"  # inclusive - origens de treino do modelo (interno)
VAL_END = "2022-12-01"  # inclusive - origens de validacao do modelo (interno)
HINDCAST_LEN = 12  # meses de historico usados pelo encoder
LAGS = range(1, 25)  # meses a frente, igual ao teste real


def load_all_datasets() -> dict[str, xr.Dataset]:
    """Baixa (se preciso) e carrega todos os .nc/.csv da competicao."""
    path = download_competition_data()
    _dataframes, datasets = load_dataset_files(path)
    return datasets


def stack_features(datasets: dict[str, xr.Dataset]) -> xr.Dataset:
    """Junta as variaveis de treino (um arquivo .nc por variavel) num Dataset unico
    com dims (time, lat, lon) e uma data_var por variavel em ALL_VARS."""
    data_vars = {var: datasets[key][var] for var, key in TRAIN_FILES.items()}
    return xr.Dataset(data_vars)


def compute_normalization_stats(ds: xr.Dataset, train_end: str = TRAIN_END) -> dict[str, tuple[float, float]]:
    """Media/desvio por variavel, calculados so no periodo de treino.

    Usa dtype=float64 na reducao (ver nota em eda.py sobre o bug de precisao do
    bottleneck em arrays float32 grandes).
    """
    ds_train = ds.sel(time=slice(None, train_end))
    stats = {}
    for var in ds.data_vars:
        media = float(ds_train[var].mean(skipna=True, dtype="float64"))
        desvio = float(ds_train[var].std(skipna=True, dtype="float64"))
        stats[var] = (media, desvio)
    return stats


def normalize(ds: xr.Dataset, stats: dict[str, tuple[float, float]]) -> xr.Dataset:
    """Aplica z-score usando as estatisticas do treino."""
    ds_norm = ds.copy()
    for var, (media, desvio) in stats.items():
        ds_norm[var] = (ds[var] - media) / desvio
    return ds_norm


def temporal_split(ds: xr.Dataset, train_end: str = TRAIN_END, val_end: str = VAL_END):
    """Corta o Dataset em treino / validacao, respeitando a ordem temporal (nunca split aleatorio)."""
    ds_train = ds.sel(time=slice(None, train_end))
    ds_val = ds.sel(time=slice(train_end, val_end))
    return ds_train, ds_val


def build_examples(
    component_series: dict[str, np.ndarray],
    origin_start_idx: int,
    origin_end_idx: int,
    last_valid_idx: int,
    hindcast_len: int = HINDCAST_LEN,
    lags: range = LAGS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gera exemplos (hindcast, alvo_atm, tp_congelado, lag) -> y a partir das series
    de coeficientes PCA de cada variavel (ja alinhadas no tempo, indice 0 = primeiro
    mes do dataset completo).

    component_series: {variavel: array (n_meses_total, n_componentes)}, uma entrada
    por variavel em ALL_VARS (mesmas n_componentes por variavel, mas podem diferir).
    origin_start_idx/origin_end_idx: intervalo (inclusive) de indices de origem `o`
    a considerar (ex.: origens de treino ou de validacao).
    last_valid_idx: ultimo indice de tempo com dado real disponivel (para nao gerar
    exemplo cujo alvo o+L caia fora dos dados).

    Retorna:
        hindcast: (N, hindcast_len, F_total) - todas as variaveis, H meses ate a origem
        alvo_atm: (N, F_atm) - variaveis atmosfericas (sem tp) no mes o+L-1
        tp_congelado: (N, C_tp) - tp no mes da origem o (simula tp_ultima_obs)
        lag: (N,) - L normalizado em [1/24, 1]
        y: (N, C_tp) - tp no mes o+L (o que o modelo deve prever)
        origin_idx: (N,) - indice absoluto de tempo da origem `o` (para baselines/plots)
        alvo_idx: (N,) - indice absoluto de tempo do alvo `o+L` (para baselines/plots)
    """
    atm_vars = [v for v in ALL_VARS if v != TP_VAR]
    hindcast_list, alvo_atm_list, tp_congelado_list, lag_list, y_list = [], [], [], [], []
    origin_idx_list, alvo_idx_list = [], []

    for o in range(max(origin_start_idx, hindcast_len - 1), origin_end_idx + 1):
        hindcast_o = np.concatenate(
            [component_series[v][o - hindcast_len + 1 : o + 1] for v in ALL_VARS],
            axis=1,
        )  # (hindcast_len, F_total)
        tp_congelado_o = component_series[TP_VAR][o]

        for lag in lags:
            alvo_idx = o + lag
            if alvo_idx > last_valid_idx:
                break  # lags maiores tambem estourariam - origens perto do fim tem menos lags

            feature_idx = alvo_idx - 1
            alvo_atm = np.concatenate([component_series[v][feature_idx] for v in atm_vars])
            y = component_series[TP_VAR][alvo_idx]

            hindcast_list.append(hindcast_o)
            alvo_atm_list.append(alvo_atm)
            tp_congelado_list.append(tp_congelado_o)
            lag_list.append(lag / max(lags))
            y_list.append(y)
            origin_idx_list.append(o)
            alvo_idx_list.append(alvo_idx)

    return (
        np.asarray(hindcast_list, dtype=np.float32),
        np.asarray(alvo_atm_list, dtype=np.float32),
        np.asarray(tp_congelado_list, dtype=np.float32),
        np.asarray(lag_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(origin_idx_list, dtype=np.int64),
        np.asarray(alvo_idx_list, dtype=np.int64),
    )
