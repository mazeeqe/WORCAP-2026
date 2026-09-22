"""Pos-processamento: correcao de vies condicionada ao ENSO (indice ONI), aplicada em
cima de um Modelo A ja treinado (ver src/models/pca_lstm/train.py).

Motivacao (ver conversa): o teste real (2023-2024) cai quase inteiro dentro de um dos
El Ninos mais fortes ja registrados (pico ONI +2.0 em NDJ/2023), e o modelo nao recebe
nenhum indice de ENSO/SST como entrada - so variaveis atmosfericas locais. A analise de
tendencia de longo prazo (1940-2022) nao achou tendencia significativa em `tp` (p=0.62),
entao o ajuste aqui e por ENSO, nao por uma tendencia de aquecimento global na propria
precipitacao (essa nao apareceu nos dados).

Metodo: para cada ponto de grade, ajusta uma regressao linear simples
    residuo(lat, lon) = a(lat, lon) + b(lat, lon) * ONI(mes-alvo)
sobre os residuos (verdadeiro - previsto) do periodo de validacao interna (2019-2022,
genuinamente fora da amostra - usa checkpoint_selecao_epocas.pt, treinado so ate 2018-12).
Fechado por OLS (vetorizado por pixel, nao iterativo). Validado via leave-one-year-out
dentro da propria validacao (4 anos) antes de decidir se vale aplicar no teste real.

O ONI de cada mes-alvo, ao aplicar em 2023-2024, e limitado (clipado) a faixa observada
no ajuste (2019-2022) - o pico de 2023-2024 (~+2.0) e bem mais alto que qualquer coisa
vista na validacao (~+1.0 max), entao extrapolar sem limite seria arriscado.

Tabela do ONI em src/oni.py (NOAA CPC, ONI v6, 1950-01 a 2024-12 - compartilhada com a
feature de entrada opcional --use-oni-feature em src/models/pca_lstm/train.py).

Uso:
    python3 postprocess_enso.py                              # usa RUN_DIRS["pls_lagged"] (melhor RMSE)
    python3 postprocess_enso.py --run-dir models/pca_lstm_run1
    python3 postprocess_enso.py --apply-to-test               # tambem gera submissao corrigida
"""

from __future__ import annotations

import argparse
import gc
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.baseline import climatology_baseline, persistence_baseline
from src.data import (
    ALL_VARS,
    FEATURE_VARS,
    HINDCAST_LEN,
    LAGS,
    TP_VAR,
    TRAIN_FILES,
    TRAIN_END,
    build_examples,
    load_all_datasets,
)
from src.evaluate import evaluate_predictions
from src.models.pca_lstm import HindcastForecastLSTM
from src.models.pca_lstm.train import RUN_DIRS, reconstruct_tp
from src.oni import oni_para_datas
from src.submit import build_submission
from download_data import download_competition_data

DEVICE = torch.device("cpu")


def fit_pixel_correction(oni: np.ndarray, residuo: np.ndarray, granularity: str = "pixel") -> tuple[np.ndarray, np.ndarray]:
    """OLS fechado: residuo[n, lat, lon] ~ a + b*oni[n].

    `granularity="pixel"`: ajusta (a, b) por pixel (lat, lon) - flexivel, mas com 852
    exemplos cobrindo so 4 anos (~4 estados de ENSO independentes), e um risco real de
    overfitting por pixel (78561 pixels x 2 parametros).
    `granularity="global"`: um unico (a, b) pra grade inteira (media espacial do residuo
    por exemplo) - so 2 parametros no total, bem mais robusto ao pouco dado, mas nao
    captura o dipolo espacial do ENSO (sul mais chuvoso / norte mais seco).

    `residuo` com shape (N, lat, lon). Retorna (a, b): (lat, lon) se pixel, escalares
    (broadcastable) se global."""
    oni_c = oni - oni.mean()
    var_oni = float(np.mean(oni_c**2))

    if granularity == "global":
        residuo_medio = residuo.mean(axis=(1, 2))  # (N,)
        b = float(np.mean(oni_c * (residuo_medio - residuo_medio.mean())) / var_oni)
        a = float(residuo_medio.mean() - b * oni.mean())
        return np.float32(a), np.float32(b)

    residuo_c = residuo - residuo.mean(axis=0, keepdims=True)
    b = np.mean(oni_c[:, None, None] * residuo_c, axis=0) / var_oni
    a = residuo.mean(axis=0) - b * oni.mean()
    return a.astype("float32"), b.astype("float32")


def aplicar_correcao(pred_grid: np.ndarray, oni: np.ndarray, a, b, oni_min: float, oni_max: float) -> np.ndarray:
    """`a`/`b` escalares (correcao global) ou (lat, lon) (correcao por pixel) - o broadcast
    do numpy cobre os dois casos sem checagem explicita de shape."""
    oni_clip = np.clip(oni, oni_min, oni_max)
    correcao = a + b * oni_clip[:, None, None]
    return np.clip(pred_grid + correcao, 0, None)


def _carregar_modelo(run_dir: str, checkpoint_nome: str, reduction_objects, stats) -> HindcastForecastLSTM:
    n_components_tp = reduction_objects[TP_VAR].n_components_
    n_features_hindcast = sum(reduction_objects[v].n_components_ for v in ALL_VARS)
    n_features_atm = sum(reduction_objects[v].n_components_ for v in FEATURE_VARS)
    modelo = HindcastForecastLSTM(
        n_features_hindcast=n_features_hindcast, n_features_atm=n_features_atm,
        n_components_tp=n_components_tp, hidden_size=128, dropout=0.1,
    ).to(DEVICE)
    ckpt = torch.load(f"{run_dir}/{checkpoint_nome}", map_location=DEVICE)
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()
    return modelo


def prever_teste_corrigido(
    run_dir: str,
    reduction_objects: dict,
    stats: dict,
    component_series: dict,
    last_valid_idx: int,
    a,
    b,
    oni_min: float,
    oni_max: float,
    output_path: str | None = None,
):
    """Gera a previsao corrigida do teste real (2023-2024) a partir de um Modelo A ja
    retreinado com todo o historico (checkpoint_retrain_final.pt) e uma correcao ENSO
    ja ajustada (a, b, oni_min/oni_max - ver fit_pixel_correction). Reusada tanto pelo
    fluxo padrao (main(), ajuste so em 2019-2022) quanto pela CV walk-forward
    (enso_correction_cv.py, ajuste pooled em varias decadas de residuos)."""
    modelo_final = _carregar_modelo(run_dir, "checkpoint_retrain_final.pt", reduction_objects, stats)

    datasets = load_all_datasets()
    teste_ds = datasets["teste_features"]

    origem_idx = last_valid_idx
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
        atm_teste_list.append(reduction_objects[var].transform(normalizado))
    atm_teste = np.concatenate(atm_teste_list, axis=1).astype("float32")

    hindcast_batch = np.repeat(hindcast_teste[None, :, :], n_meses_teste, axis=0)
    tp_congelado_batch = np.repeat(tp_congelado_teste[None, :], n_meses_teste, axis=0)
    lag_batch = (teste_ds["lag_meses"].values / max(LAGS)).astype("float32")

    with torch.no_grad():
        pred_pca_teste = modelo_final(
            torch.from_numpy(hindcast_batch), torch.from_numpy(atm_teste),
            torch.from_numpy(tp_congelado_batch), torch.from_numpy(lag_batch),
        ).numpy()
    pred_grid_teste_bruto = reconstruct_tp(reduction_objects[TP_VAR], stats[TP_VAR], pred_pca_teste)
    pred_grid_teste_bruto = np.clip(pred_grid_teste_bruto, 0, None)

    datas_teste = pd.DatetimeIndex(teste_ds["time"].values)
    oni_teste = oni_para_datas(datas_teste)
    print(f"  ONI no teste: min={oni_teste.min():.2f} max={oni_teste.max():.2f} "
          f"(clipado para [{oni_min:.2f}, {oni_max:.2f}] antes de aplicar)")

    pred_grid_teste_corrigido = aplicar_correcao(pred_grid_teste_bruto, oni_teste, a, b, oni_min, oni_max)

    predictions_da = xr.DataArray(
        pred_grid_teste_corrigido, dims=("time", "lat", "lon"),
        coords={"time": teste_ds["time"].values, "lat": teste_ds["lat"].values, "lon": teste_ds["lon"].values},
    )
    os.makedirs("submissions", exist_ok=True)
    competition_path = download_competition_data()
    sample_path = os.path.join(competition_path, "sample_submission.csv")
    if output_path is None:
        run_tag = os.path.basename(os.path.normpath(run_dir)).replace("_run1", "")
        output_path = f"submissions/submission_{run_tag}_enso_corrected.csv"
    submission_df = build_submission(predictions_da, sample_path, output_path)
    print(f"  salvo em {output_path} ({len(submission_df)} linhas)")
    return submission_df


def main(run_dir: str, apply_to_test: bool, granularity: str = "pixel"):
    print(f"=== Carregando artefatos de {run_dir} ===")
    artefatos = joblib.load(f"{run_dir}/reduction_and_stats.joblib")
    reduction_objects, stats, method = artefatos["reduction_objects"], artefatos["stats"], artefatos["method"]

    modelo_val = _carregar_modelo(run_dir, "checkpoint_selecao_epocas.pt", reduction_objects, stats)
    print("  checkpoint de selecao de epocas carregado (treinado so em 0..2018-12)")

    print("\n=== Carregando dados (um var por vez, liberando memoria a cada passo) ===")
    datasets = load_all_datasets()
    datasets.pop("teste_features", None)
    datasets.pop("sample_submission", None)
    gc.collect()

    tp_key = TRAIN_FILES[TP_VAR]
    time_index = pd.DatetimeIndex(datasets[tp_key]["time"].values)
    train_end_idx = time_index.get_indexer([pd.Timestamp(TRAIN_END)])[0]
    last_valid_idx = len(time_index) - 1

    media_tp, desvio_tp = stats[TP_VAR]
    tp_raw = datasets[tp_key][TP_VAR].values.astype("float32").copy()
    tp_normalizado = (tp_raw - media_tp) / desvio_tp
    component_series = {TP_VAR: reduction_objects[TP_VAR].transform(tp_normalizado).astype("float32")}
    del tp_normalizado, datasets[tp_key]
    gc.collect()

    for var in FEATURE_VARS:
        key = TRAIN_FILES[var]
        media, desvio = stats[var]
        bruto = datasets[key][var].values.astype("float32")
        normalizado = (bruto - media) / desvio
        component_series[var] = reduction_objects[var].transform(normalizado).astype("float32")
        del bruto, normalizado, datasets[key]
        gc.collect()
    del datasets
    gc.collect()
    print("  dados brutos liberados da memoria")

    print("\n=== Gerando exemplos de validacao interna (2019-2022, genuinamente fora da amostra) ===")
    val_ex = build_examples(component_series, train_end_idx + 1, last_valid_idx, last_valid_idx=last_valid_idx)
    hindcast, atm, tp_froz, lag, y, origin_idx, alvo_idx = val_ex
    datas_alvo = time_index[alvo_idx]
    oni_val = oni_para_datas(datas_alvo)
    print(f"  {len(y)} exemplos | ONI no periodo: min={oni_val.min():.2f} max={oni_val.max():.2f}")

    with torch.no_grad():
        pred_pca_val = modelo_val(
            torch.from_numpy(hindcast), torch.from_numpy(atm), torch.from_numpy(tp_froz), torch.from_numpy(lag)
        ).numpy()
    pred_grid_val = reconstruct_tp(reduction_objects[TP_VAR], stats[TP_VAR], pred_pca_val)
    true_grid_val = tp_raw[alvo_idx]

    metrics_sem = evaluate_predictions(true_grid_val, pred_grid_val)
    print(f"\n  RMSE sem correcao (validacao inteira, referencia): {metrics_sem['rmse']:.4f}")

    print("\n=== Validando a correcao via leave-one-year-out (dentro de 2019-2022) ===")
    anos_val = datas_alvo.year.values
    rmses_sem, rmses_com = [], []
    for ano_fora in sorted(set(anos_val)):
        treino_mask = anos_val != ano_fora
        teste_mask = anos_val == ano_fora
        if teste_mask.sum() == 0 or treino_mask.sum() < 10:
            continue
        a, b = fit_pixel_correction(oni_val[treino_mask], true_grid_val[treino_mask] - pred_grid_val[treino_mask], granularity)
        oni_min_fold, oni_max_fold = oni_val[treino_mask].min(), oni_val[treino_mask].max()
        pred_corrigido = aplicar_correcao(pred_grid_val[teste_mask], oni_val[teste_mask], a, b, oni_min_fold, oni_max_fold)

        m_sem = evaluate_predictions(true_grid_val[teste_mask], pred_grid_val[teste_mask])
        m_com = evaluate_predictions(true_grid_val[teste_mask], pred_corrigido)
        rmses_sem.append(m_sem["rmse"])
        rmses_com.append(m_com["rmse"])
        print(f"  ano {ano_fora} fora (n={teste_mask.sum()}): RMSE sem={m_sem['rmse']:.4f}  com={m_com['rmse']:.4f}"
              f"  ({'melhora' if m_com['rmse'] < m_sem['rmse'] else 'piora'})")

    rmse_cv_sem = float(np.mean(rmses_sem))
    rmse_cv_com = float(np.mean(rmses_com))
    print(f"\n  RMSE medio (CV leave-one-year-out): sem correcao={rmse_cv_sem:.4f}  com correcao={rmse_cv_com:.4f}")
    ajuda = rmse_cv_com < rmse_cv_sem
    print(f"  {'A correcao ajuda' if ajuda else 'A correcao NAO ajuda'} em validacao cruzada honesta")

    print("\n=== Ajustando a correcao final com toda a validacao (2019-2022) ===")
    a_final, b_final = fit_pixel_correction(oni_val, true_grid_val - pred_grid_val, granularity)
    oni_min, oni_max = float(oni_val.min()), float(oni_val.max())
    print(f"  faixa de ONI usada no ajuste (fora disso, o ONI de teste sera clipado): [{oni_min:.2f}, {oni_max:.2f}]")

    resultado = {
        "run_dir": run_dir,
        "method": method,
        "rmse_val_sem_correcao": metrics_sem["rmse"],
        "mae_val_sem_correcao": metrics_sem["mae"],
        "rmse_cv_leave_one_year_out_sem_correcao": rmse_cv_sem,
        "rmse_cv_leave_one_year_out_com_correcao": rmse_cv_com,
        "correcao_ajuda_em_cv": ajuda,
        "oni_min_ajuste": oni_min,
        "oni_max_ajuste": oni_max,
        "fonte_oni": "NOAA CPC, ONI v6, consultado 2026-09-16",
    }

    os.makedirs(f"{run_dir}/enso_correction", exist_ok=True)
    with open(f"{run_dir}/enso_correction/report.json", "w") as f:
        json.dump(resultado, f, indent=2)
    joblib.dump({"a": a_final, "b": b_final, "oni_min": oni_min, "oni_max": oni_max},
                f"{run_dir}/enso_correction/coeficientes.joblib")
    print(f"\n  relatorio salvo em {run_dir}/enso_correction/report.json")
    print(f"  coeficientes salvos em {run_dir}/enso_correction/coeficientes.joblib")

    if not ajuda:
        print("\nAVISO: a correcao NAO melhorou o RMSE em validacao cruzada honesta - "
              "nao aplicando no teste real mesmo que --apply-to-test tenha sido passado.")
        return resultado

    if apply_to_test:
        print("\n=== Aplicando a correcao na previsao real do teste (2023-2024) ===")
        prever_teste_corrigido(
            run_dir, reduction_objects, stats, component_series, last_valid_idx,
            a_final, b_final, oni_min, oni_max,
        )

    return resultado


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=RUN_DIRS["pls_lagged"],
                         help="Pasta do run do Modelo A ja treinado (padrao: %(default)s, o de melhor RMSE).")
    parser.add_argument("--apply-to-test", action="store_true",
                         help="Alem de avaliar a correcao, gera uma submissao corrigida para 2023-2024 "
                         "(so se a correcao ajudar em validacao cruzada).")
    parser.add_argument("--granularity", choices=["pixel", "global"], default="pixel",
                         help="'pixel': (a,b) por ponto de grade (flexivel, mas 78561x2 parametros "
                         "pra so 852 exemplos/4 anos). 'global': um unico (a,b) pra grade inteira "
                         "(so 2 parametros - mais robusto ao pouco dado, mas nao captura o dipolo "
                         "espacial do ENSO). Padrao: %(default)s.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(run_dir=args.run_dir, apply_to_test=args.apply_to_test, granularity=args.granularity)
