"""Cross-validation temporal walk-forward para a correcao ENSO pos-hoc (Modelo A,
metodo pls_lagged - ver postprocess_enso.py).

Motivacao: a correcao pos-hoc original so usa residuos de 2019-2022 (4 anos, ONI
fraco/moderado: -1.1 a +0.9) para ajustar (a,b) por pixel - o teste real
(2023-2024, pico ONI +2.0) fica fora dessa faixa e por isso e clipado antes de
aplicar a correcao (ver postprocess_enso.aplicar_correcao). Agora que src/oni.py
cobre 1950-2024, este script retreina o Modelo A em varios cortes historicos
(walk-forward, nao sobreposto - nunca "buraco no meio" da linha do tempo, pra
preservar a estrutura causal hindcast+lag) para coletar residuos genuinamente
fora da amostra cobrindo decadas com El Ninos mais extremos que o de 2023-24
(1982-83 pico +2.1, 1997-98 pico +2.4, 2015-16 pico +2.6), e ajusta/valida a
correcao ENSO nesse conjunto bem maior e mais diverso.

Dobras (ver PLANO): reusa o run ja treinado (models/pls_lagged_lstm_run1, corte
2018-12) como ultima dobra, sem retreinar; mais 4 dobras novas (cortes 1969-12,
1979-12, 1989-12, 2009-12), sempre com os MESMOS hiperparametros ja promovidos
(lr=5e-4, hidden_size=128, dropout=0.3) - isso nao e um re-tuning, so uma
reamostragem temporal para a correcao ENSO.

Cada dobra roda num PROCESSO SEPARADO (subprocess, chamado com --fold <nome>),
salvando so os residuos (true_grid/pred_grid/oni) em disco antes de sair - a memoria
liberada 100% entre dobras evita o padrao que causou 3 OOM kills seguidos numa versao
anterior deste script (que mantinha `ds`, os 5 conjuntos de residuos e os workers do
joblib todos vivos ao mesmo tempo num unico processo de longa duracao). Reaproveita
tambem o mesmo padrao ja usado com sucesso em run_hparam_sweep.py. Bonus: retomavel -
se uma dobra falhar (ou o processo for morto), rodar de novo pula as que ja tem
residuos salvos em vez de refazer tudo.

Uso (a partir da raiz do repositorio):
    python3 -m scripts.enso_correction_cv                  # orquestrador: roda as dobras que faltam + agrega
    python3 -m scripts.enso_correction_cv --apply-to-test
    python3 -m scripts.enso_correction_cv --fold 1969       # uso interno (worker de 1 dobra so)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from src.data import (
    ALL_VARS,
    FEATURE_VARS,
    TP_VAR,
    build_examples,
    compute_normalization_stats,
    load_all_datasets,
    stack_features,
)
from src.evaluate import evaluate_predictions
from src.models.pca_lstm.train import (
    BATCH_SIZE,
    MAX_EPOCHS,
    PATIENCE,
    PLS_LAG_SHIFT,
    RUN_DIRS,
    SEED,
    fit_reduction_per_variable,
    make_loader,
    reconstruct_tp,
    train_model,
)
from src.oni import oni_para_datas
from scripts.postprocess_enso import _carregar_modelo, aplicar_correcao, fit_pixel_correction, prever_teste_corrigido

HIDDEN_SIZE, DROPOUT, LR = 128, 0.1, 5e-4  # hiperparametros ja promovidos (ver models/pls_lagged_lstm_run1)
N_JOBS_REDUCTION_CV = 2  # cada dobra agora roda isolada em subprocesso (ver acima), entao nao
# acumula mais memoria entre dobras - 2 workers e seguro de novo (era 1, sequencial, quando
# o processo ainda precisava sobreviver as 5 dobras inteiras)

BEST_RUN_DIR = RUN_DIRS["pls_lagged"]
FOLD_RUN_DIR = "models/_enso_cv_folds"
OUTPUT_DIR = f"{BEST_RUN_DIR}/enso_correction_walkforward"

# dobras novas (walk-forward, nao sobrepostas): cobrem os 3 El Ninos mais extremos do
# registro (1982-83, 1997-98, 2015-16) sem precisar retreinar em toda decada intermediaria
# (2000-2009 fica so como dado de treino da dobra "2009", sem ser validado)
NEW_FOLDS = [
    {"nome": "1969", "train_end": "1969-12-01", "val_end": "1979-12-01"},
    {"nome": "1979", "train_end": "1979-12-01", "val_end": "1989-12-01"},
    {"nome": "1989", "train_end": "1989-12-01", "val_end": "1999-12-01"},
    {"nome": "2009", "train_end": "2009-12-01", "val_end": "2018-12-01"},
]
REUSED_FOLD_NOME = "2018"  # models/pls_lagged_lstm_run1, ja treinado ate 2018-12, validado 2019-2022


def _idx(time_index: pd.DatetimeIndex, date: str) -> int:
    return time_index.get_indexer([pd.Timestamp(date)])[0]


def _residuals_path(nome: str) -> str:
    return f"{FOLD_RUN_DIR}/{nome}/residuals.npz"


def treinar_dobra(ds, time_index, tp_raw, train_end: str, val_end: str, nome: str):
    print(f"\n{'=' * 70}\nDobra {nome}: treino 1940-{train_end[:4]}, validacao {int(train_end[:4]) + 1}-{val_end[:4]}\n{'=' * 70}")
    train_end_idx = _idx(time_index, train_end)
    val_end_idx = _idx(time_index, val_end)

    stats = compute_normalization_stats(ds, train_end=train_end)
    reduction_objects, component_series = fit_reduction_per_variable(
        ds, stats, train_end_idx, method="pls_lagged", pls_lag_shift=PLS_LAG_SHIFT, n_jobs=N_JOBS_REDUCTION_CV,
    )

    n_components_tp = reduction_objects[TP_VAR].n_components_
    n_features_hindcast = sum(reduction_objects[v].n_components_ for v in ALL_VARS)
    n_features_atm = sum(reduction_objects[v].n_components_ for v in FEATURE_VARS)

    train_ex = build_examples(component_series, 0, train_end_idx, last_valid_idx=train_end_idx)
    val_ex = build_examples(component_series, train_end_idx + 1, val_end_idx, last_valid_idx=val_end_idx)
    hindcast_tr, atm_tr, tp_froz_tr, lag_tr, y_tr, origin_tr, alvo_tr = train_ex
    hindcast_va, atm_va, tp_froz_va, lag_va, y_va, origin_va, alvo_va = val_ex
    print(f"  treino: {len(y_tr)} exemplos | validacao: {len(y_va)} exemplos")

    train_loader = make_loader(hindcast_tr, atm_tr, tp_froz_tr, lag_tr, y_tr, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(hindcast_va, atm_va, tp_froz_va, lag_va, y_va, BATCH_SIZE, shuffle=False)

    run_dir = f"{FOLD_RUN_DIR}/{nome}"
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model, best_epoch, best_val_loss, _ = train_model(
        train_loader, val_loader, n_features_hindcast, n_features_atm, n_components_tp,
        MAX_EPOCHS, PATIENCE, run_dir, hidden_size=HIDDEN_SIZE, dropout=DROPOUT, lr=LR,
    )

    model.eval()
    with torch.no_grad():
        pred_pca_va = model(
            torch.from_numpy(hindcast_va), torch.from_numpy(atm_va),
            torch.from_numpy(tp_froz_va), torch.from_numpy(lag_va),
        ).numpy()
    pred_grid = reconstruct_tp(reduction_objects[TP_VAR], stats[TP_VAR], pred_pca_va)
    true_grid = tp_raw[alvo_va]
    metrics = evaluate_predictions(true_grid, pred_grid)
    print(f"  RMSE da dobra (validacao, sem correcao ENSO): {metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  (melhor epoca: {best_epoch})")

    oni_fold = oni_para_datas(time_index[alvo_va])

    del model, train_loader, val_loader, hindcast_tr, atm_tr, tp_froz_tr, lag_tr, y_tr
    del hindcast_va, atm_va, tp_froz_va, lag_va, y_va, pred_pca_va
    del reduction_objects, component_series
    gc.collect()

    return true_grid.astype("float32"), pred_grid.astype("float32"), oni_fold, metrics["rmse"]


def dobra_reuso_existente(ds, time_index, tp_raw):
    print(f"\n{'=' * 70}\nDobra {REUSED_FOLD_NOME} (reuso de {BEST_RUN_DIR}, sem retreinar): validacao 2019-2022\n{'=' * 70}")
    artefatos = joblib.load(f"{BEST_RUN_DIR}/reduction_and_stats.joblib")
    reduction_objects, stats = artefatos["reduction_objects"], artefatos["stats"]
    modelo = _carregar_modelo(BEST_RUN_DIR, "checkpoint_selecao_epocas.pt", reduction_objects, stats)

    component_series = {}
    media_tp, desvio_tp = stats[TP_VAR]
    component_series[TP_VAR] = reduction_objects[TP_VAR].transform(
        (ds[TP_VAR].values.astype("float32") - media_tp) / desvio_tp
    ).astype("float32")
    for var in FEATURE_VARS:
        media, desvio = stats[var]
        component_series[var] = reduction_objects[var].transform(
            (ds[var].values.astype("float32") - media) / desvio
        ).astype("float32")

    train_end_idx = _idx(time_index, "2018-12-01")
    val_end_idx = _idx(time_index, "2022-12-01")
    val_ex = build_examples(component_series, train_end_idx + 1, val_end_idx, last_valid_idx=val_end_idx)
    hindcast_va, atm_va, tp_froz_va, lag_va, y_va, origin_va, alvo_va = val_ex

    with torch.no_grad():
        pred_pca_va = modelo(
            torch.from_numpy(hindcast_va), torch.from_numpy(atm_va),
            torch.from_numpy(tp_froz_va), torch.from_numpy(lag_va),
        ).numpy()
    pred_grid = reconstruct_tp(reduction_objects[TP_VAR], stats[TP_VAR], pred_pca_va)
    true_grid = tp_raw[alvo_va]
    metrics = evaluate_predictions(true_grid, pred_grid)
    print(f"  RMSE da dobra (validacao, sem correcao ENSO): {metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}")

    oni_fold = oni_para_datas(time_index[alvo_va])

    del modelo, component_series, pred_pca_va, reduction_objects
    gc.collect()

    return true_grid.astype("float32"), pred_grid.astype("float32"), oni_fold, metrics["rmse"]


def run_fold_worker(nome: str):
    """Roda UMA dobra ate salvar os residuos em disco, e sai - chamado como subprocesso
    isolado pelo orquestrador (main()), nunca diretamente pelo usuario. Carrega os dados
    do zero (nao reusa nada de outro processo) porque e exatamente isso que garante a
    liberacao de memoria entre dobras (ver docstring do modulo)."""
    print(f"=== Carregando dados (worker da dobra {nome}) ===")
    datasets = load_all_datasets()
    ds = stack_features(datasets)
    time_index = pd.DatetimeIndex(ds["time"].values)
    tp_raw = ds[TP_VAR].values.astype("float32").copy()
    del datasets
    gc.collect()

    if nome == REUSED_FOLD_NOME:
        true_grid, pred_grid, oni_fold, rmse = dobra_reuso_existente(ds, time_index, tp_raw)
    else:
        fold = next(f for f in NEW_FOLDS if f["nome"] == nome)
        true_grid, pred_grid, oni_fold, rmse = treinar_dobra(
            ds, time_index, tp_raw, fold["train_end"], fold["val_end"], nome
        )
    del ds, tp_raw
    gc.collect()

    out_path = _residuals_path(nome)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, true_grid=true_grid, pred_grid=pred_grid, oni=oni_fold, rmse=np.float32(rmse))
    print(f"  residuos da dobra {nome} salvos em {out_path} ({len(oni_fold)} exemplos, rmse={rmse:.4f})")


def main(apply_to_test: bool):
    nomes_dobras = [f["nome"] for f in NEW_FOLDS] + [REUSED_FOLD_NOME]

    for nome in nomes_dobras:
        if os.path.exists(_residuals_path(nome)):
            print(f"  dobra {nome}: ja tem residuos salvos em {_residuals_path(nome)} - pulando "
                  f"(apague o arquivo pra refazer essa dobra)")
            continue
        print(f"\n{'#' * 70}\n# Dobra {nome}: rodando em subprocesso isolado\n{'#' * 70}")
        resultado = subprocess.run([sys.executable, "-m", "scripts.enso_correction_cv", "--fold", nome])
        if resultado.returncode != 0:
            print(f"AVISO: dobra {nome} falhou (returncode={resultado.returncode}) - "
                  f"rode 'python3 -m scripts.enso_correction_cv --fold {nome}' isolado pra depurar. "
                  f"Continuando com as demais dobras.")

    true_grids, pred_grids, onis, fold_ids, rmses_dobra = [], [], [], [], {}
    for nome in nomes_dobras:
        path = _residuals_path(nome)
        if not os.path.exists(path):
            print(f"  dobra {nome}: sem residuos (falhou) - excluida do pool")
            continue
        d = np.load(path)
        true_grids.append(d["true_grid"])
        pred_grids.append(d["pred_grid"])
        onis.append(d["oni"])
        fold_ids.append(np.full(len(d["oni"]), nome))
        rmses_dobra[nome] = float(d["rmse"])

    if len(true_grids) < 2:
        print("\nAVISO: menos de 2 dobras com resultado - nao da pra fazer leave-one-fold-out CV. Parando.")
        return None

    true_grid_pool = np.concatenate(true_grids, axis=0)
    pred_grid_pool = np.concatenate(pred_grids, axis=0)
    oni_pool = np.concatenate(onis, axis=0)
    fold_id_pool = np.concatenate(fold_ids, axis=0)
    nomes_com_resultado = list(rmses_dobra.keys())
    print(f"\n=== Pool de residuos: {len(oni_pool)} exemplos de {len(nomes_com_resultado)} "
          f"dobras ({', '.join(nomes_com_resultado)}) ===")
    print(f"  ONI no pool: min={oni_pool.min():.2f} max={oni_pool.max():.2f}")

    metrics_sem_pool = evaluate_predictions(true_grid_pool, pred_grid_pool)
    print(f"  RMSE sem correcao (pool inteiro, referencia): {metrics_sem_pool['rmse']:.4f}")

    print("\n=== Validando a correcao via leave-one-dobra-out ===")
    rmses_sem, rmses_com = [], []
    for nome_fora in nomes_com_resultado:
        mask_fora = fold_id_pool == nome_fora
        mask_dentro = ~mask_fora

        a, b = fit_pixel_correction(oni_pool[mask_dentro], true_grid_pool[mask_dentro] - pred_grid_pool[mask_dentro])
        oni_min_fold, oni_max_fold = oni_pool[mask_dentro].min(), oni_pool[mask_dentro].max()
        pred_corrigido = aplicar_correcao(pred_grid_pool[mask_fora], oni_pool[mask_fora], a, b, oni_min_fold, oni_max_fold)

        m_sem = evaluate_predictions(true_grid_pool[mask_fora], pred_grid_pool[mask_fora])
        m_com = evaluate_predictions(true_grid_pool[mask_fora], pred_corrigido)
        rmses_sem.append(m_sem["rmse"])
        rmses_com.append(m_com["rmse"])
        print(f"  dobra {nome_fora} fora (n={mask_fora.sum()}): RMSE sem={m_sem['rmse']:.4f}  com={m_com['rmse']:.4f}"
              f"  ({'melhora' if m_com['rmse'] < m_sem['rmse'] else 'piora'})")

    rmse_cv_sem = float(np.mean(rmses_sem))
    rmse_cv_com = float(np.mean(rmses_com))
    print(f"\n  RMSE medio (CV leave-one-dobra-out): sem correcao={rmse_cv_sem:.4f}  com correcao={rmse_cv_com:.4f}")
    ajuda = rmse_cv_com < rmse_cv_sem
    print(f"  {'A correcao ajuda' if ajuda else 'A correcao NAO ajuda'} em validacao cruzada honesta")

    print(f"\n=== Ajustando a correcao final com todo o pool ({len(nomes_com_resultado)} dobras) ===")
    a_final, b_final = fit_pixel_correction(oni_pool, true_grid_pool - pred_grid_pool)
    oni_min, oni_max = float(oni_pool.min()), float(oni_pool.max())
    print(f"  faixa de ONI usada no ajuste final: [{oni_min:.2f}, {oni_max:.2f}] (vs. [-1.1, 0.9] do ajuste original)")

    resultado = {
        "run_dir": BEST_RUN_DIR,
        "method": "pls_lagged",
        "dobras_planejadas": nomes_dobras,
        "dobras": nomes_com_resultado,
        "n_exemplos_por_dobra": {n: int((fold_id_pool == n).sum()) for n in nomes_com_resultado},
        "rmse_por_dobra_sem_correcao": rmses_dobra,
        "rmse_pool_sem_correcao": metrics_sem_pool["rmse"],
        "rmse_cv_leave_one_fold_out_sem_correcao": rmse_cv_sem,
        "rmse_cv_leave_one_fold_out_com_correcao": rmse_cv_com,
        "correcao_ajuda_em_cv": ajuda,
        "oni_min_ajuste": oni_min,
        "oni_max_ajuste": oni_max,
        "oni_min_ajuste_original_2019_2022": -1.1,
        "oni_max_ajuste_original_2019_2022": 0.9,
        "fonte_oni": "NOAA CPC, ONI v6, 1950-2024 (ver src/oni.py)",
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/report.json", "w") as f:
        json.dump(resultado, f, indent=2)
    joblib.dump({"a": a_final, "b": b_final, "oni_min": oni_min, "oni_max": oni_max}, f"{OUTPUT_DIR}/coeficientes.joblib")
    print(f"\n  relatorio salvo em {OUTPUT_DIR}/report.json")
    print(f"  coeficientes salvos em {OUTPUT_DIR}/coeficientes.joblib")

    if not ajuda:
        print("\nAVISO: a correcao NAO melhorou o RMSE em validacao cruzada honesta - "
              "nao aplicando no teste real mesmo que --apply-to-test tenha sido passado.")
        return resultado

    if apply_to_test:
        print("\n=== Aplicando a correcao (walk-forward) na previsao real do teste (2023-2024) ===")
        artefatos = joblib.load(f"{BEST_RUN_DIR}/reduction_and_stats.joblib")
        reduction_objects, stats = artefatos["reduction_objects"], artefatos["stats"]

        # recarrega os dados brutos (o orquestrador nunca os manteve em memoria - cada
        # dobra roda em subprocesso isolado, ver docstring do modulo) so para montar
        # component_series de novo, na mesma reducao do run vencedor (2018-12)
        component_series = {}
        datasets = load_all_datasets()
        ds_full = stack_features(datasets)
        time_index_full = pd.DatetimeIndex(ds_full["time"].values)
        media_tp, desvio_tp = stats[TP_VAR]
        component_series[TP_VAR] = reduction_objects[TP_VAR].transform(
            (ds_full[TP_VAR].values.astype("float32") - media_tp) / desvio_tp
        ).astype("float32")
        for var in FEATURE_VARS:
            media, desvio = stats[var]
            component_series[var] = reduction_objects[var].transform(
                (ds_full[var].values.astype("float32") - media) / desvio
            ).astype("float32")
        last_valid_idx = len(time_index_full) - 1
        del datasets, ds_full
        gc.collect()

        prever_teste_corrigido(
            BEST_RUN_DIR, reduction_objects, stats, component_series, last_valid_idx,
            a_final, b_final, oni_min, oni_max,
            output_path="submissions/submission_pls_lagged_lstm_enso_corrected_walkforward.csv",
        )

    return resultado


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply-to-test", action="store_true",
                         help="Alem de avaliar a correcao, gera uma submissao corrigida para 2023-2024 "
                         "(so se a correcao ajudar em validacao cruzada) - "
                         "submissions/submission_pls_lagged_lstm_enso_corrected_walkforward.csv")
    parser.add_argument("--fold", default=None, metavar="NOME",
                         help="Uso interno: roda so essa dobra (worker isolado em subprocesso, chamado "
                         "pelo proprio script - nao chame diretamente a menos que esteja depurando uma "
                         "dobra que falhou, ver mensagem de erro do orquestrador).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.fold:
        run_fold_worker(args.fold)
    else:
        main(apply_to_test=args.apply_to_test)
