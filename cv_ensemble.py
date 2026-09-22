"""CV temporal walk-forward para selecionar configuracao, numero de epocas, ensemble de
seeds e encolhimento em direcao a climatologia do Modelo A (metodo pls_lagged).

Motivacao (ver conversa): o RMSE de uma unica janela de validacao (2019-2022) e ruidoso
(o sweep inteiro variou 0.03) e o early stopping usa a propria janela de validacao para
escolher a epoca, entao o numero reportado e otimista. Aqui:

  - 5 dobras walk-forward (mesmos cortes de enso_correction_cv.py): treino 1940-1969 /
    val 1970-79, treino ate 1979 / val 80-89, treino ate 1989 / val 90-99, treino ate 2009
    / val 2010-18, treino ate 2018 / val 2019-22. Sempre treino antes da validacao.
  - Numero de epocas FIXO (sem early stopping por dobra): cada execucao guarda a previsao
    em espaco de componentes PCA de tp apos CADA epoca; a epoca e escolhida depois, com
    a media das 5 dobras (um unico escalar - vies desprezivel).
  - Cada dobra roda num subprocesso isolado (memoria liberada entre dobras - ver
    enso_correction_cv.py) e e retomavel: pula (config, seed) ja salvos em disco.

Tudo que vem depois (media de seeds, encolhimento, empilhamento) e linear no espaco de
componentes, entao so as previsoes de componentes (N x C, minusculas) sao salvas. O RMSE na
grade e recuperado exatamente: o PCA de tp e ortonormal, entao
    MSE_grade = MSE_piso + sigma^2 * media_n(||dc_n||^2) / n_pixels
onde MSE_piso e o erro de reconstruir o proprio tp verdadeiro a partir dos componentes
(calculado uma vez por dobra) e dc = componente verdadeiro - previsto. O worker confere essa
identidade contra a reconstrucao direta da grade para cada modelo treinado.

Uso (a partir da raiz do repositorio):
    python3 cv_ensemble.py --configs base anom anom_wd --seeds 0          # roda o que falta
    python3 cv_ensemble.py --configs anom_wd --seeds 1 2 3 4               # so mais seeds
    python3 cv_ensemble.py --report                                        # tabelas + JSON
    python3 cv_ensemble.py --fold 1969 --specs base:0 anom:0               # uso interno
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.data import (
    FEATURE_VARS,
    LAGS,
    TP_VAR,
    build_examples,
    compute_normalization_stats,
    load_all_datasets,
    stack_features,
)
from src.models.pca_lstm import HindcastForecastLSTM
from src.models.pca_lstm.train import (
    BATCH_SIZE,
    PLS_LAG_SHIFT,
    _reduction_cache_config,
    fit_reduction_per_variable,
    load_or_fit_reduction,
    make_loader,
    reconstruct_tp,
)
from src.oni import oni_series_or_nan

CV_DIR = "models/_cv"
N_JOBS_REDUCTION_CV = 2  # 4 workers ja levaram OOM kill nesta maquina (15GB) - ver enso_correction_cv.py
EPOCHS = 8  # sem early stopping; a epoca boa e escolhida no --report (melhores epocas antes: 1-6)
EVAL_CHUNK = 256  # exemplos por bloco ao reconstruir a grade (78.561 pixels x float32 cada)

# mesmas dobras de enso_correction_cv.py; a ultima reaproveita o cache de reducao do run promovido
FOLDS = [
    {"nome": "1969", "train_end": "1969-12-01", "val_end": "1979-12-01"},
    {"nome": "1979", "train_end": "1979-12-01", "val_end": "1989-12-01"},
    {"nome": "1989", "train_end": "1989-12-01", "val_end": "1999-12-01"},
    {"nome": "2009", "train_end": "2009-12-01", "val_end": "2018-12-01"},
    {"nome": "2018", "train_end": "2018-12-01", "val_end": "2022-12-01"},
]
FOLD_BY_NAME = {f["nome"]: f for f in FOLDS}

# `anomaly`: o LSTM preve tp - climatologia(mes-alvo) em espaco de componentes (a sazonalidade
# deixa de ser trabalho da rede) e a climatologia e somada de volta antes de salvar.
# `weight_decay`: AdamW (desacoplado; 0.0 equivale ao Adam original).
CONFIGS = {
    "base": dict(hidden_size=128, dropout=0.1, lr=5e-4, weight_decay=0.0, anomaly=False),
    "anom": dict(hidden_size=128, dropout=0.1, lr=5e-4, weight_decay=0.0, anomaly=True),
    "anom_wd": dict(hidden_size=128, dropout=0.1, lr=5e-4, weight_decay=0.5, anomaly=True),
    "anom_wd_do3": dict(hidden_size=128, dropout=0.3, lr=5e-4, weight_decay=0.5, anomaly=True),
    "anom_h32": dict(hidden_size=32, dropout=0.3, lr=5e-4, weight_decay=0.5, anomaly=True),
    # mesmos hiperparametros do melhor LSTM da screening (anom_wd_do3) + indice ONI como input do
    # decoder (ver --use-oni-feature em src/models/pca_lstm/train.py) - reavalia sob esta CV de 5
    # dobras a conclusao anterior (piorou 1.840->1.865, mas numa unica janela de validacao 2019-2022)
    "anom_wd_do3_oni": dict(hidden_size=128, dropout=0.3, lr=5e-4, weight_decay=0.5, anomaly=True, use_oni_feature=True),
}
RIDGE_ALPHAS = [100, 1000, 10000]  # config "ridge_a<alpha>": ridge linear sobre o input do decoder
for _a in RIDGE_ALPHAS:
    CONFIGS[f"ridge_a{_a}"] = dict(ridge_alpha=_a)


def _fold_dir(cv_dir: str, nome: str) -> str:
    return f"{cv_dir}/{nome}"


def _pred_path(cv_dir: str, nome: str, config: str, seed: int) -> str:
    return f"{_fold_dir(cv_dir, nome)}/{config}_s{seed}.npz"


def _idx(time_index: pd.DatetimeIndex, date: str) -> int:
    return time_index.get_indexer([pd.Timestamp(date)])[0]


# --------------------------------------------------------------------------------------
# Worker (uma dobra, processo isolado)
# --------------------------------------------------------------------------------------


def _fold_reduction(cv_dir: str, nome: str, ds, stats: dict, train_end_idx: int):
    """Reducao PLS(defasado) ajustada so no treino da dobra, com cache em disco (o ajuste e
    a parte cara - nao depende do LSTM). A dobra 2018 usa o cache ja existente do run promovido."""
    if nome == "2018":
        return load_or_fit_reduction(
            ds, stats, train_end_idx, method="pls_lagged", pls_lag_shift=PLS_LAG_SHIFT, n_jobs=N_JOBS_REDUCTION_CV
        )
    path = f"{_fold_dir(cv_dir, nome)}/reduction.joblib"
    config = _reduction_cache_config("pls_lagged", PLS_LAG_SHIFT, train_end_idx)
    if os.path.exists(path):
        cache = joblib.load(path)
        if cache["config"] == config:
            print(f"  reducao da dobra {nome} reaproveitada de {path}")
            reduction_objects = cache["reduction_objects"]
            component_series = {
                var: reduction_objects[var]
                .transform((ds[var].values.astype("float32") - stats[var][0]) / stats[var][1])
                .astype("float32")
                for var in reduction_objects
            }
            return reduction_objects, component_series
    reduction_objects, component_series = fit_reduction_per_variable(
        ds, stats, train_end_idx, method="pls_lagged", pls_lag_shift=PLS_LAG_SHIFT, n_jobs=N_JOBS_REDUCTION_CV
    )
    os.makedirs(_fold_dir(cv_dir, nome), exist_ok=True)
    joblib.dump({"reduction_objects": reduction_objects, "config": config}, path)
    return reduction_objects, component_series


def _grid_mse(tp_pca, stats_tp, coeffs: np.ndarray, tp_raw: np.ndarray, alvo: np.ndarray) -> float:
    """MSE direto na grade (em blocos, sem materializar N x 78.561 de uma vez)."""
    sse, n_pix = 0.0, tp_raw.shape[1] * tp_raw.shape[2]
    for i in range(0, len(alvo), EVAL_CHUNK):
        rec = reconstruct_tp(tp_pca, stats_tp, coeffs[i : i + EVAL_CHUNK])
        sse += float(((tp_raw[alvo[i : i + EVAL_CHUNK]] - rec) ** 2).sum(dtype="float64"))
    return sse / (len(alvo) * n_pix)


def _predict(model, hindcast, atm, tp_froz, lag, oni=None, batch: int = 512) -> np.ndarray:
    model.eval()
    saidas = []
    with torch.no_grad():
        for i in range(0, len(lag), batch):
            saidas.append(
                model(
                    torch.from_numpy(hindcast[i : i + batch]),
                    torch.from_numpy(atm[i : i + batch]),
                    torch.from_numpy(tp_froz[i : i + batch]),
                    torch.from_numpy(lag[i : i + batch]),
                    oni=torch.from_numpy(oni[i : i + batch]) if oni is not None else None,
                ).numpy()
            )
    return np.concatenate(saidas)


def _train_lstm(cfg, seed, n_hind, n_atm, n_tp, tr, y_tr_target, va, shift_va, y_va, epochs, oni_tr=None, oni_va=None):
    """Treina `epochs` epocas fixas e devolve a previsao (em componentes, ja somada a `shift_va`)
    apos cada epoca, junto com o MSE(componentes) de treino/validacao por epoca. `oni_tr`/`oni_va`
    (opcionais): indice ONI por exemplo, ver `use_oni_feature` em CONFIGS - acrescenta 1 feature
    ao decoder (ver HindcastForecastLSTM)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    hindcast_tr, atm_tr, tp_froz_tr, lag_tr = tr
    n_oni_features = 1 if oni_tr is not None else 0
    loader = make_loader(hindcast_tr, atm_tr, tp_froz_tr, lag_tr, y_tr_target, BATCH_SIZE, shuffle=True, oni=oni_tr)
    model = HindcastForecastLSTM(
        n_features_hindcast=n_hind,
        n_features_atm=n_atm,
        n_components_tp=n_tp,
        hidden_size=cfg["hidden_size"],
        dropout=cfg["dropout"],
        n_oni_features=n_oni_features,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = torch.nn.MSELoss()

    preds, train_mse, val_mse = [], [], []
    for epoch in range(1, epochs + 1):
        model.train()
        soma, n = 0.0, 0
        for batch in loader:
            hindcast, atm, tp_froz, lag, y = batch[:5]
            oni = batch[5] if n_oni_features > 0 else None
            optimizer.zero_grad()
            loss = criterion(model(hindcast, atm, tp_froz, lag, oni=oni), y)
            loss.backward()
            optimizer.step()
            soma += loss.item() * hindcast.size(0)
            n += hindcast.size(0)
        pred = _predict(model, *va, oni=oni_va) + shift_va
        preds.append(pred)
        train_mse.append(soma / n)
        val_mse.append(float(np.mean((pred - y_va) ** 2)) if y_va is not None else float("nan"))
        print(f"    epoca {epoch:2d}/{epochs} | treino MSE {train_mse[-1]:.3f} | val MSE(comp) {val_mse[-1]:.3f}", flush=True)
    return np.stack(preds).astype("float32"), np.array(train_mse), np.array(val_mse)


def _fit_ridge(alpha, tr, y_tr_target, va, shift_va):
    def feats(hindcast, atm, tp_froz, lag):
        return np.concatenate([atm, tp_froz, lag[:, None]], axis=1)

    x_tr, x_va = feats(*tr), feats(*va)
    scaler = StandardScaler().fit(x_tr)
    modelo = Ridge(alpha=alpha).fit(scaler.transform(x_tr), y_tr_target)
    pred = modelo.predict(scaler.transform(x_va)) + shift_va
    return pred[None].astype("float32")  # (1, N, C): "1 epoca", mesmo formato do LSTM


def run_fold_worker(cv_dir: str, nome: str, specs: list[tuple[str, int]], epochs: int):
    fold = FOLD_BY_NAME[nome]
    print(f"=== Dobra {nome}: treino 1940-{fold['train_end'][:4]}, validacao {int(fold['train_end'][:4]) + 1}-{fold['val_end'][:4]} ===")
    os.makedirs(_fold_dir(cv_dir, nome), exist_ok=True)

    datasets = load_all_datasets()
    ds = stack_features(datasets)
    del datasets
    time_index = pd.DatetimeIndex(ds["time"].values)
    train_end_idx = _idx(time_index, fold["train_end"])
    val_end_idx = _idx(time_index, fold["val_end"])
    stats = compute_normalization_stats(ds, train_end=fold["train_end"])

    reduction_objects, component_series = _fold_reduction(cv_dir, nome, ds, stats, train_end_idx)
    tp_pca, stats_tp = reduction_objects[TP_VAR], stats[TP_VAR]
    n_tp = tp_pca.n_components_
    n_atm = sum(reduction_objects[v].n_components_ for v in FEATURE_VARS)
    n_hind = n_atm + n_tp

    tp_raw = ds[TP_VAR].values.astype("float32")
    for var in FEATURE_VARS:
        del ds[var]
    gc.collect()
    assert not np.isnan(tp_raw).any(), "tp com NaN: a identidade MSE_grade <-> componentes nao vale"

    tr_ex = build_examples(component_series, 0, train_end_idx, last_valid_idx=train_end_idx)
    va_ex = build_examples(component_series, train_end_idx + 1, val_end_idx, last_valid_idx=val_end_idx)
    hind_tr, atm_tr, tpf_tr, lag_tr, y_tr, orig_tr, alvo_tr = tr_ex
    hind_va, atm_va, tpf_va, lag_va, y_va, orig_va, alvo_va = va_ex
    print(f"  treino: {len(y_tr)} exemplos | validacao: {len(y_va)} exemplos | componentes tp: {n_tp}")

    meses = time_index.month.values
    tp_c = component_series[TP_VAR][: train_end_idx + 1]
    clim_comp = np.stack([tp_c[meses[: train_end_idx + 1] == m].mean(axis=0) for m in range(1, 13)]).astype("float32")
    clim_va = clim_comp[meses[alvo_va] - 1]
    clim_tr = clim_comp[meses[alvo_tr] - 1]

    # so monta o conjunto de treino com ONI (descarta origens antes de 1950, ver src.oni) se algum
    # spec pedir (CONFIGS[c]["use_oni_feature"]) - a validacao nunca cai antes de 1970 em nenhuma
    # dobra, entao o ONI dela nunca tem NaN e a lista de exemplos e identica a `va_ex` sem ONI
    tr_oni = va_oni = clim_tr_oni = y_tr_oni = None
    if any(CONFIGS[c].get("use_oni_feature") for c, _ in specs):
        oni_full = oni_series_or_nan(time_index)
        tr_ex_oni = build_examples(component_series, 0, train_end_idx, last_valid_idx=train_end_idx, oni_series=oni_full)
        va_ex_oni = build_examples(
            component_series, train_end_idx + 1, val_end_idx, last_valid_idx=val_end_idx, oni_series=oni_full
        )
        hind_tr_o, atm_tr_o, tpf_tr_o, lag_tr_o, y_tr_oni, orig_tr_o, alvo_tr_o, oni_tr = tr_ex_oni
        _, _, _, _, y_va_oni, _, alvo_va_oni, oni_va = va_ex_oni
        assert np.array_equal(alvo_va_oni, alvo_va) and np.allclose(y_va_oni, y_va), (
            "validacao com ONI deveria ser identica a validacao sem ONI (nenhuma dobra valida antes de 1970)"
        )
        tr_oni = (hind_tr_o, atm_tr_o, tpf_tr_o, lag_tr_o)
        clim_tr_oni = clim_comp[meses[alvo_tr_o] - 1]
        print(f"  treino com ONI: {len(y_tr_oni)} exemplos (vs. {len(y_tr)} sem - origens antes de 1950 descartadas)")

    meta_path = f"{_fold_dir(cv_dir, nome)}/meta.npz"
    if not os.path.exists(meta_path):
        piso = _grid_mse(tp_pca, stats_tp, y_va, tp_raw, alvo_va)  # erro de reconstruir o proprio tp verdadeiro
        np.savez(
            meta_path,
            y_va=y_va, clim_va=clim_va, clim_comp=clim_comp, alvo_va=alvo_va, origin_va=orig_va,
            lag_va=(lag_va * max(LAGS)).round().astype(int), sigma_tp=np.float64(stats_tp[1]),
            n_pixels=np.int64(tp_raw.shape[1] * tp_raw.shape[2]), floor_mse=np.float64(piso),
        )
        joblib.dump({"tp_pca": tp_pca, "stats_tp": stats_tp}, f"{_fold_dir(cv_dir, nome)}/tp_pca.joblib")
        print(f"  meta salvo (piso de reconstrucao: RMSE {np.sqrt(piso):.4f})")
    meta = np.load(meta_path)
    n_pix, sigma = int(meta["n_pixels"]), float(meta["sigma_tp"])

    tr = (hind_tr, atm_tr, tpf_tr, lag_tr)
    va = (hind_va, atm_va, tpf_va, lag_va)
    for config, seed in specs:
        out = _pred_path(cv_dir, nome, config, seed)
        if os.path.exists(out):
            continue
        cfg = CONFIGS[config]
        t0 = time.time()
        print(f"\n  -- {config} (seed {seed}) {cfg}", flush=True)
        if "ridge_alpha" in cfg:
            pred = _fit_ridge(cfg["ridge_alpha"], tr, y_tr - clim_tr, va, clim_va)
            train_mse = val_mse = np.array([np.mean((pred[0] - y_va) ** 2)])
        elif cfg.get("use_oni_feature"):
            shift_tr = clim_tr_oni if cfg["anomaly"] else np.zeros_like(clim_tr_oni)
            shift_va = clim_va if cfg["anomaly"] else np.zeros_like(clim_va)
            pred, train_mse, val_mse = _train_lstm(
                cfg, seed, n_hind, n_atm, n_tp, tr_oni, y_tr_oni - shift_tr, va, shift_va, y_va, epochs,
                oni_tr=oni_tr, oni_va=oni_va,
            )
        else:
            shift_tr = clim_tr if cfg["anomaly"] else np.zeros_like(clim_tr)
            shift_va = clim_va if cfg["anomaly"] else np.zeros_like(clim_va)
            pred, train_mse, val_mse = _train_lstm(
                cfg, seed, n_hind, n_atm, n_tp, tr, y_tr - shift_tr, va, shift_va, y_va, epochs
            )

        # conferencia da identidade: MSE na grade (direto) == piso + sigma^2 * media ||dc||^2 / n_pixels
        via_comp = float(meta["floor_mse"]) + sigma**2 * float(np.mean(np.sum((pred[-1] - y_va) ** 2, axis=1))) / n_pix
        direto = _grid_mse(tp_pca, stats_tp, pred[-1], tp_raw, alvo_va)
        print(f"  RMSE grade (ultima epoca): direto {np.sqrt(direto):.4f} | via componentes {np.sqrt(via_comp):.4f}")
        assert abs(direto - via_comp) / direto < 1e-3, "identidade grade<->componentes falhou"

        np.savez(out, pred=pred, train_mse=train_mse, val_mse=val_mse, rmse_grade_ultima=np.sqrt(direto))
        print(f"  salvo em {out} ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------------------
# Orquestrador
# --------------------------------------------------------------------------------------


def orchestrate(cv_dir: str, configs: list[str], seeds: list[int], folds: list[str], epochs: int):
    for nome in folds:
        specs = []
        for config in configs:
            for seed in [0] if "ridge_alpha" in CONFIGS[config] else seeds:
                if not os.path.exists(_pred_path(cv_dir, nome, config, seed)):
                    specs.append(f"{config}:{seed}")
        if not specs:
            print(f"dobra {nome}: nada a fazer")
            continue
        print(f"\n{'#' * 70}\n# Dobra {nome}: {len(specs)} execucoes pendentes ({' '.join(specs)})\n{'#' * 70}", flush=True)
        r = subprocess.run(
            [sys.executable, __file__, "--cv-dir", cv_dir, "--epochs", str(epochs), "--fold", nome, "--specs", *specs]
        )
        if r.returncode != 0:
            print(f"AVISO: dobra {nome} falhou (returncode={r.returncode}); rode de novo para retomar.", flush=True)


# --------------------------------------------------------------------------------------
# Relatorio (so numpy: previsoes de componentes sao pequenas)
# --------------------------------------------------------------------------------------


def _load_meta(cv_dir: str, nome: str) -> dict | None:
    path = f"{_fold_dir(cv_dir, nome)}/meta.npz"
    return dict(np.load(path)) if os.path.exists(path) else None


def _rmse_fold(meta: dict, pred: np.ndarray) -> float:
    dc2 = np.mean(np.sum((pred - meta["y_va"]) ** 2, axis=1))
    mse = float(meta["floor_mse"]) + float(meta["sigma_tp"]) ** 2 * dc2 / int(meta["n_pixels"])
    return float(np.sqrt(mse))


def _seeds_completas(cv_dir: str, config: str, folds: list[str]) -> list[int]:
    por_dobra = []
    for nome in folds:
        d = _fold_dir(cv_dir, nome)
        seeds = {int(f.split("_s")[-1][:-4]) for f in os.listdir(d) if f.startswith(f"{config}_s") and f.endswith(".npz")}
        por_dobra.append(seeds)
    return sorted(set.intersection(*por_dobra)) if por_dobra else []


def _preds(cv_dir: str, nome: str, config: str, seeds: list[int]) -> np.ndarray:
    """(E, N, C): media das seeds, por epoca."""
    return np.mean([np.load(_pred_path(cv_dir, nome, config, s))["pred"] for s in seeds], axis=0)


def _fit_alphas(deltas: list[np.ndarray], alvo: np.ndarray) -> np.ndarray:
    """Minimos quadrados de (y - clim) ~ sum_k alpha_k (p_k - clim), pooled em todos os exemplos/componentes."""
    a = np.stack([d.reshape(-1) for d in deltas], axis=1)
    return np.linalg.lstsq(a, alvo.reshape(-1), rcond=None)[0]


def report(cv_dir: str, folds: list[str], out_json: str):
    metas = {n: _load_meta(cv_dir, n) for n in folds}
    folds = [n for n in folds if metas[n] is not None]
    if len(folds) < 2:
        print("menos de 2 dobras com resultado - nada a reportar")
        return
    print(f"dobras: {' '.join(folds)}\n")

    resultado = {"folds": folds}
    clim_rmse = {n: _rmse_fold(metas[n], metas[n]["clim_va"]) for n in folds}
    piso = {n: float(np.sqrt(metas[n]["floor_mse"])) for n in folds}
    cabecalho = "  ".join(f"{n:>7}" for n in folds)
    print(f"{'RMSE na grade (mm/dia)':<34} {'media':>7}  {cabecalho}")
    print(f"{'piso de reconstrucao (PCA de tp)':<34} {np.mean(list(piso.values())):7.4f}  " + "  ".join(f"{piso[n]:7.4f}" for n in folds))
    print(f"{'climatologia (em componentes)':<34} {np.mean(list(clim_rmse.values())):7.4f}  " + "  ".join(f"{clim_rmse[n]:7.4f}" for n in folds))
    resultado["climatologia"] = clim_rmse

    melhores: dict[str, dict] = {}
    configs = [c for c in CONFIGS if all(os.path.isdir(_fold_dir(cv_dir, n)) and _seeds_completas(cv_dir, c, [n]) for n in folds)]
    for config in configs:
        seeds = _seeds_completas(cv_dir, config, folds)
        if not seeds:
            continue
        por_epoca = {n: _preds(cv_dir, n, config, seeds) for n in folds}
        n_ep = por_epoca[folds[0]].shape[0]
        media_ep = [np.mean([_rmse_fold(metas[n], por_epoca[n][e]) for n in folds]) for e in range(n_ep)]
        e_star = int(np.argmin(media_ep))
        por_dobra = {n: _rmse_fold(metas[n], por_epoca[n][e_star]) for n in folds}
        melhores[config] = {"seeds": seeds, "epoca": e_star + 1, "rmse_media": float(media_ep[e_star]), "por_dobra": por_dobra}
        nome_linha = f"{config} (E={e_star + 1}, {len(seeds)} seed{'s' if len(seeds) > 1 else ''})"
        print(f"{nome_linha:<34} {media_ep[e_star]:7.4f}  " + "  ".join(f"{por_dobra[n]:7.4f}" for n in folds))
    resultado["configs"] = melhores
    if not melhores:
        json.dump(resultado, open(out_json, "w"), indent=2)
        return

    lstm = {c: v for c, v in melhores.items() if not c.startswith("ridge")}
    print("\nRMSE medio por epoca (media das dobras e seeds):")
    for config in lstm:
        seeds = melhores[config]["seeds"]
        pe = {n: _preds(cv_dir, n, config, seeds) for n in folds}
        linha = [np.mean([_rmse_fold(metas[n], pe[n][e]) for n in folds]) for e in range(pe[folds[0]].shape[0])]
        print(f"  {config:<12} " + " ".join(f"{x:.4f}" for x in linha))

    # encolhimento + empilhamento, avaliados leave-one-fold-out (alfa ajustado nas outras dobras)
    if lstm:
        lstm_ordenado = sorted(lstm, key=lambda c: melhores[c]["rmse_media"])
        melhor = lstm_ordenado[0]
        print(f"\nMelhor LSTM: {melhor} (epoca {melhores[melhor]['epoca']}, seeds {melhores[melhor]['seeds']})")

        candidatos = {"lstm": (melhor, melhores[melhor]["epoca"] - 1, melhores[melhor]["seeds"])}
        ridges = [c for c in melhores if c.startswith("ridge")]
        if ridges:
            r = min(ridges, key=lambda c: melhores[c]["rmse_media"])
            candidatos["ridge"] = (r, 0, melhores[r]["seeds"])

        # top-3 LSTMs (configs diferentes, nao so seeds do mesmo) como candidatos individuais para
        # o empilhamento abaixo - diversidade de regularizacao entre configs pode valer mais que
        # so mais seeds do config vencedor (ver conversa)
        top3 = lstm_ordenado[: min(3, len(lstm_ordenado))]
        for c in top3:
            candidatos[f"m_{c}"] = (c, melhores[c]["epoca"] - 1, melhores[c]["seeds"])

        def pred_de(n, k):
            c, e, s = candidatos[k]
            return _preds(cv_dir, n, c, s)[e]

        combos = [["lstm"]] + ([["lstm", "ridge"]] if "ridge" in candidatos else [])
        if len(top3) > 1:
            multi = [f"m_{c}" for c in top3]
            combos.append(multi)
            if "ridge" in candidatos:
                combos.append(multi + ["ridge"])

        for nomes_modelos in combos:
            rm_sem, rm_lofo, alfas_todas = [], [], []
            deltas = {n: [pred_de(n, k) - metas[n]["clim_va"] for k in nomes_modelos] for n in folds}
            alvo = {n: metas[n]["y_va"] - metas[n]["clim_va"] for n in folds}
            for n in folds:
                outros = [m for m in folds if m != n]
                alfa = _fit_alphas(
                    [np.concatenate([deltas[m][i] for m in outros]) for i in range(len(nomes_modelos))],
                    np.concatenate([alvo[m] for m in outros]),
                )
                pred = metas[n]["clim_va"] + sum(a * d for a, d in zip(alfa, deltas[n]))
                rm_lofo.append(_rmse_fold(metas[n], pred))
                rm_sem.append(_rmse_fold(metas[n], metas[n]["clim_va"] + sum(deltas[n])))
                alfas_todas.append(alfa)
            alfa_final = _fit_alphas(
                [np.concatenate([deltas[m][i] for m in folds]) for i in range(len(nomes_modelos))],
                np.concatenate([alvo[m] for m in folds]),
            )
            rotulo = "+".join(nomes_modelos)
            print(f"\n[{rotulo}] sem encolher : media {np.mean(rm_sem):.4f} | " + "  ".join(f"{n}={x:.4f}" for n, x in zip(folds, rm_sem)))
            print(f"[{rotulo}] alfa LOFO    : media {np.mean(rm_lofo):.4f} | " + "  ".join(f"{n}={x:.4f}" for n, x in zip(folds, rm_lofo)))
            print(f"[{rotulo}] alfa (todas as dobras) = {np.round(alfa_final, 3).tolist()}")
            resultado[f"encolhimento_{rotulo}"] = {
                "rmse_sem": float(np.mean(rm_sem)), "rmse_lofo": float(np.mean(rm_lofo)),
                "alfa": alfa_final.tolist(), "por_dobra_lofo": dict(zip(folds, map(float, rm_lofo))),
            }

        # ganho do ensemble de seeds (mesma epoca, so o melhor LSTM)
        seeds = melhores[melhor]["seeds"]
        if len(seeds) > 1:
            e = melhores[melhor]["epoca"] - 1
            print(f"\nEnsemble de seeds ({melhor}, epoca {e + 1}):")
            for k in range(1, len(seeds) + 1):
                rm = np.mean([_rmse_fold(metas[n], _preds(cv_dir, n, melhor, seeds[:k])[e]) for n in folds])
                print(f"  {k} seed{'s' if k > 1 else ' '}: {rm:.4f}")

    json.dump(resultado, open(out_json, "w"), indent=2)
    print(f"\nrelatorio salvo em {out_json}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cv-dir", default=CV_DIR)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--folds", nargs="+", default=[f["nome"] for f in FOLDS], choices=list(FOLD_BY_NAME))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--report", action="store_true")
    p.add_argument("--fold", help="uso interno: worker de uma dobra")
    p.add_argument("--specs", nargs="+", help="uso interno: config:seed ...")
    a = p.parse_args()

    if a.fold:
        specs = [(s.split(":")[0], int(s.split(":")[1])) for s in a.specs]
        run_fold_worker(a.cv_dir, a.fold, specs, a.epochs)
    elif a.report:
        report(a.cv_dir, a.folds, f"{a.cv_dir}/report.json")
    else:
        orchestrate(a.cv_dir, a.configs, a.seeds, a.folds, a.epochs)


if __name__ == "__main__":
    main()
