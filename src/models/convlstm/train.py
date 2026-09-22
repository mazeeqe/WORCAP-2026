"""Treina o Modelo B (ConvLSTM) e gera a previsao/submissao.

Diferenca central para o Modelo A (src/models/pca_lstm/train.py): aqui a grade
espacial NAO e reduzida por PCA/PLS -- o ConvLSTM processa (uma versao
reamostrada de) a grade 301x261 diretamente via convolucoes, usando o mesmo
contrato origem+lag do resto do time (Fase 0 do PLANO_TRABALHO.md), mas com
uma montagem de exemplos "preguicosa" (ConvLSTMDataset abaixo) em vez de
`src.data.build_examples`: essa funcao concatena tudo em arrays densos, o que
e barato para poucas dezenas de componentes PCA mas custaria ~800GB de RAM na
grade cheia (301*261 pontos * 10 variaveis * 12 meses de hindcast).

Uso (a partir da raiz do repositorio):
    python3 -m src.models.convlstm.train
    python3 -m src.models.convlstm.train --spatial-downsample 4   # grade mais fina (mais lento)
    python3 -m src.models.convlstm.train --validate-only          # pula retreino+submissao (~mais rapido)
    python3 -m src.models.convlstm.train --smoke-test             # 1 epoca, poucos exemplos -- so pra checar que roda

Passos:
    1. Carrega os dados e normaliza (mesmas stats/convencao do Modelo A).
    2. Reamostra cada grade espacialmente (`--spatial-downsample`, media por
       blocos via adaptive_avg_pool2d) -- sem isso o treino em CPU nao e
       viavel no tempo do hackathon; ver PLANO_TRABALHO.md/configs/convlstm.yaml.
    3. Treina o ConvLSTM na grade reduzida, com split temporal interno
       (treino/validacao) para early stopping, prevendo tp diretamente em
       mm/dia (nao em z-score, que pode ser negativo mesmo com tp>=0 -- por
       isso a Softplus da saida do modelo so faz sentido em unidade fisica).
    4. Reconstroi a previsao de volta pra grade cheia (301x261, interpolacao
       bilinear) e reporta RMSE/MAE contra os baselines nessa mesma
       resolucao -- garante que o numero e comparavel ao dos outros modelos
       (que sempre avaliam na grade cheia), mesmo o ConvLSTM tendo treinado
       numa grade menor.
    5. (a menos que --validate-only) Retreina com todo o historico rotulado e
       gera a previsao/submissao do teste real (2023-2024).
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader, Dataset

from download_data import download_competition_data
from src.baseline import climatology_baseline, persistence_baseline
from src.data import (
    ALL_VARS,
    FEATURE_VARS,
    HINDCAST_LEN,
    LAGS,
    TP_VAR,
    TRAIN_END,
    compute_normalization_stats,
    load_all_datasets,
    stack_features,
)
from src.evaluate import evaluate_predictions
from src.models.convlstm import ConvLSTMForecaster
from src.submit import build_submission

RUN_DIR = "models/convlstm_run1"

SPATIAL_DOWNSAMPLE = 6  # 301x261 -> ~50x44; CPU-only (ver train.py), reduzir custo ~36x
HIDDEN_CHANNELS = 32
NUM_LAYERS = 2
KERNEL_SIZE = 3
LR = 1e-3
BATCH_SIZE = 8
MAX_EPOCHS = 50
PATIENCE = 5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)


def downsample_grids(grid: np.ndarray, factor: int) -> np.ndarray:
    """`grid`: (n, lat, lon) -> (n, ceil(lat/factor), ceil(lon/factor)), media por
    blocos (adaptive_avg_pool2d lida com 301/261 nao sendo multiplos exatos de
    `factor`). Funciona tanto para grades normalizadas (treino) quanto fisicas
    (mm/dia, ver upsample_to_full_res para o caminho inverso)."""
    if factor <= 1:
        return grid
    lat, lon = grid.shape[-2:]
    tamanho_alvo = (-(-lat // factor), -(-lon // factor))  # ceil division
    tensor = torch.from_numpy(grid).unsqueeze(1)  # (n, 1, lat, lon)
    reduzido = F.adaptive_avg_pool2d(tensor, tamanho_alvo)
    return reduzido.squeeze(1).numpy()


def upsample_to_full_res(grid_reduzido: np.ndarray, lat: int, lon: int) -> np.ndarray:
    """(n, h, w) reduzido -> (n, lat, lon) via interpolacao bilinear -- usado so na
    hora de reportar metricas/submissao, pra ficar comparavel com os outros modelos
    (que sempre avaliam na grade cheia)."""
    tensor = torch.from_numpy(grid_reduzido).unsqueeze(1)  # (n, 1, h, w)
    cheio = F.interpolate(tensor, size=(lat, lon), mode="bilinear", align_corners=False)
    return cheio.squeeze(1).numpy()


class ConvLSTMDataset(Dataset):
    """Gera (hindcast, atm_alvo, lag, y) a partir de grades ja normalizadas/reamostradas,
    de forma preguicosa (so faz o np.stack de cada exemplo quando pedido) -- ver nota de
    modulo sobre por que nao dá pra usar src.data.build_examples aqui (memoria)."""

    def __init__(
        self,
        grids_norm: dict[str, np.ndarray],
        tp_raw_ds: np.ndarray,
        origin_start_idx: int,
        origin_end_idx: int,
        last_valid_idx: int,
        hindcast_len: int = HINDCAST_LEN,
        lags: range = LAGS,
    ):
        self.grids = grids_norm
        self.tp_raw_ds = tp_raw_ds
        self.hindcast_len = hindcast_len
        self.pairs: list[tuple[int, int, int]] = []  # (origin, alvo, lag)
        for o in range(max(origin_start_idx, hindcast_len - 1), origin_end_idx + 1):
            for lag in lags:
                alvo_idx = o + lag
                if alvo_idx > last_valid_idx:
                    break
                self.pairs.append((o, alvo_idx, lag))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        o, alvo_idx, lag = self.pairs[i]
        feature_idx = alvo_idx - 1  # mesma convencao de src.data.build_examples (sem vazamento)
        hindcast = np.stack(
            [self.grids[v][o - self.hindcast_len + 1: o + 1] for v in ALL_VARS], axis=1
        )  # (H, C, h, w)
        atm = np.stack([self.grids[v][feature_idx] for v in FEATURE_VARS], axis=0)  # (C_atm, h, w)
        y = self.tp_raw_ds[alvo_idx]  # (h, w) mm/dia
        lag_norm = np.float32(lag / max(LAGS))
        return (
            torch.from_numpy(hindcast),
            torch.from_numpy(atm),
            torch.tensor(lag_norm),
            torch.from_numpy(y),
            o,
            alvo_idx,
            lag,
        )


def train_model(train_loader, val_loader, n_hindcast_features, n_features_atm, run_dir,
                 hidden_channels=HIDDEN_CHANNELS, num_layers=NUM_LAYERS, kernel_size=KERNEL_SIZE,
                 lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    model = ConvLSTMForecaster(
        n_hindcast_features=n_hindcast_features, n_features_atm=n_features_atm,
        hidden_channels=hidden_channels, num_layers=num_layers, kernel_size=kernel_size,
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
        train_loss, n_train = 0.0, 0
        for hindcast, atm, lag, y, *_ in train_loader:
            hindcast, atm, lag, y = hindcast.to(DEVICE), atm.to(DEVICE), lag.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(hindcast, atm, lag)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * hindcast.size(0)
            n_train += hindcast.size(0)
        train_loss /= n_train

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for hindcast, atm, lag, y, *_ in val_loader:
                hindcast, atm, lag, y = hindcast.to(DEVICE), atm.to(DEVICE), lag.to(DEVICE), y.to(DEVICE)
                pred = model(hindcast, atm, lag)
                loss = criterion(pred, y)
                val_loss += loss.item() * hindcast.size(0)
                n_val += hindcast.size(0)
        val_loss /= n_val

        print(f"  epoca {epoch:2d}/{max_epochs} | treino MSE(mm/dia, res. reduzida) {train_loss:.4f} "
              f"| val MSE {val_loss:.4f}")
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_sem_melhora = 0
        else:
            epochs_sem_melhora += 1

        pd.DataFrame(history).to_csv(f"{run_dir}/history.csv", index=False)
        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(),
             "best_val_loss": best_val_loss, "best_epoch": best_epoch},
            f"{run_dir}/checkpoint_selecao_epocas.pt",
        )
        if epochs_sem_melhora >= patience:
            print(f"  early stopping na epoca {epoch} (melhor: {best_epoch})")
            break

    model.load_state_dict(best_state)
    return model, best_epoch, best_val_loss, history


@torch.no_grad()
def predict_loader(model, loader, lat: int, lon: int):
    """Roda o modelo num DataLoader inteiro e devolve as previsoes JA na grade cheia
    (upsample_to_full_res), junto dos indices de origem/alvo/lag de cada exemplo --
    usado tanto pra avaliar a validacao quanto (em tese) reaproveitavel pra outros usos."""
    model.eval()
    preds, origins, alvos, lags = [], [], [], []
    for hindcast, atm, lag, _y, o, alvo_idx, lag_int in loader:
        hindcast, atm, lag = hindcast.to(DEVICE), atm.to(DEVICE), lag.to(DEVICE)
        pred = model(hindcast, atm, lag).cpu().numpy()
        preds.append(upsample_to_full_res(pred, lat, lon))
        origins.append(o.numpy())
        alvos.append(alvo_idx.numpy())
        lags.append(lag_int.numpy())
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(origins), np.concatenate(alvos), np.concatenate(lags),
    )


def main(spatial_downsample: int = SPATIAL_DOWNSAMPLE, run_dir: str | None = None,
         validate_only: bool = False, smoke_test: bool = False,
         hidden_channels: int = HIDDEN_CHANNELS, num_layers: int = NUM_LAYERS,
         lr: float = LR, batch_size: int = BATCH_SIZE, max_epochs: int = MAX_EPOCHS):
    run_dir = run_dir or RUN_DIR
    os.makedirs(run_dir, exist_ok=True)
    if smoke_test:
        max_epochs = 1

    print("=== 1. Carregando dados ===")
    datasets = load_all_datasets()
    ds = stack_features(datasets)
    time_index = pd.DatetimeIndex(ds["time"].values)
    train_end_idx = time_index.get_indexer([pd.Timestamp(TRAIN_END)])[0]
    last_valid_idx = len(time_index) - 1
    lat, lon = ds.sizes["lat"], ds.sizes["lon"]
    print(f"  {len(time_index)} meses ({time_index[0].date()} a {time_index[-1].date()}), grade {lat}x{lon}")

    stats = compute_normalization_stats(ds, train_end=TRAIN_END)
    tp_raw_full = ds[TP_VAR].values.astype("float32").copy()  # (n_meses, lat, lon), mm/dia

    print(f"\n=== 2. Normalizando e reamostrando a grade (fator {spatial_downsample}, "
          f"{-(-lat // spatial_downsample)}x{-(-lon // spatial_downsample)}) ===")
    grids_norm: dict[str, np.ndarray] = {}
    for var in ALL_VARS:
        media, desvio = stats[var]
        bruto = ds[var].values.astype("float32")
        normalizado = (bruto - media) / desvio
        grids_norm[var] = downsample_grids(normalizado, spatial_downsample)
        del bruto, normalizado
        print(f"  {var}: {grids_norm[var].shape[1:]}")
    tp_raw_ds = downsample_grids(tp_raw_full, spatial_downsample)  # alvo do treino (mm/dia, ainda >=0)

    print("\n=== 3. Montando datasets (contrato origem + lag, montagem preguicosa) ===")
    train_ds = ConvLSTMDataset(grids_norm, tp_raw_ds, 0, train_end_idx, last_valid_idx=train_end_idx)
    val_ds = ConvLSTMDataset(grids_norm, tp_raw_ds, train_end_idx + 1, last_valid_idx, last_valid_idx=last_valid_idx)
    if smoke_test:
        train_ds.pairs = train_ds.pairs[:32]
        val_ds.pairs = val_ds.pairs[:16]
    print(f"  treino: {len(train_ds)} exemplos | validacao: {len(val_ds)} exemplos")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"\n=== 4. Treinando ConvLSTM (hidden_channels={hidden_channels}, num_layers={num_layers}) ===")
    model, best_epoch, best_val_loss, _history = train_model(
        train_loader, val_loader, n_hindcast_features=len(ALL_VARS), n_features_atm=len(FEATURE_VARS),
        run_dir=run_dir, hidden_channels=hidden_channels, num_layers=num_layers, lr=lr, max_epochs=max_epochs,
    )
    print(f"  melhor epoca (early stopping): {best_epoch} (val MSE res.reduzida={best_val_loss:.4f})")

    print("\n=== 5. Avaliando contra baselines na grade cheia (mm/dia) ===")
    pred_grid, origin_va, alvo_va, lag_inteiro_va = predict_loader(model, val_loader, lat, lon)
    true_grid = tp_raw_full[alvo_va]
    persist_grid = persistence_baseline(tp_raw_full[origin_va])
    meses_alvo = time_index.month.values[alvo_va]
    clim_grid = climatology_baseline(ds[TP_VAR].sel(time=slice(None, TRAIN_END)), list(meses_alvo))

    metrics_model = evaluate_predictions(true_grid, pred_grid, lags=lag_inteiro_va)
    metrics_persist = evaluate_predictions(true_grid, persist_grid, lags=lag_inteiro_va)
    metrics_clim = evaluate_predictions(true_grid, clim_grid, lags=lag_inteiro_va)

    print(f"  ConvLSTM      RMSE={metrics_model['rmse']:.3f}  MAE={metrics_model['mae']:.3f}  (mm/dia)")
    print(f"  Persistencia  RMSE={metrics_persist['rmse']:.3f}  MAE={metrics_persist['mae']:.3f}  (mm/dia)")
    print(f"  Climatologia  RMSE={metrics_clim['rmse']:.3f}  MAE={metrics_clim['mae']:.3f}  (mm/dia)")

    print("\n=== 5b. Salvando artefatos para visualizacao (models/) ===")
    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump({
            "spatial_downsample": spatial_downsample,
            "hyperparams": {"hidden_channels": hidden_channels, "num_layers": num_layers, "lr": lr},
            "best_epoch": best_epoch, "best_val_loss_res_reduzida": best_val_loss,
            "modelo": metrics_model, "persistencia": metrics_persist, "climatologia": metrics_clim,
        }, f, indent=2)

    lags_amostra = [l for l in [1, 3, 6, 12, 18, 24] if l in lag_inteiro_va]
    if lags_amostra:
        amostras = {"lat": ds["lat"].values, "lon": ds["lon"].values, "lags": np.array(lags_amostra)}
        for nome, grade in [("true", true_grid), ("pred", pred_grid), ("persist", persist_grid), ("clim", clim_grid)]:
            escolhidas = [grade[np.where(lag_inteiro_va == l)[0][0]] for l in lags_amostra]
            amostras[nome] = np.stack(escolhidas)
        np.savez_compressed(f"{run_dir}/sample_grids.npz", **amostras)
    print(f"  history.csv, metrics.json e sample_grids.npz salvos em {run_dir}/")

    if validate_only:
        print("\n[--validate-only] Pulando retreino com historico completo e geracao de submissao.")
        return None, None

    print("\n=== 6. Retreinando com todo o historico rotulado (1940-2022) ===")
    full_ds = ConvLSTMDataset(grids_norm, tp_raw_ds, 0, last_valid_idx, last_valid_idx=last_valid_idx)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)
    final_model = ConvLSTMForecaster(
        n_hindcast_features=len(ALL_VARS), n_features_atm=len(FEATURE_VARS),
        hidden_channels=hidden_channels, num_layers=num_layers, kernel_size=KERNEL_SIZE,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    retrain_history = []
    for epoch in range(1, best_epoch + 1):
        final_model.train()
        epoch_loss, n_seen = 0.0, 0
        for hindcast, atm, lag, y, *_ in full_loader:
            hindcast, atm, lag, y = hindcast.to(DEVICE), atm.to(DEVICE), lag.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = final_model(hindcast, atm, lag)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * hindcast.size(0)
            n_seen += hindcast.size(0)
        epoch_loss /= n_seen
        print(f"  epoca {epoch:2d}/{best_epoch} | MSE(mm/dia, res.reduzida) {epoch_loss:.4f}")
        retrain_history.append({"epoch": epoch, "train_mse": epoch_loss})
        pd.DataFrame(retrain_history).to_csv(f"{run_dir}/retrain_history.csv", index=False)

    torch.save(final_model.state_dict(), f"{run_dir}/model_final.pt")
    torch.save({"stats": stats, "spatial_downsample": spatial_downsample}, f"{run_dir}/stats.pt")
    print(f"  modelo final salvo em {run_dir}/")

    print("\n=== 7. Prevendo o teste real (2023-2024) ===")
    teste_ds = datasets["teste_features"]
    origem_idx = last_valid_idx
    hindcast_teste = np.stack(
        [grids_norm[v][origem_idx - HINDCAST_LEN + 1: origem_idx + 1] for v in ALL_VARS], axis=1
    )  # (H, C, h, w)
    n_meses_teste = teste_ds.sizes["time"]

    atm_teste_list = []
    for var in FEATURE_VARS:
        media, desvio = stats[var]
        bruto = teste_ds[var].values.astype("float32")
        normalizado = (bruto - media) / desvio
        atm_teste_list.append(downsample_grids(normalizado, spatial_downsample))
    atm_teste = np.stack(atm_teste_list, axis=1)  # (n_meses_teste, C_atm, h, w)

    hindcast_batch = np.repeat(hindcast_teste[None, :, :, :, :], n_meses_teste, axis=0)
    lag_batch = (teste_ds["lag_meses"].values / max(LAGS)).astype("float32")

    final_model.eval()
    with torch.no_grad():
        pred_ds_teste = final_model(
            torch.from_numpy(hindcast_batch).to(DEVICE),
            torch.from_numpy(atm_teste).to(DEVICE),
            torch.from_numpy(lag_batch).to(DEVICE),
        ).cpu().numpy()
    pred_grid_teste = np.clip(upsample_to_full_res(pred_ds_teste, lat, lon), 0, None)

    predictions_da = xr.DataArray(
        pred_grid_teste, dims=("time", "lat", "lon"),
        coords={"time": teste_ds["time"].values, "lat": teste_ds["lat"].values, "lon": teste_ds["lon"].values},
    )

    print("\n=== 8. Gerando submissao ===")
    os.makedirs("submissions", exist_ok=True)
    competition_path = download_competition_data()
    sample_path = os.path.join(competition_path, "sample_submission.csv")
    output_path = "submissions/submission_convlstm.csv"
    submission_df = build_submission(predictions_da, sample_path, output_path)
    print(f"  salvo em {output_path} ({len(submission_df)} linhas)")

    return predictions_da, submission_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-downsample", type=int, default=SPATIAL_DOWNSAMPLE,
                         help=f"Fator de reamostragem da grade (padrao: {SPATIAL_DOWNSAMPLE}; "
                              f"1 = resolucao cheia, bem mais lento em CPU).")
    parser.add_argument("--hidden-channels", type=int, default=HIDDEN_CHANNELS)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--run-dir", default=None, help=f"Pasta de saida (padrao: {RUN_DIR}).")
    parser.add_argument("--validate-only", action="store_true",
                         help="Para no passo 5 (avaliacao); pula retreino com historico completo e submissao.")
    parser.add_argument("--smoke-test", action="store_true",
                         help="1 epoca, poucos exemplos -- so pra checar que o pipeline roda de ponta a ponta.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        spatial_downsample=args.spatial_downsample, run_dir=args.run_dir,
        validate_only=args.validate_only, smoke_test=args.smoke_test,
        hidden_channels=args.hidden_channels, num_layers=args.num_layers,
        lr=args.lr, batch_size=args.batch_size, max_epochs=args.max_epochs,
    )
