"""Sweep em grade (produto cartesiano) de hiperparametros do LSTM (Modelo A), para
um unico metodo de reducao dimensional.

Uso (a partir da raiz do repositorio):
    python3 run_hparam_sweep.py                                  # metodo = melhor RMSE em
                                                                    # models/run_all_summary.json
    python3 run_hparam_sweep.py --method pls_concurrent
    python3 run_hparam_sweep.py --method pca --lrs 5e-4 1e-3 --hidden-sizes 64 128 --dropouts 0.1

Grade padrao ("pequena"): lr x hidden_size x dropout = 3 x 3 x 2 = 18 execucoes.

Cada combinacao roda `python3 -m src.models.pca_lstm.train --reduction <method> --lr <lr>
--hidden-size <hs> --dropout <do> --run-dir <pasta>` num processo separado (mesma
logica de isolamento do run_all_models.py), salvando os artefatos em
`models/hparam_sweep/<method>_lr<lr>_hs<hs>_do<do>/`.

IMPORTANTE: a escolha automatica do "melhor" metodo le
`models/run_all_summary.json`, que so fica atualizado depois de rodar
`run_all_models.py`. Se esse arquivo foi gerado antes de uma mudanca na reducao
dimensional (ex.: criterio de variancia explicada em SpatialPCA/SpatialPLS), ele
pode nao refletir o metodo realmente melhor sob o pipeline atual - nesse caso, rode
`run_all_models.py` de novo antes deste script, ou passe --method explicitamente.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

from src.models.pca_lstm.train import DROPOUT, HIDDEN_SIZE, LR, N_JOBS_REDUCTION, PLS_LAG_SHIFT, REDUCTION_METHODS

SUMMARY_PATH = Path("models/run_all_summary.json")
SWEEP_DIR = Path("models/hparam_sweep")

DEFAULT_LRS = [5e-4, 1e-3, 2e-3]
DEFAULT_HIDDEN_SIZES = [64, 128, 256]
DEFAULT_DROPOUTS = [0.1, 0.3]


def pick_best_method() -> str:
    if not SUMMARY_PATH.exists():
        print(
            f"aviso: {SUMMARY_PATH} nao existe (rode run_all_models.py antes para escolher o metodo "
            f"automaticamente); usando 'pca' como padrao."
        )
        return "pca"
    with open(SUMMARY_PATH) as f:
        resumos = json.load(f)
    com_rmse = [r for r in resumos if "rmse" in r]
    if not com_rmse:
        print(f"aviso: {SUMMARY_PATH} nao tem execucoes com RMSE; usando 'pca' como padrao.")
        return "pca"
    melhor = min(com_rmse, key=lambda r: r["rmse"])
    # "variation" e o nome novo (run_all_models.py); "method" e o formato antigo (run_experiments.py) -
    # aceita os dois pra nao quebrar se o resumo ainda nao foi regerado pelo script novo.
    metodo = melhor.get("variation", melhor.get("method"))
    print(
        f"Metodo escolhido automaticamente por RMSE em {SUMMARY_PATH}: '{metodo}' "
        f"(RMSE={melhor['rmse']:.3f}). Se esse resumo estiver desatualizado, passe --method para forcar."
    )
    return metodo


def fmt(x: float) -> str:
    """Formata um numero para uso em nome de pasta (sem ponto/notacao cientifica)."""
    return f"{x:g}".replace(".", "p").replace("-", "m")


def run_dir_for(method: str, lr: float, hidden_size: int, dropout: float) -> Path:
    return SWEEP_DIR / f"{method}_lr{fmt(lr)}_hs{hidden_size}_do{fmt(dropout)}"


def run_one(method: str, pls_lag_shift: int, lr: float, hidden_size: int, dropout: float, n_jobs: int) -> dict:
    run_dir = run_dir_for(method, lr, hidden_size, dropout)
    cmd = [
        sys.executable,
        "-m",
        "src.models.pca_lstm.train",
        "--reduction",
        method,
        "--pls-lag-shift",
        str(pls_lag_shift),
        "--lr",
        str(lr),
        "--hidden-size",
        str(hidden_size),
        "--dropout",
        str(dropout),
        "--run-dir",
        str(run_dir),
        "--n-jobs",
        str(n_jobs),
    ]
    print(f"\n{'#' * 70}\n# lr={lr} hidden_size={hidden_size} dropout={dropout}: {' '.join(cmd)}\n{'#' * 70}\n")

    t0 = time.time()
    resultado = subprocess.run(cmd)
    duracao = time.time() - t0

    status = "ok" if resultado.returncode == 0 else "falhou"
    print(f"\n>>> lr={lr} hidden_size={hidden_size} dropout={dropout} {status} em {duracao:.0f}s")

    resumo = {
        "method": method,
        "lr": lr,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "run_dir": str(run_dir),
        "status": status,
        "returncode": resultado.returncode,
        "duracao_s": round(duracao, 1),
    }

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        resumo["rmse"] = metrics["modelo"]["rmse"]
        resumo["mae"] = metrics["modelo"]["mae"]

    return resumo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--method",
        choices=REDUCTION_METHODS,
        default=None,
        help="Metodo de reducao dimensional a usar (padrao: melhor RMSE em models/run_all_summary.json).",
    )
    parser.add_argument(
        "--pls-lag-shift",
        type=int,
        default=PLS_LAG_SHIFT,
        help="So usado se --method (ou o escolhido automaticamente) for pls_lagged (padrao: %(default)s).",
    )
    parser.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS, help="Valores de LR a testar.")
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=DEFAULT_HIDDEN_SIZES, help="Valores de hidden_size a testar."
    )
    parser.add_argument(
        "--dropouts", type=float, nargs="+", default=DEFAULT_DROPOUTS, help="Valores de dropout a testar."
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Para a fila assim que uma combinacao falhar, em vez de continuar com as demais.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS_REDUCTION,
        help="Repassado para --n-jobs em cada treino (so importa em caso de cache miss do "
        "ajuste de reducao - ver load_or_fit_reduction; com cache, cada combinacao do "
        "sweep usa o mesmo metodo/pls-lag-shift, entao so a 1a chamada pagaria esse custo "
        "de qualquer forma). Padrao: %(default)s.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    method = args.method or pick_best_method()

    combos = list(product(args.lrs, args.hidden_sizes, args.dropouts))
    print(
        f"\nMetodo: {method} | grade: {len(args.lrs)} lr x {len(args.hidden_sizes)} hidden_size x "
        f"{len(args.dropouts)} dropout = {len(combos)} execucoes\n"
        f"lrs={args.lrs} hidden_sizes={args.hidden_sizes} dropouts={args.dropouts}\n"
        f"(padrao do repositorio: lr={LR}, hidden_size={HIDDEN_SIZE}, dropout={DROPOUT})"
    )

    resumos = []
    for lr, hidden_size, dropout in combos:
        resumo = run_one(method, args.pls_lag_shift, lr, hidden_size, dropout, args.n_jobs)
        resumos.append(resumo)
        SWEEP_DIR.mkdir(parents=True, exist_ok=True)
        with open(SWEEP_DIR / "sweep_summary.json", "w") as f:
            json.dump(resumos, f, indent=2)
        if resumo["status"] == "falhou" and args.stop_on_error:
            print(f"\nParando a fila: combinacao lr={lr} hidden_size={hidden_size} dropout={dropout} falhou.")
            break

    print(f"\n{'=' * 70}\nResumo do sweep de hiperparametros ({method})\n{'=' * 70}")
    ordenados = sorted(resumos, key=lambda r: r.get("rmse", float("inf")))
    for r in ordenados:
        if "rmse" in r:
            print(
                f"  lr={r['lr']:<8} hidden_size={r['hidden_size']:<4} dropout={r['dropout']:<4} "
                f"{r['status']:<8} {r['duracao_s']:>7.0f}s  RMSE={r['rmse']:.3f}  MAE={r['mae']:.3f}"
            )
        else:
            print(
                f"  lr={r['lr']:<8} hidden_size={r['hidden_size']:<4} dropout={r['dropout']:<4} "
                f"{r['status']:<8} {r['duracao_s']:>7.0f}s  (sem metrics.json)"
            )

    print(f"\nResumo salvo em {SWEEP_DIR / 'sweep_summary.json'}")

    if any(r["status"] == "falhou" for r in resumos):
        sys.exit(1)


if __name__ == "__main__":
    main()
