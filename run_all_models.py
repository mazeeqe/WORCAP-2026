"""Treina todos os modelos disponiveis (e suas variacoes), em sequencia.

Uso (a partir da raiz do repositorio):
    python3 run_all_models.py                                        # todos os modelos/variacoes
    python3 run_all_models.py --models pca_lstm                      # so um modelo (todas as variacoes dele)
    python3 run_all_models.py --variations pca pls_concurrent        # so essas variacoes (de qualquer modelo que as tenha)
    python3 run_all_models.py --variations pls_lagged --pls-lag-shift 2

Cada variacao roda o modulo de treino daquele modelo (`python3 -m <MODEL_RUNNERS[modelo].module>
--reduction <variacao>`) num processo separado (memoria isolada entre rodadas) e ja
salva sozinha, dentro da sua pasta em `models/<variacao>_lstm_run1/`:
    - checkpoint_selecao_epocas.pt / history.csv     (backup por epoca, fase de selecao)
    - checkpoint_retrain_final.pt / retrain_history.csv (backup por epoca, retreino final)
    - train.log                                       (log completo daquela execucao)
    - metrics.json, sample_grids.npz, model_final.pt, reduction_and_stats.joblib

Para adicionar um novo modelo (ex.: ConvLSTM) a esta fila, basta registra-lo em
MODEL_RUNNERS abaixo, assim que ele tiver um script de treino proprio (ver
src/models/pca_lstm/train.py como referencia de contrato: aceitar --reduction/nome
da variacao, --run-dir, e salvar metrics.json com a chave "modelo": {"rmse", "mae"}).

Este script so orquestra as execucoes e, ao final, imprime e salva um resumo
comparando RMSE/MAE de cada variacao (`models/run_all_summary.json`). Se uma
variacao falhar, as outras continuam rodando (a menos que --stop-on-error seja
passado) - o resumo final mostra quais falharam.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.models.pca_lstm.train import N_JOBS_REDUCTION, PLS_LAG_SHIFT, REDUCTION_METHODS, RUN_DIRS

# Registro dos modelos disponiveis: cada um roda como `python3 -m <module> --reduction <variacao> ...`.
# `variations` sao os valores aceitos por --reduction daquele modelo; `run_dirs` mapeia
# variacao -> pasta de artefatos (usada so pra achar o metrics.json depois do treino).
MODEL_RUNNERS = {
    "pca_lstm": {
        "module": "src.models.pca_lstm.train",
        "variations": REDUCTION_METHODS,
        "run_dirs": RUN_DIRS,
    },
    # "convlstm": {...},  # TODO: registrar aqui quando existir um script de treino
    #                       (hoje so ha o modelo em si, sem script de treino - ver conversa)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_RUNNERS),
        default=list(MODEL_RUNNERS),
        help="Quais modelos treinar (padrao: todos os registrados em MODEL_RUNNERS).",
    )
    parser.add_argument(
        "--variations",
        nargs="+",
        default=None,
        help="Quais variacoes treinar, de qualquer modelo selecionado (padrao: todas as variacoes "
        "de cada modelo). Ex.: --variations pca pls_lagged",
    )
    parser.add_argument(
        "--pls-lag-shift",
        type=int,
        default=PLS_LAG_SHIFT,
        help="Repassado para --pls-lag-shift em cada treino com reducao pls_lagged (padrao: %(default)s).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Para a fila assim que uma variacao falhar, em vez de continuar com as demais.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS_REDUCTION,
        help="Repassado para --n-jobs em cada treino (paralelismo do ajuste PCA/PLS por variavel, "
        "padrao: %(default)s = todos os nucleos). Reduza se a maquina tiver pouca RAM - cada "
        "worker mantem sua propria copia da grade normalizada (ver fit_reduction_per_variable).",
    )
    return parser.parse_args()


def run_one(model: str, module: str, variation: str, run_dir: str, pls_lag_shift: int, n_jobs: int) -> dict:
    cmd = [
        sys.executable, "-m", module,
        "--reduction", variation,
        "--pls-lag-shift", str(pls_lag_shift),
        "--n-jobs", str(n_jobs),
    ]
    print(f"\n{'#' * 70}\n# Iniciando '{model}/{variation}': {' '.join(cmd)}\n{'#' * 70}\n")

    t0 = time.time()
    resultado = subprocess.run(cmd)
    duracao = time.time() - t0

    status = "ok" if resultado.returncode == 0 else "falhou"
    print(f"\n>>> '{model}/{variation}' {status} em {duracao:.0f}s (returncode={resultado.returncode})")

    resumo = {
        "model": model,
        "variation": variation,
        "status": status,
        "returncode": resultado.returncode,
        "duracao_s": round(duracao, 1),
    }

    metrics_path = Path(run_dir) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        resumo["rmse"] = metrics["modelo"]["rmse"]
        resumo["mae"] = metrics["modelo"]["mae"]

    return resumo


def main():
    args = parse_args()
    resumos = []

    fila = []
    for model in args.models:
        runner = MODEL_RUNNERS[model]
        variations = args.variations if args.variations is not None else runner["variations"]
        for variation in variations:
            if variation not in runner["run_dirs"]:
                continue  # variacao nao pertence a esse modelo (ex.: --variations misturando modelos)
            fila.append((model, runner["module"], variation, runner["run_dirs"][variation]))

    for model, module, variation, run_dir in fila:
        resumo = run_one(model, module, variation, run_dir, args.pls_lag_shift, args.n_jobs)
        resumos.append(resumo)
        if resumo["status"] == "falhou" and args.stop_on_error:
            print(f"\nParando a fila: '{model}/{variation}' falhou e --stop-on-error foi passado.")
            break

    print(f"\n{'=' * 70}\nResumo da fila de treinos\n{'=' * 70}")
    for r in resumos:
        rotulo = f"{r['model']}/{r['variation']}"
        if "rmse" in r:
            print(f"  {rotulo:<24} {r['status']:<8} {r['duracao_s']:>7.0f}s  RMSE={r['rmse']:.3f}  MAE={r['mae']:.3f}")
        else:
            print(f"  {rotulo:<24} {r['status']:<8} {r['duracao_s']:>7.0f}s  (sem metrics.json)")

    Path("models").mkdir(exist_ok=True)
    with open("models/run_all_summary.json", "w") as f:
        json.dump(resumos, f, indent=2)
    print("\nResumo salvo em models/run_all_summary.json")

    if any(r["status"] == "falhou" for r in resumos):
        sys.exit(1)


if __name__ == "__main__":
    main()
