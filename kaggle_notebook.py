"""Executor para Kaggle Notebook.

Antes de executar, anexe a competição em Add Input. O script localiza os 13
arquivos em /kaggle/input, treina o modelo corrigido e salva a submissão em
/kaggle/working.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from download_data import EXPECTED_FILES, download_competition_data
from src.train_pca_lstm import main as train_pca_lstm

EXPECTED_ROWS = 1_885_464


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()

    data_path = Path(download_competition_data())
    found = {item.name for item in data_path.iterdir() if item.is_file()}
    missing = EXPECTED_FILES - found
    if missing:
        raise SystemExit(f"Dataset incompleto: {', '.join(sorted(missing))}")

    print(f"Dataset oficial validado: {data_path}")
    _, submission = train_pca_lstm()
    if len(submission) != EXPECTED_ROWS:
        raise SystemExit(
            f"Submissão possui {len(submission):,} linhas; esperado {EXPECTED_ROWS:,}."
        )
    if submission["id"].duplicated().any():
        raise SystemExit("Submissão contém IDs duplicados.")
    if not submission["tp_mm_day"].notna().all() or (submission["tp_mm_day"] < 0).any():
        raise SystemExit("Submissão contém previsão ausente ou negativa.")

    args.output.mkdir(parents=True, exist_ok=True)
    source = Path("submissions/submission_pca_lstm.csv")
    target = args.output / source.name
    shutil.copy2(source, target)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": "PCA/EOF + LSTM",
        "temporal_contract": "M→M+1",
        "rows": len(submission),
        "sha256": file_sha256(target),
        "dataset_path": str(data_path),
        "official_files": sorted(EXPECTED_FILES),
        "leaderboard_score": None,
    }
    manifest_path = args.output / "submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Submissão pronta: {target}")
    print(f"Manifesto: {manifest_path}")


if __name__ == "__main__":
    main()
