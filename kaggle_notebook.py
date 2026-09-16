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


def publish_to_gcp(
    project: str,
    bucket_name: str,
    table: str | None,
    submission_path: Path,
    manifest_path: Path,
    manifest: dict,
) -> None:
    """Publica artefatos no GCS e, opcionalmente, proveniência no BigQuery."""
    try:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        credential = user_secrets.get_gcloud_credential()
        user_secrets.set_tensorflow_credential(credential)
        print("Credencial Google Cloud vinculada pelo Kaggle Secrets.")
    except ImportError:
        # Fora do Kaggle, google-cloud usa Application Default Credentials.
        pass

    try:
        from google.cloud import bigquery, storage
    except ImportError as exc:
        raise SystemExit(
            "Para publicar no GCP, instale: pip install -r requirements-gcp.txt"
        ) from exc

    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    prefix = f"worcap/{manifest['sha256'][:12]}"
    bucket.blob(f"{prefix}/{submission_path.name}").upload_from_filename(submission_path)
    bucket.blob(f"{prefix}/{manifest_path.name}").upload_from_filename(manifest_path)
    print(f"Artefatos publicados em gs://{bucket_name}/{prefix}/")

    if table:
        bigquery_client = bigquery.Client(project=project)
        errors = bigquery_client.insert_rows_json(table, [manifest])
        if errors:
            raise RuntimeError(f"Falha ao registrar manifesto no BigQuery: {errors}")
        print(f"Proveniência registrada em {table}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--gcp-project")
    parser.add_argument("--gcs-bucket")
    parser.add_argument("--bigquery-table")
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

    if bool(args.gcp_project) != bool(args.gcs_bucket):
        raise SystemExit("Use --gcp-project e --gcs-bucket juntos.")
    if args.gcp_project and args.gcs_bucket:
        publish_to_gcp(
            args.gcp_project,
            args.gcs_bucket,
            args.bigquery_table,
            target,
            manifest_path,
            manifest,
        )


if __name__ == "__main__":
    main()
