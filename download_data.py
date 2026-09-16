"""
Baixa os dados da competição Kaggle "previsao-climatica-de-precipitacao-sobre-a-america-do-sul"
e carrega os arquivos encontrados: .csv em DataFrames do pandas e .nc (NetCDF) em Datasets
do xarray.

Pré-requisitos:
    pip install kagglehub pandas xarray netCDF4
    Credenciais da API do Kaggle configuradas em uma das formas suportadas pelo kagglehub:
      - ~/.kaggle/access_token (token novo formato "KGAT_...", uma linha só com o token)
      - variável de ambiente KAGGLE_API_TOKEN
      - ~/.kaggle/kaggle.json legado com {"username": "...", "key": "..."}
    Também é preciso ter aceitado as regras da competição em kaggle.com antes de baixar.
"""

import os

import pandas as pd
import xarray as xr

COMPETITION = "previsao-climatica-de-precipitacao-sobre-a-america-do-sul"


def download_competition_data(competition: str = COMPETITION) -> str:
    """Baixa (ou reaproveita o cache local) os arquivos da competição e retorna o path."""
    import kagglehub

    path = kagglehub.competition_download(competition)
    print(f"Path to competition files: {path}")
    return path


def load_dataset_files(path: str) -> tuple[dict[str, pd.DataFrame], dict[str, xr.Dataset]]:
    """Percorre o diretório baixado carregando .csv (pandas) e .nc (xarray)."""
    dataframes: dict[str, pd.DataFrame] = {}
    datasets: dict[str, xr.Dataset] = {}

    for root, _dirs, files in sorted(os.walk(path)):
        for filename in sorted(files):
            file_path = os.path.join(root, filename)
            key = os.path.splitext(filename)[0]
            lower_name = filename.lower()

            try:
                if lower_name.endswith(".csv"):
                    df = dataframes[key] = pd.read_csv(file_path)
                    print(f"Carregado '{key}' (csv): {df.shape[0]} linhas x {df.shape[1]} colunas")
                elif lower_name.endswith(".nc"):
                    ds = datasets[key] = xr.open_dataset(file_path)
                    dims = ", ".join(f"{d}={n}" for d, n in ds.sizes.items())
                    print(f"Carregado '{key}' (netcdf): dims=({dims}), variáveis={list(ds.data_vars)}")
                else:
                    continue
            except Exception as exc:
                print(f"Falha ao ler {file_path}: {exc}")

    return dataframes, datasets


def main() -> tuple[dict[str, pd.DataFrame], dict[str, xr.Dataset]]:
    path = download_competition_data()
    dataframes, datasets = load_dataset_files(path)

    for name, df in dataframes.items():
        print(f"\n--- {name} (csv) ---")
        print(df.head())

    for name, ds in datasets.items():
        print(f"\n--- {name} (netcdf) ---")
        print(ds)

    if not dataframes and not datasets:
        print("Nenhum arquivo .csv ou .nc encontrado no diretório baixado.")

    return dataframes, datasets


if __name__ == "__main__":
    main()
