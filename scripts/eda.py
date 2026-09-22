"""
Análise exploratória (tabular) dos dados da competição: estatísticas por variável,
percentual de dados faltantes e correlação entre variáveis.

Gera resumos em `eda_output/` (CSV) que também são reaproveitados pelo notebook
`eda.ipynb` para os gráficos, evitando reprocessar os arquivos NetCDF (grandes)
duas vezes.

Uso (a partir da raiz do repositorio):
    python3 -m scripts.eda
"""

import os

import pandas as pd
import xarray as xr

from scripts.download_data import download_competition_data, load_dataset_files

OUTPUT_DIR = "eda_output"


def variable_summary(datasets: dict[str, xr.Dataset]) -> pd.DataFrame:
    """Estatísticas básicas (min, max, média, desvio padrão, % faltante) por variável.

    Importante: média/desvio são calculados com `dtype="float64"`. Os arquivos vêm em
    float32 e, para arrays grandes (dezenas de milhões de pontos), a soma acumulada em
    float32 usada pelo `bottleneck` (dependência do xarray) perde precisão de forma
    catastrófica — chegamos a ver uma "média" de 109 para uma variável cujo mínimo real
    era 252. Forçar float64 na redução evita esse bug silencioso.
    """
    rows = []

    for name, ds in datasets.items():
        for var in ds.data_vars:
            da = ds[var]
            n_total = da.size
            n_nan = int(da.isnull().sum())

            rows.append(
                {
                    "dataset": name,
                    "variavel": var,
                    "min": float(da.min(skipna=True)),
                    "max": float(da.max(skipna=True)),
                    "media": float(da.mean(skipna=True, dtype="float64")),
                    "desvio_padrao": float(da.std(skipna=True, dtype="float64")),
                    "pct_faltante": 100 * n_nan / n_total if n_total else 0.0,
                    "tempo_inicio": str(ds["time"].min().values) if "time" in ds.coords else None,
                    "tempo_fim": str(ds["time"].max().values) if "time" in ds.coords else None,
                }
            )

    return pd.DataFrame(rows)


def spatial_mean_frame(datasets: dict[str, xr.Dataset], names: list[str]) -> pd.DataFrame:
    """Para cada variável dos datasets selecionados, calcula a média espacial (lat/lon)
    em cada instante de tempo, retornando um DataFrame indexado por tempo."""
    series = {}

    for name in names:
        ds = datasets[name]
        for var in ds.data_vars:
            spatial_dims = [d for d in ds[var].dims if d != "time"]
            series[var] = ds[var].mean(dim=spatial_dims, skipna=True, dtype="float64").to_series()

    return pd.DataFrame(series)


def main() -> None:
    path = download_competition_data()
    _dataframes, datasets = load_dataset_files(path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = variable_summary(datasets)
    print("\n=== Resumo por variável ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_DIR, "resumo_variaveis.csv"), index=False)

    treino_names = sorted(n for n in datasets if n.startswith("treino_"))
    if treino_names:
        treino_frame = spatial_mean_frame(datasets, treino_names)
        treino_frame.to_csv(os.path.join(OUTPUT_DIR, "series_temporais_treino.csv"))

        treino_corr = treino_frame.corr()
        print("\n=== Correlação entre variáveis (treino, média espacial) ===")
        print(treino_corr.to_string())
        treino_corr.to_csv(os.path.join(OUTPUT_DIR, "correlacao_treino.csv"))

    if "teste_features" in datasets:
        teste_frame = spatial_mean_frame(datasets, ["teste_features"])
        teste_frame.to_csv(os.path.join(OUTPUT_DIR, "series_temporais_teste.csv"))

        teste_corr = teste_frame.corr()
        print("\n=== Correlação entre variáveis (teste, média espacial) ===")
        print(teste_corr.to_string())
        teste_corr.to_csv(os.path.join(OUTPUT_DIR, "correlacao_teste.csv"))

    print(f"\nResumos salvos em '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
