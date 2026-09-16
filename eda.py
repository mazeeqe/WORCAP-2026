"""
Análise exploratória (tabular) dos dados da competição: estatísticas por variável,
percentual de dados faltantes e correlação entre variáveis.

Gera resumos em `eda_output/` (CSV) que também são reaproveitados pelo notebook
`eda.ipynb` para os gráficos, evitando reprocessar os arquivos NetCDF (grandes)
duas vezes.

Uso:
    python3 eda.py
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from download_data import download_competition_data, load_dataset_files

OUTPUT_DIR = "eda_output"
SEED = 42
N_AMOSTRA = 2_000_000  # pontos aleatorios (de dezenas de milhoes) p/ moda/assimetria/curtose
Z_LIMITE_OUTLIER = 2.5  # outlier = valor > media_global + Z_LIMITE_OUTLIER * desvio_global


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


def outlier_summary(datasets: dict[str, xr.Dataset], z_limite: float = Z_LIMITE_OUTLIER) -> pd.DataFrame:
    """Contagem/percentual de outliers por variavel (valor > media_global + z_limite*desvio,
    so o lado de cima -- chuva/valor anormalmente alto). Media/desvio em dtype=float64
    (ver nota em variable_summary sobre o bug de precisao do bottleneck)."""
    rows = []
    for name, ds in datasets.items():
        for var in ds.data_vars:
            if "time" not in ds[var].dims:
                continue
            da = ds[var]
            media = float(da.mean(skipna=True, dtype="float64"))
            desvio = float(da.std(skipna=True, dtype="float64"))
            limiar = media + z_limite * desvio
            n_validos = int(da.notnull().sum())
            n_outliers = int((da > limiar).sum())
            rows.append({
                "dataset": name, "variavel": var, "media_global": media, "desvio_global": desvio,
                "limiar": limiar, "n_outliers": n_outliers,
                "pct_outliers": 100 * n_outliers / n_validos if n_validos else 0.0,
            })
    return pd.DataFrame(rows)


def outliers_by_year(ds: xr.Dataset, var: str, z_limite: float = Z_LIMITE_OUTLIER) -> pd.DataFrame:
    """Media e numero de outliers POR ANO pra uma variavel de um Dataset com dim 'time'.

    O limiar de outlier (media + z_limite*desvio) e calculado sobre o dataset INTEIRO
    (nao recalculado a cada ano), pra ter uma regua fixa e comparavel entre anos --
    um limiar recalculado por ano mudaria de ano pra ano, invalidando a comparacao.
    """
    da = ds[var]
    media_global = float(da.mean(skipna=True, dtype="float64"))
    desvio_global = float(da.std(skipna=True, dtype="float64"))
    limiar = media_global + z_limite * desvio_global

    media_por_ano = da.groupby("time.year").mean(dim=[d for d in da.dims], skipna=True, dtype="float64")
    outliers_por_ano = (da > limiar).groupby("time.year").sum(dim=[d for d in da.dims])
    pontos_por_ano = da.notnull().groupby("time.year").sum(dim=[d for d in da.dims])

    df = pd.DataFrame({
        "media": media_por_ano.values,
        "n_outliers": outliers_por_ano.values.astype(int),
        "n_pontos": pontos_por_ano.values.astype(int),
    }, index=media_por_ano.year.values)
    df.index.name = "ano"
    df["pct_outliers"] = 100 * df["n_outliers"] / df["n_pontos"]
    df.attrs["media_global"] = media_global
    df.attrs["desvio_global"] = desvio_global
    df.attrs["limiar"] = limiar
    return df


def descriptive_stats_sample(datasets: dict[str, xr.Dataset], names: list[str],
                              n_amostra: int = N_AMOSTRA, seed: int = SEED) -> pd.DataFrame:
    """Media, mediana, moda, desvio, assimetria e curtose por variavel, calculados sobre
    uma AMOSTRA aleatoria (moda/assimetria/curtose nao tem uma reducao vetorizada barata
    em xarray puro pra dezenas de milhoes de pontos; a amostra e reprodutivel via seed)."""
    rng = np.random.default_rng(seed)
    variaveis = {}
    for name in names:
        ds = datasets[name]
        for var in ds.data_vars:
            if "time" not in ds[var].dims:
                continue
            variaveis[var] = ds[var]

    if not variaveis:
        return pd.DataFrame()

    exemplo = next(iter(variaveis.values()))
    total = exemplo.size
    n = min(n_amostra, total)
    idx_flat = rng.choice(total, size=n, replace=False)
    idx_multi = np.unravel_index(idx_flat, exemplo.shape)

    colunas = {var: da.values[idx_multi] for var, da in variaveis.items()}
    df = pd.DataFrame(colunas).dropna()

    resumo = pd.DataFrame({
        "media": df.mean(), "mediana": df.median(),
        "moda": df.round(1).mode().iloc[0],
        "desvio_padrao": df.std(), "min": df.min(), "max": df.max(),
        "assimetria": df.skew(), "curtose": df.kurt(),
    })
    resumo.attrs["n_amostra"] = len(df)
    return resumo


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

    outliers = outlier_summary(datasets)
    print(f"\n=== Outliers por variável (z > {Z_LIMITE_OUTLIER}) ===")
    print(outliers.to_string(index=False))
    outliers.to_csv(os.path.join(OUTPUT_DIR, "outliers_resumo.csv"), index=False)

    if "treino_tp" in datasets:
        tp_por_ano = outliers_by_year(datasets["treino_tp"], "tp")
        print(f"\n=== tp: média e outliers por ano (z > {Z_LIMITE_OUTLIER}, "
              f"limiar={tp_por_ano.attrs['limiar']:.4f} mm/dia) ===")
        print(tp_por_ano.to_string())
        tp_por_ano.to_csv(os.path.join(OUTPUT_DIR, "tp_outliers_por_ano.csv"))

    if treino_names:
        descritiva = descriptive_stats_sample(datasets, treino_names)
        print(f"\n=== Estatística descritiva (amostra, n={descritiva.attrs.get('n_amostra', '?')}) ===")
        print(descritiva.to_string())
        descritiva.to_csv(os.path.join(OUTPUT_DIR, "estatistica_descritiva_amostra.csv"))

    print(f"\nResumos salvos em '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
