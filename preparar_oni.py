"""
Converte a tabela bruta do ONI (Oceanic Nino Index — medida oficial de
El Nino/La Nina, publicada pelo NOAA CPC) numa serie MENSAL.

Fonte: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
(dado publico, baixado por verificar_jan2024/download antes deste script
via: curl -o oni_raw.txt https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt)

O arquivo original tem 1 linha por "temporada" de 3 meses (media movel),
ex: "DJF 1950 25.01 -1.32" = media Dez/1949-Jan/1950-Fev/1950, coluna ANOM
= anomalia de temperatura (o valor que define El Nino/La Nina: ANOM >= +0.5
por 5 temporadas seguidas = El Nino; <= -0.5 = La Nina).

Convertemos pra 1 valor por MES: cada temporada de 3 letras representa o
mes central dela (ex: DJF -> Janeiro do ano informado na coluna YR).
"""
import pandas as pd
import os

PASTA = os.path.dirname(os.path.abspath(__file__))
CAMINHO_BRUTO = os.path.join(PASTA, 'oni_raw.txt')
CAMINHO_SAIDA = os.path.join(PASTA, 'oni_mensal.csv')

# temporada de 3 letras -> mes central (ver docstring)
MES_CENTRAL = {
    'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
    'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12,
}


def main():
    df = pd.read_csv(CAMINHO_BRUTO, sep=r'\s+')
    df['mes'] = df['SEAS'].map(MES_CENTRAL)
    df['data'] = pd.to_datetime(dict(year=df['YR'], month=df['mes'], day=1))
    df = df[['data', 'ANOM']].rename(columns={'ANOM': 'oni'}).sort_values('data')
    df.to_csv(CAMINHO_SAIDA, index=False)
    print(f"OK: {len(df)} meses de ONI ({df['data'].min().date()} a {df['data'].max().date()}) -> {CAMINHO_SAIDA}")


if __name__ == '__main__':
    main()
