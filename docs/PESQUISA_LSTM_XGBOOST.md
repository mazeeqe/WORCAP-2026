# Pesquisa: Ensemble LSTM + XGBoost para Previsão Climática

Resumo e adaptação de uma conversa exportada do Gemini
(`Previsão Climática com LSTM e XGBoost.pdf`) sobre um pipeline que combina LSTM e
XGBoost via stacking para prever variáveis climáticas de longo prazo (sazonal/mensal).
Mantido aqui como referência técnica para o time — ver a seção 6 para como isso se
encaixa (ou não) no [PLANO_TRABALHO.md](PLANO_TRABALHO.md).

## 1. Ideia central: ensemble por stacking/blending

Em vez de escolher entre LSTM ou XGBoost, usar os dois e combinar as saídas:

- **LSTM**: captura memória temporal contínua, ciclos sazonais e efeitos de longa
  duração (ex.: evolução do El Niño/La Niña ao longo de meses).
- **XGBoost**: mapeia relações não-lineares complexas em dados tabulares (pressão x
  umidade x vento x geopotencial) num instante específico.
- **Meta-modelo** (ex.: `Ridge`): treinado só sobre as previsões dos dois modelos no
  conjunto de **validação**, aprende o peso ideal de cada um e gera a previsão final.
  Nunca treinar o meta-modelo com dados de teste (vazamento).

Alternativa mencionada (não detalhada no material): usar o hidden state da LSTM como
"feature extraída" e concatenar com as variáveis físicas antes de entregar tudo ao
XGBoost, em vez de combinar as previsões finais.

## 2. Engenharia de features

- **Lags de teleconexões**: o El Niño/La Niña afeta o clima regional com atraso de
  3 a 6 meses. Criar colunas como `oni_lag_1`, `oni_lag_3`, `oni_lag_6`, `oni_lag_12`
  (e o mesmo padrão de lag para as próprias variáveis físicas).
- **Sazonalidade cíclica**: codificar o mês do ano como seno/cosseno em vez de inteiro
  1-12, para o modelo entender que dezembro está "colado" a janeiro:

  ```python
  mes_seno = sin(2 * pi * mes / 12)
  mes_cosseno = cos(2 * pi * mes / 12)
  ```

- **Formato de entrada**: LSTM espera 3D `(amostras, time_steps, features)`
  (ex.: 12 meses de histórico → `(N, 12, n_features)`); XGBoost espera 2D
  `(amostras, features_com_lag)`.

## 3. Por que prever anomalias em vez de valores brutos

1. Remove a sazonalidade trivial (o modelo não gasta capacidade aprendendo que
   "janeiro é quente" — isso já é a climatologia; o que importa é o desvio).
2. Lida melhor com não-estacionariedade (a média global/local mudou entre 1940 e hoje).
3. Coloca variáveis de escalas diferentes numa mesma faixa comparável.

Cálculo (climatologia de referência por mês, usando só o período de treino):

```
X̄_mês = média histórica de X naquele mês do calendário (no período base)
A_ano,mês = X_ano,mês - X̄_mês
```

Para variáveis de alta variância (precipitação), usar **anomalia padronizada**
(z-score: `(X - X̄) / desvio_padrão_do_mês`) ou **anomalia percentual**
(`(X / X̄ - 1) * 100`) — evita que meses secos (onde 10mm já é um desvio enorme)
sejam ofuscados por meses chuvosos.

Reconstrução do valor real após a previsão: `valor_previsto = anomalia_prevista + X̄_mês_alvo`.

## 4. Alvo alternativo: índices SPI / SPEI

- **SPI (Standardized Precipitation Index)**: transforma a chuva bruta (distribuição
  assimétrica tipo Gamma) numa distribuição normal padrão (média 0, desvio 1) por
  escala temporal (SPI-1, SPI-3, SPI-6...). Só usa precipitação.
- **SPEI**: mesma ideia, mas sobre o **balanço hídrico** `D = P - PET` (precipitação
  menos evapotranspiração potencial). Captura melhor o aquecimento global porque
  "secas quentes" (chuva normal, mas calor extremo evapora a umidade) aparecem como
  seca no SPEI e passariam despercebidas no SPI.
- Ambos têm a vantagem de serem invariantes a escala/região (SPI +1.5 significa a
  mesma coisa no deserto ou na floresta) e de terem distribuição bem comportada para
  regressão (sem outliers assimétricos extremos).
- PET pelo método de Hargreaves só precisa de temperatura (min/max/média) e latitude —
  não precisa de tudo que o SPEI "completo" pede.

Tabela de classificação do SPI/SPEI:

| Valor | Classificação | Probabilidade |
|---|---|---|
| ≥ +2.00 | Extremamente chuvoso | ≈ 2.3% |
| +1.50 a +1.99 | Muito chuvoso | ≈ 4.4% |
| +1.00 a +1.49 | Moderadamente chuvoso | ≈ 9.2% |
| -0.99 a +0.99 | Próximo da normalidade | ≈ 68.2% |
| -1.00 a -1.49 | Seca moderada | ≈ 9.2% |
| -1.50 a -1.99 | Seca severa | ≈ 4.4% |
| ≤ -2.00 | Extremamente seco | ≈ 2.3% |

## 5. Código de referência (consolidado do PDF)

Validação temporal estrita (nunca k-fold tradicional/shuffle — ordem temporal importa),
split em treino/validação/teste sequenciais, XGBoost + LSTM treinados separadamente e
combinados por um meta-modelo Ridge treinado só na validação:

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


# --- Bloco 1: engenharia de features (lags + sazonalidade ciclica) ---
def criar_features_temporais(df, coluna_alvo="SPI_3"):
    dados = df.copy()
    dados["mes_seno"] = np.sin(2 * np.pi * dados.index.month / 12.0)
    dados["mes_cosseno"] = np.cos(2 * np.pi * dados.index.month / 12.0)

    colunas_para_atrasar = [
        "oni_index", "geopotencial_anomaly", "pressao_anomaly",
        "umidade_anomaly", "vento_u", "vento_v", coluna_alvo,
    ]
    for coluna in colunas_para_atrasar:
        if coluna in dados.columns:
            for meses_atras in [1, 2, 3, 6, 12]:
                dados[f"{coluna}_atras_{meses_atras}m"] = dados[coluna].shift(meses_atras)

    return dados.dropna()


# --- Bloco 2: dataset PyTorch para a LSTM ---
class DatasetClimatico(Dataset):
    def __init__(self, X_sequencias, y_valores):
        self.X = torch.tensor(X_sequencias, dtype=torch.float32)
        self.y = torch.tensor(y_valores, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def montar_janelas_3d(X_dados, y_dados, tamanho_janela=12):
    X_seq, y_seq = [], []
    for i in range(len(X_dados) - tamanho_janela):
        X_seq.append(X_dados[i : i + tamanho_janela])
        y_seq.append(y_dados[i + tamanho_janela])
    return np.array(X_seq), np.array(y_seq)


# --- Bloco 3: arquitetura da LSTM ---
class RedeClimaticaLSTM(nn.Module):
    def __init__(self, quantidade_variaveis, neuronios=64, camadas=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=quantidade_variaveis,
            hidden_size=neuronios,
            num_layers=camadas,
            batch_first=True,
            dropout=0.2 if camadas > 1 else 0.0,
        )
        self.camada_saida = nn.Sequential(
            nn.Linear(neuronios, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        saida_lstm, _ = self.lstm(x)
        return self.camada_saida(saida_lstm[:, -1, :])  # ultimo passo da janela


# --- Bloco 4: pipeline completo (features -> split -> XGBoost + LSTM -> meta-modelo) ---
def treinar_sistema_previsao(df, coluna_alvo="SPI_3", janela_meses=12):
    df_processado = criar_features_temporais(df, coluna_alvo=coluna_alvo)

    colunas_x = [c for c in df_processado.columns if c != coluna_alvo]
    X_bruto = df_processado[colunas_x].values
    y_bruto = df_processado[coluna_alvo].values

    # Split temporal sequencial - sem shuffle, sem k-fold tradicional
    n = len(df_processado)
    corte_treino = int(n * 0.70)  # ex.: 1940 ate ~2000
    corte_val = int(n * 0.85)  # ex.: ate ~2015 (o resto e teste)

    padronizador = StandardScaler()
    X_escalado = padronizador.fit_transform(X_bruto)

    X_3d, y_3d = montar_janelas_3d(X_escalado, y_bruto, tamanho_janela=janela_meses)

    X_treino_3d, y_treino = X_3d[: corte_treino - janela_meses], y_3d[: corte_treino - janela_meses]
    X_val_3d, y_val = (
        X_3d[corte_treino - janela_meses : corte_val - janela_meses],
        y_3d[corte_treino - janela_meses : corte_val - janela_meses],
    )
    X_teste_3d, y_teste = X_3d[corte_val - janela_meses :], y_3d[corte_val - janela_meses :]

    # XGBoost usa so o ultimo mes de cada janela (2D)
    X_treino_2d, X_val_2d, X_teste_2d = X_treino_3d[:, -1, :], X_val_3d[:, -1, :], X_teste_3d[:, -1, :]

    modelo_xgb = XGBRegressor(
        n_estimators=500, learning_rate=0.02, max_depth=5, random_state=42
    )
    modelo_xgb.fit(X_treino_2d, y_treino, eval_set=[(X_val_2d, y_val)], verbose=False)
    previsoes_xgb_val = modelo_xgb.predict(X_val_2d)
    previsoes_xgb_teste = modelo_xgb.predict(X_teste_2d)

    dataset_treino = DatasetClimatico(X_treino_3d, y_treino)
    carregador_treino = DataLoader(dataset_treino, batch_size=32, shuffle=False)

    modelo_lstm = RedeClimaticaLSTM(quantidade_variaveis=X_treino_3d.shape[2])
    funcao_perda = nn.MSELoss()
    otimizador = torch.optim.Adam(modelo_lstm.parameters(), lr=0.001)

    for _epoca in range(30):
        modelo_lstm.train()
        for batch_x, batch_y in carregador_treino:
            otimizador.zero_grad()
            perda = funcao_perda(modelo_lstm(batch_x), batch_y)
            perda.backward()
            otimizador.step()

    modelo_lstm.eval()
    with torch.no_grad():
        previsoes_lstm_val = modelo_lstm(torch.tensor(X_val_3d, dtype=torch.float32)).numpy().squeeze()
        previsoes_lstm_teste = modelo_lstm(torch.tensor(X_teste_3d, dtype=torch.float32)).numpy().squeeze()

    # Meta-modelo (Ridge): aprende o peso ideal de cada modelo, so na validacao
    opinioes_validacao = np.column_stack((previsoes_xgb_val, previsoes_lstm_val))
    opinioes_teste = np.column_stack((previsoes_xgb_teste, previsoes_lstm_teste))

    modelo_meta = Ridge()
    modelo_meta.fit(opinioes_validacao, y_val)
    previsao_final = modelo_meta.predict(opinioes_teste)

    erro_final = np.sqrt(mean_squared_error(y_teste, previsao_final))
    print(f"RMSE final: {erro_final:.4f}")
    print(f"Pesos XGBoost / LSTM: {modelo_meta.coef_}")

    return modelo_xgb, modelo_lstm, modelo_meta, padronizador
```

## 6. Como isso se aplica ao nosso projeto (e o que precisa de ajuste)

Diferença estrutural importante: o material assume **uma série temporal por
local/região**, com observações reais chegando a cada novo mês (janela rolante). O
nosso teste real (ver `PLANO_TRABALHO.md`) **não é assim** — é uma previsão
multi-horizonte a partir de uma origem fixa (dez/2022), com `tp_ultima_obs` congelado
e `lag_meses` indo de 1 a 24. Nenhuma observação real de precipitação chega durante o
período de teste.

**O que dá para aproveitar direto:**
- A ideia de ensemble (stacking XGBoost + LSTM com meta-modelo Ridge) é uma
  arquitetura extra válida além do PCA+LSTM (P2) e ConvLSTM (P3) já planejados — pode
  virar um "Modelo C" se sobrar tempo.
- Codificação cíclica de mês (seno/cosseno).
- Prever a **anomalia** de `tp_alvo` em vez do valor bruto — dado que o treino cobre
  1940–2022, uma tendência de longo prazo é bem provável; vale testar se ajuda a
  reduzir o erro. Isso pode entrar em `src/data.py` como uma etapa opcional de
  pré-processamento do alvo (P1).
- SPI como alvo alternativo é uma ideia interessante de diferencial para o hackathon
  (jurados costumam valorizar), mas dá trabalho extra: precisa ser calculado por
  ponto de grade e reconstruído de volta para `tp_mm_day` na submissão.
- SPEI exigiria estimar evapotranspiração potencial — não temos PET direto, mas dá
  para aproximar via Hargreaves com `t2` (temperatura) que já temos; teria que estimar
  min/max mensal a partir do que está disponível.
- Índice ONI (El Niño/La Niña) não está no dataset da competição, mas é uma série
  pública da NOAA que pode ser baixada e usada como feature externa — teleconexão com
  lag de 3 a 6 meses é uma hipótese testável.

**O que precisa ser adaptado (não copiar direto):**
- Os lags do material assumem que a "última observação real" está sempre disponível
  e avança a cada mês. Isso é falso no nosso teste (congelado em dez/2022). Qualquer
  exemplo de treino construído para esse "Modelo C" precisa seguir exatamente o
  contrato já definido em `src/data.py` (origem `o` + lag `L`), senão o modelo aprende
  uma dependência de "última obs recente" que não existe na hora de gerar a submissão
  real — um erro sutil de vazamento de estrutura, não de dados.
- O material trabalha com série tabular (uma linha por mês), não com grade espacial
  301×261. Para usar XGBoost nos nossos dados, duas opções: (a) treinar um XGBoost por
  ponto de grade (caro: ~78 mil modelos), ou (b) incluir `lat`/`lon` como features e
  treinar um único XGBoost para todos os pontos — mais escalável e mais comum em
  competições espaciais; recomendado se decidirem seguir por esse caminho.
