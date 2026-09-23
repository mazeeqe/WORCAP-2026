# Plano de Trabalho — WORCAP 2026 (Previsão de Precipitação sobre a América do Sul)

Esquema para 3 pessoas trabalharem em paralelo, da EDA até a submissão no Kaggle,
testando dois modelos: **PCA/EOF + LSTM** e **ConvLSTM**.

## Contexto técnico (resumo do que já sabemos)

- Grade: 301 lat × 261 lon, mensal, treino 1940–2022 (996 meses), teste 2023–2024 (24 meses).
- `tp_alvo[t] = tp[t+1]` no treino (alvo é sempre o mês seguinte).
- No teste, `tp_ultima_obs` é **constante** (= `tp` de dez/2022) em todos os 24 meses —
  ou seja, não é previsão recursiva mês a mês, é **previsão multi-horizonte a partir de
  uma única origem** (`lag_meses` = 1 a 24). As demais variáveis atmosféricas são dadas
  para cada mês alvo.
- Isso define o "contrato" que os dois modelos precisam respeitar (ver Fase 0).

## Papéis

| Pessoa | Responsabilidade principal |
|---|---|
| **P1 — Dados & Infra** | EDA, pipeline de dados compartilhado, baselines, avaliação, submissão |
| **P2 — Modelo A** | PCA/EOF + LSTM (hindcast/forecast) |
| **P3 — Modelo B** | ConvLSTM |

Divisão pensada para minimizar dependência sequencial: P2 e P3 trabalham em paralelo assim
que o pipeline da P1 estiver de pé (Fase 1), e todos convergem na Fase 4.

## Fase 0 — Alinhamento (todos, ~meio dia)

Antes de codar em paralelo, combinar por escrito (evita retrabalho e conflito de merge):

1. **Split temporal fixo**: ex. treino 1940–2015, validação 2016–2022, teste 2023–2024.
   Nunca split aleatório (vazaria informação futura).
2. **Formato de exemplo de treino** que os dois modelos vão consumir, simulando o cenário
   do teste real: para uma origem `o` e um lag `L` (1–24), entrada = variáveis atmosféricas
   do mês `o+L` + `tp` do mês `o` (congelado) + `L`; alvo = `tp_alvo` do mês `o+L`.
3. **Métrica única** de comparação (RMSE e MAE por ponto de grade, calculado após
   reconstrução espacial).
4. **Formato do `submission.csv`** (igual ao `sample_submission.csv`: `id`, `tp_mm_day`).
5. Estrutura de branches no git (ver seção Git abaixo).

## Fase 1 — EDA + pipeline de dados compartilhado (P1)

- Expandir `eda.py`/`eda.ipynb` (já existentes) com foco em: sazonalidade, tendência de
  longo prazo, missingness, correlação entre variáveis e o alvo.
- Construir `src/data.py`:
  - Carregamento dos `.nc` via `xarray`, alinhamento em tensor `(tempo, lat, lon, variável)`.
  - Normalização (z-score por variável, estatísticas calculadas só no treino).
  - Função que gera pares `(X, y)` no formato combinado na Fase 0, para qualquer lag.
- Implementar baselines em `src/baseline.py`:
  - **Persistência**: repetir `tp_ultima_obs`.
  - **Climatologia**: média histórica do mês do calendário correspondente.
  - Essencial para saber se os modelos de fato agregam valor.
- Assim que isso estiver rodando, disponibilizar no branch principal para P2 e P3 puxarem.

## Fase 2 — Desenvolvimento dos modelos (P2 e P3, em paralelo)

**P2 — PCA/EOF + LSTM** (`src/models/pca_lstm/`)
- Ajustar PCA (ou `IncrementalPCA`) por variável, só com dados de treino; reter componentes
  suficientes para ~90–95% da variância.
- LSTM hindcast (janela histórica) → vetor de contexto → decoder condicionado em
  `[variáveis do mês alvo, L]` → coeficientes PCA previstos de `tp_alvo`.
- Reconstrução espacial via transformação inversa do PCA.

**P3 — ConvLSTM** (`src/models/convlstm.py`)
- Mantém a grade espacial (sem achatar); avaliar se é viável treinar na resolução cheia
  (301×261) ou se precisa reduzir (downsample, patches) dado o hardware disponível.
- Mesma lógica de condicionamento em `L` e nas variáveis do mês alvo.

Ambos devem consumir os dados via `src/data.py` e reportar métricas via `src/evaluate.py`
(mesma função para os dois, garantindo comparação justa).

## Fase 3 — Ajuste de hiperparâmetros (P2 e P3, cada um no seu modelo)

- **PCA+LSTM**: nº de componentes PCA, tamanho do histórico (hindcast), hidden size,
  nº de camadas, dropout, learning rate.
- **ConvLSTM**: nº de filtros, tamanho de kernel, nº de camadas, learning rate, batch size.
- Ferramenta leve para não perder tempo de hackathon: grid/random search manual com log em
  CSV (`models/`), ou Optuna se alguém já tiver familiaridade.
- Critério de seleção: erro na validação temporal (nunca no teste).

## Fase 4 — Comparação, ensemble e submissão (todos)

- Tabela final comparando: persistência, climatologia, PCA+LSTM, ConvLSTM (e um ensemble
  simples — média das previsões — se dois modelos forem competitivos).
- `src/submit.py` (mantido por P1) gera o `submission.csv` no formato exigido, a partir da
  saída de qualquer modelo — interface única evita duplicar lógica de formatação.
- Submeter no Kaggle, conferir o leaderboard, iterar se sobrar tempo.

## Estrutura de pastas sugerida

```
worcap_2026/
├── download_data.py        (já existe)
├── eda.py / eda.ipynb       (já existe — P1)
├── src/
│   ├── data.py              (pipeline compartilhado — P1)
│   ├── baseline.py          (persistência / climatologia — P1)
│   ├── evaluate.py          (métrica comum — P1)
│   ├── submit.py            (gera submission.csv — P1)
│   └── models/
│       ├── pca_lstm/        (P2)
│       │   ├── model.py     (SpatialPCA/SpatialPLS/HindcastForecastLSTM)
│       │   └── train.py     (script de treino, python3 -m src.models.pca_lstm.train)
│       └── convlstm.py      (P3)
├── configs/
│   ├── pca_lstm.yaml
│   └── convlstm.yaml
└── models/                  (logs/resultados de cada rodada de tuning)
```

## Git

- `main` como branch estável.
- Branches de trabalho: `feature/pipeline-dados` (P1), `feature/pca-lstm` (P2),
  `feature/convlstm` (P3).
- PR para `main` só depois de rodar localmente contra o mesmo split de validação da Fase 0.
- Interface combinada na Fase 0 evita que dois mexam no mesmo arquivo ao mesmo tempo.

## Cronograma sugerido (ajustar à duração real do hackathon)

| Quando | O quê |
|---|---|
| Dia 1 | Fase 0 + Fase 1 |
| Dia 2 | Fase 2 (primeira versão treinável dos dois modelos) |
| Dia 3 | Fase 3 (tuning) + início da Fase 4 |
| Dia 4 (se houver) | Fase 4 completa + submissão final |
