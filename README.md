# WORCAP 2026 — Previsão Climática de Precipitação sobre a América do Sul

Repositório para o Hackathon WORCAP 2026

Membros da equipe:

- Tomáz Antonio Bortoletto Giansante
- Beatriz Karoline Cordeiro da Silva
- Daiane Fonseca
-

Código para baixar e carregar os dados da competição Kaggle
[previsao-climatica-de-precipitacao-sobre-a-america-do-sul](https://kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul).

## Estrutura do repositório

### Documentação (`docs/`)

- `docs/PLANO_TRABALHO.md` — esquema de trabalho do time: papéis, fases, contrato de
  dados combinado (origem + lag) e cronograma.
- `docs/PESQUISA_LSTM_XGBOOST.md` — notas de pesquisa sobre um ensemble LSTM+XGBoost,
  anomalias climáticas e índices SPI/SPEI, com uma seção explícita sobre o que
  precisa ser adaptado antes de usar essas ideias no formato real da competição.

### Scripts (`scripts/`)

Rodados como módulo a partir da raiz do repositório (ex.: `python3 -m scripts.download_data`),
não como arquivo solto — é o que deixa `from src... import ...` resolver corretamente.

- `scripts/download_data.py` — baixa (via `kagglehub`, com cache local) e carrega os
  `.csv`/`.nc` da competição.
- `scripts/eda.py` — estatísticas por variável (min/max/média/desvio-padrão/% de dados
  faltantes) e correlação entre variáveis; salva os resumos em `eda_output/`.
- `scripts/run_all_models.py` — treina em sequência os modelos/variações disponíveis
  (registro em `MODEL_RUNNERS`; hoje só o Modelo A, com `--variations pca
  pls_concurrent pls_lagged`, padrão: todas), cada um num processo separado; ao
  final salva/imprime um resumo comparando RMSE/MAE (`models/run_all_summary.json`).
  Se uma variação falhar, as outras continuam (a menos que `--stop-on-error` seja
  passado).
- `scripts/run_hparam_sweep.py` — sweep em grade (produto cartesiano) de hiperparâmetros
  (`--lrs`/`--hidden-sizes`/`--dropouts`) para um método de redução do Modelo A
  (`--method`, padrão: o de melhor RMSE em `models/run_all_summary.json`); cada
  combinação salva seus artefatos em `models/hparam_sweep/`.
- `scripts/compare_reducao_dimensional.py`, `scripts/cv_ensemble.py`,
  `scripts/final_ensemble.py`, `scripts/final_blend.py`, `scripts/postprocess_enso.py`,
  `scripts/enso_correction_cv.py` — triagem de métodos de redução, CV walk-forward de
  5 dobras (config/época/seeds/encolhimento em direção à climatologia), geração da
  submissão final e correção de viés ENSO pós-hoc; ver o docstring de cada um (`--help`)
  para o uso detalhado.

### Notebooks (`notebooks/`)

- `notebooks/eda.ipynb` — gráficos da análise exploratória (reaproveita
  `scripts/download_data.py` e `scripts/eda.py`).
- `notebooks/resultados_pca_lstm.ipynb` — visualiza os resultados de treino do modelo
  PCA+LSTM (curvas de treino, comparação com os baselines, erro por horizonte de
  previsão, mapas espaciais de exemplo); lê os artefatos salvos em
  `models/pca_lstm_run1/`, sem reprocessar os dados nem re-treinar.

### Pipeline de modelagem (`src/`)

- `src/data.py` — pipeline de dados compartilhado: carregamento/alinhamento da
  grade (lat/lon/tempo), normalização (z-score no período de treino) e
  `build_examples` (monta os exemplos seguindo o contrato origem + lag).
- `src/baseline.py` — baselines de comparação: persistência (repete a última
  observação real) e climatologia (média histórica por mês do calendário).
- `src/evaluate.py` — métricas de avaliação (RMSE/MAE, com quebra por horizonte
  de previsão).
- `src/submit.py` — formata a previsão final no formato exigido pelo Kaggle
  (arquivos em `submissions/`).
- `src/models/pca_lstm/train.py` (rode com `python3 -m src.models.pca_lstm.train`)
  — script principal do **Modelo A**: reduz cada variável espacialmente (PCA ou
  PLS — ver `--reduction`), treina o LSTM hindcast/forecast com validação
  interna (early stopping), compara com os baselines, retreina com todo o
  histórico rotulado e gera a submissão real. Aceita `--reduction
  {pca,pls_concurrent,pls_lagged}` para comparar PCA (não-supervisionado)
  contra PLS (supervisionado, com `tp` como alvo — no mesmo mês ou defasado,
  via `--pls-lag-shift`), além de `--lr`/`--hidden-size`/`--dropout`/`--run-dir`
  para variar hiperparâmetros; cada combinação salva seus artefatos numa pasta
  separada em `models/`. Faz backup do progresso a cada época (checkpoint do
  modelo + `history.csv`, tanto na seleção de épocas quanto no retreino final)
  e loga tudo em `train.log` dentro da pasta do run — se o processo cair no
  meio, o treino já feito não se perde (não há retomada automática, é preciso
  rodar de novo).
- `src/models/pca_lstm/model.py` — arquitetura do Modelo A: `SpatialPCA`
  (redução espacial via PCA, mantendo o menor nº de componentes que atinja 90%
  de variância explicada), `SpatialPLS` (idem via PLS, supervisionada pelos
  componentes PCA de `tp` — busca binária pelo menor nº de componentes já que o
  sklearn não expõe a variância explicada por k componentes do PLS num só fit)
  e `HindcastForecastLSTM` (encoder LSTM + decoder condicionado no mês alvo, na
  última observação real e no lag).
- `src/models/convlstm.py` — esqueleto do **Modelo B** (ConvLSTM), ainda por
  implementar (há uma implementação funcional na branch `Beatriz`, ainda sem
  script de treino).

### Configuração e resultados (gerados, não totalmente versionados)

- `configs/pca_lstm.yaml`, `configs/convlstm.yaml` — hiperparâmetros de cada
  modelo.
- `models/` — artefatos de cada rodada de treino (histórico de épocas,
  métricas, grades de amostra ficam versionados; checkpoints de modelo e
  objetos PCA não — ver `.gitignore`).
- `submissions/` — arquivos de submissão gerados por `src/submit.py` (não
  versionado; cada pessoa gera o próprio ao rodar o treino).
- `eda_output/` — resumos gerados por `scripts/eda.py` (não versionado, ver seção 2 de "Uso").

## Setup (rodar uma vez por máquina)

1. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

2. Aceite as regras da competição (logado na sua conta Kaggle):
   https://kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul/rules

3. Configure suas credenciais da API do Kaggle. Existem duas formas — use a que
   corresponder ao tipo de token que você gerou em kaggle.com → Settings → API:

   - **Token novo (formato `KGAT_...`)**: salve o token em `~/.kaggle/access_token`
     (arquivo de uma linha só, sem aspas nem JSON):

     ```bash
     mkdir -p ~/.kaggle
     echo -n "SEU_TOKEN_KGAT_AQUI" > ~/.kaggle/access_token
     chmod 600 ~/.kaggle/access_token
     ```

   - **Token clássico (`kaggle.json` com username + key)**: salve o arquivo baixado
     do Kaggle em `~/.kaggle/kaggle.json` e restrinja a permissão:

     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
     chmod 600 ~/.kaggle/kaggle.json
     ```

   Nunca commite esses arquivos — eles já estão no `.gitignore`.

## Uso

### 1. Baixar e carregar os dados

```bash
python3 -m scripts.download_data
```

O script:
- Baixa (ou reaproveita o cache local em `~/.cache/kagglehub/`) os arquivos da competição.
- Carrega cada `.csv` em um `pandas.DataFrame`.
- Carrega cada `.nc` (NetCDF, dados climáticos em grade lat/lon) em um `xarray.Dataset`.
- Imprime um resumo (shape/variáveis) de cada arquivo carregado.

Os dados não ficam no repositório (são grandes e cada pessoa baixa a própria cópia
com o token individual do Kaggle).

### 2. Análise exploratória (estatísticas)

```bash
python3 -m scripts.eda
```

Calcula, para cada variável: min, max, média, desvio-padrão e % de dados faltantes,
além da correlação entre variáveis (usando a média espacial de cada uma por mês).
Salva os resumos em `eda_output/` (CSV, não versionado — cada pessoa regenera o próprio).

> **Atenção:** média/desvio são calculados forçando `dtype="float64"`. O `xarray`
> (via `bottleneck`) tem um bug de precisão numérica ao reduzir arrays `float32`
> grandes (dezenas de milhões de pontos) — sem o float64 explícito, a média de uma
> variável cujo range real era 252–308 aparecia como 109. Se for calcular estatísticas
> manualmente sobre os `.nc`, sempre passe `dtype="float64"` em `.mean()`/`.std()`.

### 3. Gráficos

```bash
jupyter lab notebooks/eda.ipynb
```

O notebook `notebooks/eda.ipynb` reaproveita as funções de `scripts/download_data.py` e
`scripts/eda.py` (lendo o cache de `eda_output/` quando disponível) e plota:
- barras de % de dados faltantes por variável;
- heatmap de correlação entre variáveis;
- séries temporais anuais (precipitação, temperatura, cobertura de nuvens);
- histograma da distribuição de `tp_alvo`;
- mapas espaciais (média no tempo) de `tp_alvo` e `t2` sobre a América do Sul.
