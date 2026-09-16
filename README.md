# WORCAP 2026 — Previsão Climática de Precipitação sobre a América do Sul

[![CI](https://github.com/mazeeqe/WORCAP-2026/actions/workflows/ci.yml/badge.svg?branch=Beatriz)](https://github.com/mazeeqe/WORCAP-2026/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0-0b7a75.svg)](LICENSE)
[![Reproducible](https://img.shields.io/badge/science-reproducible-6ba539.svg)](CONTRIBUTING.md)

Repositório para o Hackathon WORCAP 2026

Código aberto sob GPL-3.0. Consulte [como contribuir](CONTRIBUTING.md), [governança científica](GOVERNANCE.md), [segurança](SECURITY.md), [citação](CITATION.cff) e [histórico](CHANGELOG.md).

> **Estado auditado em 16/09/2026:** o alinhamento temporal do treino foi corrigido
> para usar a atmosfera do mês anterior ao alvo (`M → M+1`). Os artefatos existentes
> em `experiments/pca_lstm_run1/` pertencem à execução anterior e não devem ser
> apresentados como validação atual. É necessário retreinar antes da submissão.

Membros da equipe:

- Tomáz Antonio Bortoletto Giansante
-
-
-

Código para baixar e carregar os dados da competição Kaggle
[previsao-climatica-de-precipitacao-sobre-a-america-do-sul](https://kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul).

## Estrutura do repositório

### Documentação

- `PLANO_TRABALHO.md` — esquema de trabalho do time: papéis, fases, contrato de
  dados combinado (origem + lag) e cronograma.
- `PESQUISA_LSTM_XGBOOST.md` — notas de pesquisa sobre um ensemble LSTM+XGBoost,
  anomalias climáticas e índices SPI/SPEI, com uma seção explícita sobre o que
  precisa ser adaptado antes de usar essas ideias no formato real da competição.

### Scripts na raiz

- `download_data.py` — baixa (via `kagglehub`, com cache local) e carrega os
  `.csv`/`.nc` da competição.
- `eda.py` — estatísticas por variável (min/max/média/desvio-padrão/% de dados
  faltantes) e correlação entre variáveis; salva os resumos em `eda_output/`.
- `eda.ipynb` — gráficos da análise exploratória (reaproveita `download_data.py`
  e `eda.py`).
- `resultados_pca_lstm.ipynb` — visualiza os resultados de treino do modelo
  PCA+LSTM (curvas de treino, comparação com os baselines, erro por horizonte de
  previsão, mapas espaciais de exemplo); lê os artefatos salvos em
  `experiments/pca_lstm_run1/`, sem reprocessar os dados nem re-treinar.

### Pipeline de modelagem (`src/`)

- `src/data.py` — pipeline de dados compartilhado: carregamento/alinhamento da
  grade (lat/lon/tempo), normalização (z-score no período de treino) e
  `build_examples` (monta os exemplos seguindo o contrato origem + lag).
- `src/baseline.py` — baselines de comparação: persistência (repete a última
  observação real) e climatologia (média histórica por mês do calendário).
- `src/evaluate.py` — métricas de avaliação (RMSE/MAE, com quebra por horizonte
  de previsão).
- `src/submit.py` — copia os IDs oficiais de `sample_submission.csv` e preenche a
  previsão final sem reconstruir identificadores (arquivos em `submissions/`).
- `src/train_pca_lstm.py` — script principal do **Modelo A**: ajusta PCA por
  variável, treina o LSTM hindcast/forecast com validação interna (early
  stopping), compara com os baselines, retreina com todo o histórico rotulado
  e gera a submissão real.
- `src/models/pca_lstm.py` — arquitetura do Modelo A: `SpatialPCA` (redução
  espacial da grade via PCA) e `HindcastForecastLSTM` (encoder LSTM + decoder
  condicionado no mês alvo, na última observação real e no lag).
- `src/models/convlstm.py` — arquitetura funcional do **Modelo B** com células
  ConvLSTM empilhadas, condicionamento nas variáveis de `M` e no horizonte,
  preservação espacial e saída de precipitação não negativa. O treinamento
  completo continua condicionado ao dataset oficial.

### Configuração e resultados (gerados, não totalmente versionados)

- `configs/pca_lstm.yaml`, `configs/convlstm.yaml` — hiperparâmetros de cada
  modelo.
- `experiments/` — artefatos de cada rodada de treino (histórico de épocas,
  métricas, grades de amostra ficam versionados; checkpoints de modelo e
  objetos PCA não — ver `.gitignore`).
- `submissions/` — arquivos de submissão gerados por `src/submit.py` (não
  versionado; cada pessoa gera o próprio ao rodar o treino).
- `eda_output/` — resumos gerados por `eda.py` (não versionado, ver seção 2 de "Uso").

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
python3 download_data.py
```

O script:
- Baixa (ou reaproveita o cache local em `~/.cache/kagglehub/`) os arquivos da competição.
- Carrega cada `.csv` em um `pandas.DataFrame`.
- Carrega cada `.nc` (NetCDF, dados climáticos em grade lat/lon) em um `xarray.Dataset`.
- Imprime um resumo (shape/variáveis) de cada arquivo carregado.

Os dados não ficam no repositório (são grandes e cada pessoa baixa a própria cópia
com o token individual do Kaggle).

#### Executar dentro de um Kaggle Notebook

Ao anexar a competição em **Add Input**, o pipeline detecta automaticamente os
13 arquivos em `/kaggle/input`, sem exigir token dentro do notebook:

```bash
!git clone --branch Beatriz https://github.com/mazeeqe/WORCAP-2026.git
%cd WORCAP-2026
!pip install -q -r requirements.txt
!python kaggle_notebook.py
```

Use uma sessão com GPU. Ao finalizar, `/kaggle/working` terá
`submission_pca_lstm.csv` e `submission_manifest.json`, contendo contagem de
linhas, hash SHA-256, contrato temporal e proveniência. Baixe os dois arquivos
ou salve uma versão do notebook para preservar a saída.

#### Publicação opcional no Google Cloud

BigQuery e Cloud Storage podem preservar a proveniência e os artefatos, mas não
são necessários para treinar. Configure uma conta de serviço pelo mecanismo de
Secrets do Kaggle e nunca coloque o JSON de credenciais no notebook ou GitHub.
Depois de selecionar **Link account** e concluir a autorização, o executor usa
`UserSecretsClient.get_gcloud_credential()` e
`set_tensorflow_credential()` para criar as credenciais de aplicação apenas na
sessão. O valor da credencial não é impresso nem salvo como output.

```bash
!pip install -q -r requirements-gcp.txt
!python kaggle_notebook.py \
  --gcp-project SEU_PROJETO \
  --gcs-bucket SEU_BUCKET \
  --bigquery-table SEU_PROJETO.dataset.submission_runs
```

Cloud AutoML, Translation, Natural Language, Video Intelligence e Vision não
fazem parte do fluxo: não melhoram diretamente o contrato de previsão M→M+1 e
introduziriam custo e credenciais sem evidência de benefício preditivo.

### 2. Análise exploratória (estatísticas)

```bash
python3 eda.py
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
jupyter lab eda.ipynb
```

O notebook `eda.ipynb` reaproveita as funções de `download_data.py` e `eda.py` (lendo o
cache de `eda_output/` quando disponível) e plota:
- barras de % de dados faltantes por variável;
- heatmap de correlação entre variáveis;
- séries temporais anuais (precipitação, temperatura, cobertura de nuvens);
- histograma da distribuição de `tp_alvo`;
- mapas espaciais (média no tempo) de `tp_alvo` e `t2` sobre a América do Sul.

### 4. Verificar o contrato temporal e a submissão

```bash
pytest -q
```

Os testes garantem que as variáveis atmosféricas vêm de `M`, o alvo vem de
`M+1` e os IDs oficiais são copiados sem reconstrução.
