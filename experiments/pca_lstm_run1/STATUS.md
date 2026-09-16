# Estado da execução `pca_lstm_run1`

Os arquivos deste diretório registram uma execução histórica e foram preservados para rastreabilidade.

Em 16/09/2026, a auditoria identificou que o treino usava variáveis atmosféricas do próprio mês-alvo. O contrato do desafio exige usar o estado do mês anterior (`M`) para prever `M+1`. O alinhamento foi corrigido em `src/data.py`.

Consequências:

- `metrics.json`, `history.csv` e `sample_grids.npz` não representam a versão corrigida;
- os números antigos não devem ser usados em README, painel, apresentação ou comparação;
- `model_final.pt`, `pca_and_stats.joblib` e a submissão precisam ser regenerados;
- uma nova rodada só é considerada válida após preservar os IDs do `sample_submission.csv` e confirmar 1.885.464 previsões finitas e não negativas.
