# Como contribuir

Este repositório científico usa a branch `Beatriz` como branch de integração. Antes de propor alterações, abra uma issue curta descrevendo objetivo e evidência esperada.

## Preparação

```bash
python -m venv .venv
pip install -r requirements.txt
pytest
```

## Regras científicas

- preserve o contrato temporal `M → M+1`;
- compare modelos contra climatologia e persistência usando o mesmo recorte;
- ajuste normalizadores, PCA/EOF e hiperparâmetros apenas no treino;
- documente semente, período, variáveis, hardware e métricas;
- nunca versione `.nc`, credenciais, submissões pesadas ou dados restritos;
- não apresente artefato antigo, parcial ou demonstrativo como resultado validado;
- preserve a ordem e os IDs de `sample_submission.csv`.

Use commits convencionais, como `feat(model):`, `fix(data):`, `test:` e `docs:`. Pull requests precisam explicar impacto científico, testes e reprodutibilidade.
