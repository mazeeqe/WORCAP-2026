"""Modelo A: PCA/EOF + LSTM hindcast/forecast (dono: P2).

Ideia: reduzir a grade (301x261) para N componentes principais por variavel,
treinar o LSTM sobre as series temporais dos coeficientes, e reconstruir o
campo espacial com a transformacao inversa do PCA. Ver Fase 2 do
PLANO_TRABALHO.md e o contrato de exemplos em src/data.py.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from torch import nn


class SpatialPCA:
    """Ajusta um PCA por variavel sobre a dimensao espacial (lat*lon), usando so o treino.

    O numero de componentes nao e fixo: e escolhido automaticamente como o menor
    que atinge `variance_threshold` da variancia de X (criterio de contribuicao
    minima), ate um teto de `max_components` (guarda de custo/tempo - se nem
    `max_components` bastar, usa esse teto e avisa)."""

    def __init__(self, variance_threshold: float = 0.90, max_components: int = 200, random_state: int = 42):
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.random_state = random_state
        self._pca: PCA | None = None
        self.n_components_: int | None = None
        self.spatial_shape: tuple[int, int] | None = None

    def fit(self, data: np.ndarray) -> "SpatialPCA":
        """`data` com shape (tempo, lat, lon)."""
        self.spatial_shape = data.shape[1:]
        flat = data.reshape(data.shape[0], -1)

        cap = min(self.max_components, flat.shape[0] - 1, flat.shape[1])
        probe = PCA(n_components=cap, svd_solver="randomized", random_state=self.random_state)
        probe.fit(flat)
        cumulative = np.cumsum(probe.explained_variance_ratio_)

        if cumulative[-1] < self.variance_threshold:
            print(
                f"    aviso: nao atingiu {self.variance_threshold:.0%} de variancia mesmo com "
                f"max_components={cap} (alcancado: {cumulative[-1]:.1%}); usando {cap} componentes"
            )
            self.n_components_ = cap
            self._pca = probe
        else:
            self.n_components_ = int(np.searchsorted(cumulative, self.variance_threshold)) + 1
            self._pca = PCA(n_components=self.n_components_, svd_solver="randomized", random_state=self.random_state)
            self._pca.fit(flat)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """`data` com shape (tempo, lat, lon) -> (tempo, n_components)."""
        flat = data.reshape(data.shape[0], -1)
        return self._pca.transform(flat)

    def inverse_transform(self, coeffs: np.ndarray) -> np.ndarray:
        """`coeffs` com shape (tempo, n_components) -> (tempo, lat, lon)."""
        flat = self._pca.inverse_transform(coeffs)
        return flat.reshape(coeffs.shape[0], *self.spatial_shape)

    def explained_variance_ratio(self) -> float:
        return float(self._pca.explained_variance_ratio_.sum())


class SpatialPLS:
    """Ajusta um PLS (Partial Least Squares) por variavel sobre a dimensao espacial
    (lat*lon), usando so o treino.

    Diferenca em relacao ao SpatialPCA: o PCA e nao-supervisionado (maximiza so a
    variancia da propria variavel X). O PLS e supervisionado - precisa de um alvo
    Y no fit e escolhe os componentes que maximizam a covariancia entre X e Y. Aqui
    Y e sempre a serie de componentes PCA de `tp` (concorrente ou defasada no tempo),
    entao os componentes capturam a parte de cada variavel atmosferica mais ligada a
    precipitacao, em vez de so a parte de maior variancia espacial.

    O sklearn nao expõe a variancia de X explicada por k componentes num unico fit
    (como o PCA faz via `explained_variance_ratio_`), entao o numero de componentes
    e escolhido testando varios valores de k ("permutacoes"): busca binaria pelo
    menor k cuja fracao de variancia de X capturada pelos scores do PLS atinja
    `variance_threshold`, ate um teto de `max_components`. A busca binaria e valida
    porque essa fracao e nao-decrescente em k (cada componente extra do PLS e
    calculado sobre o residuo de X apos deflacao dos anteriores, entao so acrescenta
    variancia capturada).
    """

    def __init__(self, variance_threshold: float = 0.90, max_components: int = 30, max_iter: int = 100):
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        # sklearn usa NIPALS por padrao (max_iter=500): cada componente itera entre X e Y ate
        # convergir, e com Y multi-coluna (aqui, os componentes de tp) isso pode ficar bem lento -
        # reduzido pra nao deixar um fit individual rodar por muito tempo (ver conversa: com Y de
        # 85 colunas e max_components=100, um unico fit chegou a levar horas).
        self.max_iter = max_iter
        self._pls: PLSRegression | None = None
        self.n_components_: int | None = None
        self.spatial_shape: tuple[int, int] | None = None
        self._x_total_var: float | None = None

    def _fit_at(self, flat: np.ndarray, target: np.ndarray, k: int) -> tuple[PLSRegression, float]:
        pls = PLSRegression(n_components=k, scale=False, max_iter=self.max_iter)
        pls.fit(flat, target)
        ratio = float(np.var(pls.x_scores_, axis=0).sum()) / self._x_total_var
        return pls, ratio

    def fit(self, data: np.ndarray, target: np.ndarray) -> "SpatialPLS":
        """`data` com shape (tempo, lat, lon); `target` com shape (tempo, n_componentes_alvo),
        ja alinhados no tempo (ver alinhamento do lag em src/train_pca_lstm.py)."""
        self.spatial_shape = data.shape[1:]
        flat = data.reshape(data.shape[0], -1)
        self._x_total_var = float(np.var(flat, axis=0).sum())

        cap = min(self.max_components, flat.shape[0] - 1, flat.shape[1])
        best_pls, best_ratio = self._fit_at(flat, target, cap)
        best_k = cap

        if best_ratio < self.variance_threshold:
            print(
                f"    aviso: PLS nao atingiu {self.variance_threshold:.0%} de variancia de X mesmo "
                f"com max_components={cap} (alcancado: {best_ratio:.1%}); usando {cap} componentes"
            )
        else:
            lo, hi = 1, cap
            while lo < hi:
                mid = (lo + hi) // 2
                pls, ratio = self._fit_at(flat, target, mid)
                if ratio >= self.variance_threshold:
                    hi = mid
                    best_pls, best_ratio, best_k = pls, ratio, mid
                else:
                    lo = mid + 1

        self._pls = best_pls
        self.n_components_ = best_k
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """`data` com shape (tempo, lat, lon) -> (tempo, n_components)."""
        flat = data.reshape(data.shape[0], -1)
        return self._pls.transform(flat)

    def inverse_transform(self, coeffs: np.ndarray) -> np.ndarray:
        """`coeffs` com shape (tempo, n_components) -> (tempo, lat, lon)."""
        flat = self._pls.inverse_transform(coeffs)
        return flat.reshape(coeffs.shape[0], *self.spatial_shape)

    def explained_variance_ratio(self) -> float:
        """Fracao da variancia de X (nao de Y) capturada pelos scores do PLS -
        calculada so para ficar comparavel com SpatialPCA.explained_variance_ratio()
        (o PLS nao otimiza para essa quantidade, entao ela tende a ser menor que a
        do PCA com o mesmo numero de componentes)."""
        scores_var = float(np.var(self._pls.x_scores_, axis=0).sum())
        return scores_var / self._x_total_var


class HindcastForecastLSTM(nn.Module):
    """Encoder LSTM sobre a janela historica + decoder condicionado no ultimo mes com
    dado atmosferico disponivel antes do alvo (mes o+L-1, ver src/data.py), no tp
    congelado (ultima observacao real) e no lag (meses a frente). `target_month_features`
    mantem esse nome por compatibilidade, mas nao contem dado do proprio mes-alvo."""

    def __init__(
        self,
        n_features_hindcast: int,
        n_features_atm: int,
        n_components_tp: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features_hindcast,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        decoder_input_dim = hidden_size + n_features_atm + n_components_tp + 1  # +1 = lag
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_components_tp),
        )

    def forward(
        self,
        hindcast_seq: torch.Tensor,
        target_month_features: torch.Tensor,
        tp_frozen: torch.Tensor,
        lag: torch.Tensor,
    ) -> torch.Tensor:
        """Retorna os coeficientes PCA previstos de tp para o mes alvo (o+L)."""
        _, (hn, _) = self.encoder(hindcast_seq)
        context = hn[-1]  # estado oculto da ultima camada, (batch, hidden_size)

        if lag.dim() == 1:
            lag = lag.unsqueeze(-1)

        decoder_input = torch.cat([context, target_month_features, tp_frozen, lag], dim=-1)
        return self.decoder(decoder_input)
