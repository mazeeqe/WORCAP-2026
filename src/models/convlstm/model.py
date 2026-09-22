"""Modelo B: ConvLSTM (dono: P3).

Mantem a estrutura espacial da grade (sem achatar via PCA/PLS como o Modelo A) --
cada celula processa a grade inteira (ou uma versao reduzida, ver
`--spatial-downsample` em train.py) com convolucoes em vez de operar sobre
coeficientes. A saida usa Softplus (nao ReLU/clip) para garantir precipitacao
nao-negativa continuamente diferenciavel, treinado direto em mm/dia (nao em
z-score, que pode ser negativo mesmo com tp>=0 -- ver train.py).
"""

from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size deve ser ímpar para preservar a grade")
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.gates(torch.cat([x, h_prev], dim=1))
        input_gate, forget_gate, output_gate, candidate = gates.chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        cell = forget_gate * c_prev + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell

    def initial_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
        state = x.new_zeros(shape)
        return state, state.clone()


class ConvLSTMForecaster(nn.Module):
    """Empilha ConvLSTMCells; recebe a janela historica + variaveis do mes alvo + lag."""

    def __init__(
        self,
        n_hindcast_features: int,
        n_features_atm: int,
        hidden_channels: int = 32,
        num_layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers deve ser pelo menos 1")
        channels = [n_hindcast_features, *([hidden_channels] * (num_layers - 1))]
        self.cells = nn.ModuleList(
            ConvLSTMCell(in_channels, hidden_channels, kernel_size)
            for in_channels in channels
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels + n_features_atm + 1, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        hindcast_seq: torch.Tensor,
        target_month_features: torch.Tensor,
        lag: torch.Tensor,
    ) -> torch.Tensor:
        """Retorna precipitação não negativa (mm/dia) com shape `(batch, lat, lon)`."""
        if hindcast_seq.ndim != 5 or target_month_features.ndim != 4:
            raise ValueError("esperado hindcast (B,T,C,H,W) e atmosfera (B,C,H,W)")
        if hindcast_seq.shape[0] != target_month_features.shape[0]:
            raise ValueError("batch divergente entre hindcast e atmosfera")
        output = hindcast_seq
        for cell in self.cells:
            hidden, state = cell.initial_state(output[:, 0])
            sequence = []
            for step in range(output.shape[1]):
                hidden, state = cell(output[:, step], hidden, state)
                sequence.append(hidden)
            output = torch.stack(sequence, dim=1)

        context = output[:, -1]
        if lag.ndim == 1:
            lag = lag[:, None, None, None]
        if lag.ndim == 2:
            lag = lag[:, :, None, None]
        lag_grid = lag.expand(-1, 1, context.shape[-2], context.shape[-1])
        decoder_input = torch.cat([context, target_month_features, lag_grid], dim=1)
        return self.decoder(decoder_input).squeeze(1)
