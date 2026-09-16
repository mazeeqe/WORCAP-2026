import pytest

torch = pytest.importorskip("torch")

from src.models.convlstm import ConvLSTMCell, ConvLSTMForecaster


def test_cell_preserves_spatial_shape() -> None:
    cell = ConvLSTMCell(in_channels=3, hidden_channels=5)
    x = torch.randn(2, 3, 8, 7)
    hidden, state = cell.initial_state(x)

    next_hidden, next_state = cell(x, hidden, state)

    assert next_hidden.shape == (2, 5, 8, 7)
    assert next_state.shape == next_hidden.shape


def test_forecaster_returns_non_negative_grid() -> None:
    model = ConvLSTMForecaster(
        n_hindcast_features=4,
        n_features_atm=3,
        hidden_channels=6,
        num_layers=2,
    )
    hindcast = torch.randn(2, 4, 4, 8, 7)
    atmosphere_m = torch.randn(2, 3, 8, 7)

    prediction = model(hindcast, atmosphere_m, torch.tensor([1 / 24, 1.0]))

    assert prediction.shape == (2, 8, 7)
    assert torch.all(prediction >= 0)
