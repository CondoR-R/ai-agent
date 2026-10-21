import pytest
import torch
from torch import nn

from ai_agent.labs.lr0.model import MLP, build_model


@pytest.fixture(autouse=True)
def _restore_torch_rng():
    """Сохраняем/восстанавливаем глобальное состояние RNG — тесты не должны
    влиять друг на друга через torch.manual_seed."""
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


# ---------- форма выхода ----------


def test_forward_output_shape():
    """На батче (8, 64) выход имеет форму (8, n_classes=10)."""
    model = MLP(n_layers=2, hidden=32, dropout=0.1)
    x = torch.randn(8, 64)
    out = model(x)
    assert out.shape == (8, 10)


# ---------- структура ----------


def test_layer_counts():
    """nn.Linear = n_layers + 1, nn.Dropout = n_layers."""
    n_layers = 3
    model = MLP(n_layers=n_layers, hidden=16, dropout=0.2)

    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]

    assert len(linear_layers) == n_layers + 1
    assert len(dropout_layers) == n_layers


# ---------- некорректные аргументы ----------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_layers": 0, "hidden": 8, "dropout": 0.1},
        {"n_layers": 1, "hidden": 0, "dropout": 0.1},
        {"n_layers": 1, "hidden": 8, "dropout": 1.0},
        {"n_layers": 1, "hidden": 8, "dropout": 0.1, "n_classes": 1},
        {"n_layers": 1, "hidden": 8, "dropout": 0.1, "in_features": 0},
    ],
    ids=[
        "n_layers=0",
        "hidden=0",
        "dropout=1.0",
        "n_classes=1",
        "in_features=0",
    ],
)
def test_invalid_args_raise(kwargs):
    with pytest.raises(ValueError):
        MLP(**kwargs)


# ---------- режимы ----------


def test_eval_mode_is_deterministic():
    """В eval() dropout выключен: два прохода с одним входом дают одинаковый выход."""
    model = MLP(n_layers=2, hidden=16, dropout=0.5)
    model.eval()

    x = torch.randn(4, 64)
    with torch.no_grad():
        a = model(x)
        b = model(x)

    assert torch.allclose(a, b)


def test_train_mode_with_dropout_is_random():
    """В train() при dropout > 0 два прохода обычно дают разные выходы."""
    torch.manual_seed(0)
    model = MLP(n_layers=2, hidden=16, dropout=0.5)
    model.train()

    x = torch.randn(4, 64)
    with torch.no_grad():
        a = model(x)
        b = model(x)

    # маски dropout на двух проходах берутся из последовательных состояний RNG
    assert not torch.allclose(a, b)


# ---------- повторяемость ----------


def test_same_seed_same_weights():
    """После set_seed(0) две модели имеют одинаковые веса."""
    torch.manual_seed(0)
    m1 = MLP(n_layers=2, hidden=16, dropout=0.1)
    torch.manual_seed(0)
    m2 = MLP(n_layers=2, hidden=16, dropout=0.1)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_different_seed_different_weights():
    """После set_seed(1) веса отличаются от весов после set_seed(0)."""
    torch.manual_seed(0)
    m1 = MLP(n_layers=2, hidden=16, dropout=0.1)
    torch.manual_seed(1)
    m2 = MLP(n_layers=2, hidden=16, dropout=0.1)

    assert any(
        not torch.equal(p1, p2)
        for p1, p2 in zip(m1.parameters(), m2.parameters())
    )
