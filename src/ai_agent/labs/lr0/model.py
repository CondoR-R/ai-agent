"""Модель нейросети"""

from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self,
        n_layers: int,
        hidden: int,
        dropout: float,
        n_classes: int = 10,
        in_features: int = 64,
    ) -> None:
        """Создаёт MLP с заданной архитектурой.

        params:
            n_layers - число скрытых слоёв (>= 1)
            hidden - размер скрытого слоя (>= 1)
            dropout - вероятность dropout, в [0, 1)
            n_classes - число выходных классов (>= 2)
            in_features - размер входа (число признаков)

        raise:
            ValueError - при некорректных значениях аргументов
        """
        super().__init__()

        if n_layers < 1:
            raise ValueError("n_layers должен быть положительным")
        if hidden < 1:
            raise ValueError("hidden должен быть положительным")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout должен быть в пределах [0, 1)")
        if n_classes < 2:
            raise ValueError("n_classes должен состоять хотя бы из 2 классов")
        if in_features < 1:
            raise ValueError("in_features должен быть положительным")

        self.n_layers = n_layers
        self.hidden = hidden
        self.dropout = dropout
        self.n_classes = n_classes
        self.in_features = in_features

        layers: list[nn.Module] = []

        size = in_features
        for _ in range(n_layers):
            layers += [nn.Linear(size, hidden), nn.ReLU(), nn.Dropout(dropout)]
            size = hidden
        layers.append(nn.Linear(size, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Прямой проход: возвращает логиты формы (batch, n_classes)."""
        return self.net(x)


def build_model(model_cfg: dict, device: str) -> MLP:
    """Создаёт MLP из секции model конфигурации

    params:
        model_cfg - словарь {n_layers, hidden, dropout}
        devide - устройство, на котором будет модель

    return: модель с заданными параметрами
    """
    return MLP(
        n_layers=model_cfg["n_layers"],
        hidden=model_cfg["hidden"],
        dropout=model_cfg["dropout"],
    ).to(device)
