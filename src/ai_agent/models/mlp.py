"""Многослойный перцептрон для классификации."""

from torch import Tensor, nn


class MLP(nn.Module):
    """Полносвязная сеть: in_features → n_layers скрытых слоёв → n_classes.

    Каждый скрытый слой: Linear → ReLU → Dropout. Выходной слой —
    Linear без активации, возвращает логиты для CrossEntropyLoss.

    Пример архитектуры при n_layers=2:
        64 → hidden → ReLU → Dropout
        hidden → hidden → ReLU → Dropout
        hidden → 10 (логиты)
    """

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

        self.n_layers = n_layers
        self.hidden = hidden
        self.dropout = dropout
        self.n_classes = n_classes
        self.in_features = in_features

        layers = []

        # первый скрытый слой
        layers.append(nn.Linear(self.in_features, self.hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout))

        # остальные скрытые слои
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(self.hidden, self.hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

        # выходной слой
        layers.append(nn.Linear(self.hidden, self.n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Прямой проход: возвращает логиты формы (batch, n_classes)."""
        return self.net(x)
