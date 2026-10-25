from .model import MLP
import torch
from torch import nn, optim

class Trainer:
    def __init__(self, model: MLP, train_cfg: dict) -> None:
        """Создает тренера для MLP модели
        
        params:
            model - MLP модель
            train_cfg - конфигурация обучения, словарь с полями:
                {"lr", "weight_decay", "batch_size", "epochs"}
        """
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            train_cfg['lr'],
            weight_decay=train_cfg['weight_decay']
        )
        self.n_epochs = train_cfg['epochs']





