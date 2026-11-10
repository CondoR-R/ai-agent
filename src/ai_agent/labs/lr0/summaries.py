"""Сводки запуска ЛР0 для вывода в терминал"""


def config_summary(cfg: dict) -> str:
    """Краткая выжимка конфигурации ЛР0."""
    model = cfg["model"]
    train = cfg["train"]
    return (
        f"model={model['n_layers']}×{model['hidden']} "
        f"(dropout={model['dropout']}), train={train['epochs']} эпох, "
        f"batch={train['batch_size']}, lr={train['lr']}"
    )


def metrics_summary(metrics: dict) -> str:
    """Форматирует итоговые метрики ЛР0."""
    parts = []
    if "val_acc" in metrics:
        parts.append(f"val_acc={metrics['val_acc']:.4f}")
    if "val_loss" in metrics:
        parts.append(f"val_loss={metrics['val_loss']:.4f}")
    if "test_acc" in metrics:
        parts.append(f"test_acc={metrics['test_acc']:.4f}")
    if "train_time" in metrics:
        parts.append(f"train_time={metrics['train_time']:.2f} с")
    return ", ".join(parts)
