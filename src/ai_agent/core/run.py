from . import git_utils, config, metadata, paths, metrics
from .config import ConfigDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


def run(
    cfg: ConfigDict,
    series: str,
    data_hashes: dict[str, str],
    lab_fn: Callable[[Path, dict, str], None],
    metric_fn: Callable[[Path], dict],
    extra_metadata: dict = {},
    metadata_fields: tuple[str, ...] = metadata.REQUIRED_FIELDS,
) -> Path:
    """Оркестратор жизненного цикла запуска

    params:
        cfg - словарь конфигурации (отвалидирован, объединен с cli)
        series - серия экспиремента (base_seeds, final, ...)
        data_hashes - словарь вида { имя_файла: хеш }
        lab_fn - фунция лаборатории
        metric_fn - функции расчета метрик для лаборатории, 
            принимает путь к директории run
        extra_metadata = уникальные поля metadata для lr
            (по умолчанию {})
        metadata_fields - поля для metadata.json
            (по умолчанию metadata.REQUIRED_FIELDS)
    """
    git_utils.check_clean_status()

    commit_hash = git_utils.get_commit_hash()
    versions = metadata.collect_versions(cfg["tracked_libs"])
    run_id = paths.get_run_id(cfg["experiment"], series, cfg["seed"])
    run_dir = paths.create_run_id_dir(run_id)
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    config.save(run_dir / paths.CONFIG_FILENAME, cfg)

    metadata_dict = {
        **extra_metadata,
        "run_id": run_id,
        "work": cfg["experiment"],
        "series": series,
        "status": "running",
        "started_at": started_at,
        "commit": commit_hash,
        "device": cfg["runtime"]["device"],
        "seed": cfg["seed"],
        "data_sha256": data_hashes,
        "versions": versions,
    }
    metadata.write(run_dir, metadata_dict, metadata_fields)

    try:
        lab_fn(run_dir, cfg, run_id)
        metrics.write(run_dir, metric_fn(run_dir))
        metadata.update_status(run_dir, 'completed')
    except BaseException as ex:
        metadata.write_error(run_dir, ex)
        metadata.update_status(run_dir, 'failed')
        raise 

    return run_dir
