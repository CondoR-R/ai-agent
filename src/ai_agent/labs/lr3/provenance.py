"""SHA-256 входных файлов для metadata.json."""

from pathlib import Path

from ai_agent import files
from ai_agent.labs.lr2 import data, index


def input_sha256(cfg: dict, *, selection_path: Path | None = None) -> dict:
    """Хеши endpoint, манифеста, данных, индекса и ссылочных артефактов."""
    result = {
        files.ENDPOINT_PATH.name: files.get_sha256(files.ENDPOINT_PATH),
        files.MANIFEST_PATH.name: files.get_sha256(files.MANIFEST_PATH),
        data.CORPUS_REL: files.get_sha256(data.CORPUS_PATH),
        data.CLAIMS_DEV_REL: files.get_sha256(data.CLAIMS_DEV_PATH),
        data.INDEX_REL: files.get_sha256(index.save_path()),
        data.DOC_IDS_REL: files.get_sha256(data.DOC_IDS_PATH),
    }
    reference = cfg.get("dense_reference_run")
    if reference:
        path = files.PROJECT_ROOT / reference / files.RESULTS_FILENAME
        result["lr2_dense_results"] = files.get_sha256(path)
    if selection_path is not None:
        result[str(selection_path.relative_to(files.PROJECT_ROOT))] = (
            files.get_sha256(selection_path)
        )
    return result
