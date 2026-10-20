"""Тесты для src/ai_agent/data/manifests.py."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from ai_agent.data.manifests import (
    file_sha256,
    find_entry,
    load_manifest,
    update_manifest,
)


def _write_bytes(tmp_path: Path, name: str, data: bytes) -> Path:
    """Помощник: пишет байты в файл внутри tmp_path."""
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _entry(source: str = "scikit-learn", version: str = "1.9.1") -> dict:
    """Помощник: минимальная запись набора для тестов update_manifest."""
    return {
        "source": source,
        "version": version,
        "rule": "test=20%, val=25% of rest",
        "sizes": {"train": 1077, "val": 360, "test": 360},
        "files": {"data/split.json": "0" * 64},
    }


# --- file_sha256 ---

def test_matches_hashlib_reference(tmp_path: Path):
    """Независимая реализация на тех же данных"""
    content = b"hello world\n" * 1000
    p = _write_bytes(tmp_path, "f.bin", content)
    expected = hashlib.sha256(content).hexdigest()
    assert file_sha256(p) == expected


def test_matches_sha256sum(tmp_path: Path):
    """Сверка с системным sha256sum — независимый источник истины.
    Если sha256sum недоступен, тест пропускается.
    """
    content = b"some content\nwith newlines\n"
    p = _write_bytes(tmp_path, "f.txt", content)
    try:
        out = subprocess.run(
            ["sha256sum", str(p)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("sha256sum недоступен в системе")
    expected = out.split()[0]
    assert file_sha256(p) == expected


def test_deterministic(tmp_path: Path):
    """Повторный вызов даст такой же результат"""
    p = _write_bytes(tmp_path, "f.bin", b"data" * 100)
    assert file_sha256(p) == file_sha256(p)


def test_avalanche_effect(tmp_path: Path):
    """Изменение одного байта полностью меняет хеш"""
    p1 = _write_bytes(tmp_path, "a.bin", b"abcdefgh")
    p2 = _write_bytes(tmp_path, "b.bin", b"abcdefgi")
    assert file_sha256(p1) != file_sha256(p2)


def test_large_file_crosses_block_boundary(tmp_path: Path):
    """Получение SHA-256 для больших файлов (больше BLOCK_SIZE = 8192)"""
    content = b"x" * 20000
    p = _write_bytes(tmp_path, "big.bin", content)
    expected = hashlib.sha256(content).hexdigest()
    assert file_sha256(p) == expected


def test_accepts_str_path(tmp_path: Path):
    """Одинаковый результат при пути типа str и Path"""
    p = _write_bytes(tmp_path, "f.bin", b"data")
    assert file_sha256(str(p)) == file_sha256(p)


def test_missing_file_raises(tmp_path: Path):
    """Файл не существует"""
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / "nope.bin")


# --- load_manifest ---

def test_load_manifest_missing_file(tmp_path: Path):
    """Отсутствующий файл → пустой скелет"""
    result = load_manifest(tmp_path / "manifest.json")
    assert result == {"datasets": []}


def test_load_manifest_empty_object(tmp_path: Path):
    """Пустой {} → добавляется ключ datasets"""
    p = tmp_path / "manifest.json"
    p.write_text("{}", encoding="utf-8")
    result = load_manifest(p)
    assert result == {"datasets": []}


def test_load_manifest_valid(tmp_path: Path):
    """Корректный файл читается как есть"""
    p = tmp_path / "manifest.json"
    payload = {"datasets": [{"name": "x"}]}
    p.write_text(json.dumps(payload), encoding="utf-8")
    result = load_manifest(p)
    assert result == payload


def test_load_manifest_top_level_not_dict(tmp_path: Path):
    """Верхний уровень не словарь → ValueError"""
    p = tmp_path / "manifest.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="объект"):
        load_manifest(p)


# --- update_manifest ---

def test_update_manifest_creates_file(tmp_path: Path):
    """Первый вызов создаёт файл"""
    p = tmp_path / "manifest.json"
    update_manifest(p, "lr0_load_digits", _entry())
    assert p.exists()


def test_update_manifest_adds_entry(tmp_path: Path):
    """Запись появляется в манифесте с полем name"""
    p = tmp_path / "manifest.json"
    update_manifest(p, "lr0_load_digits", _entry())
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data["datasets"]) == 1
    assert data["datasets"][0]["name"] == "lr0_load_digits"
    assert data["datasets"][0]["source"] == "scikit-learn"


def test_update_manifest_replaces_by_name(tmp_path: Path):
    """Повторный вызов с тем же name не создаёт дубликат"""
    p = tmp_path / "manifest.json"
    update_manifest(p, "lr0_load_digits", _entry(version="1.9.1"))
    update_manifest(p, "lr0_load_digits", _entry(version="2.0.0"))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data["datasets"]) == 1
    assert data["datasets"][0]["version"] == "2.0.0"


def test_update_manifest_keeps_other_entries(tmp_path: Path):
    """Добавление нового набора не стирает существующие"""
    p = tmp_path / "manifest.json"
    update_manifest(p, "lr0_load_digits", _entry())
    update_manifest(p, "lr2_scifact", _entry(source="SciFact"))
    data = json.loads(p.read_text(encoding="utf-8"))
    names = {d["name"] for d in data["datasets"]}
    assert names == {"lr0_load_digits", "lr2_scifact"}


def test_update_manifest_is_multiline(tmp_path: Path):
    """Файл пишется с отступами (indent=2)"""
    p = tmp_path / "manifest.json"
    update_manifest(p, "lr0_load_digits", _entry())
    text = p.read_text(encoding="utf-8")
    assert "\n" in text


def test_update_manifest_keeps_unicode(tmp_path: Path):
    """Кириллица не экранируется в \\uXXXX"""
    p = tmp_path / "manifest.json"
    entry = _entry()
    entry["rule"] = "правило разбиения"
    update_manifest(p, "lr0_load_digits", entry)
    text = p.read_text(encoding="utf-8")
    assert "правило разбиения" in text
    assert "\\u" not in text


def test_update_manifest_creates_parent_dir(tmp_path: Path):
    """Родительский каталог создаётся, если его нет"""
    p = tmp_path / "nested" / "deep" / "manifest.json"
    update_manifest(p, "lr0_load_digits", _entry())
    assert p.exists()


# --- find_entry ---

def test_find_entry_returns_entry(tmp_path: Path):
    """Находит запись по имени"""
    p = tmp_path / "manifest.json"
    update_manifest(p, "lr0_load_digits", _entry())
    found = find_entry(p, "lr0_load_digits")
    assert found is not None
    assert found["name"] == "lr0_load_digits"
    assert found["source"] == "scikit-learn"


def test_find_entry_returns_none(tmp_path: Path):
    """Неизвестное имя → None"""
    p = tmp_path / "manifest.json"
    update_manifest(p, "lr0_load_digits", _entry())
    assert find_entry(p, "unknown") is None


def test_find_entry_on_missing_file(tmp_path: Path):
    """Отсутствующий файл → None (манифест пустой)"""
    assert find_entry(tmp_path / "manifest.json", "lr0_load_digits") is None
