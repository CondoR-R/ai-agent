"""Тесты для src/ai_agent/data/manifests.py."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from ai_agent.data.manifests import file_sha256


def _write_bytes(tmp_path: Path, name: str, data: bytes) -> Path:
    """Помощник: пишет байты в файл внутри tmp_path."""
    p = tmp_path / name
    p.write_bytes(data)
    return p


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
    h1 = file_sha256(p1)
    h2 = file_sha256(p2)
    assert h1 != h2


def test_large_file_crosses_block_boundary(tmp_path: Path):
    """Получения SHA-256 для больших файлов (больше BLOCK_SIZE = 8192)"""
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
