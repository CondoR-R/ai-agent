"""сравнение 2х csv файлов по указанным колонкам"""
import csv
from pathlib import Path


def compare_csv(
    path_a: Path,
    path_b: Path,
    columns: list[str],
) -> bool:
    """Сравнивает два CSV-файла по указанным колонкам.

    Порядок строк важен: строки сравниваются попарно (0-я с 0-й и т.д.).

    params:
        path_a  - первый файл
        path_b  - второй файл
        columns - список колонок для сравнения

    return: True, если по указанным колонкам файлы совпадают;
            False, если есть расхождения (в том числе разное число строк).

    raise:
        FileNotFoundError, если файл не найден
        KeyError, если в файле нет запрошенной колонки
    """
    path_a, path_b = Path(path_a), Path(path_b)

    with (
        path_a.open("r", newline="", encoding="utf-8") as fa,
        path_b.open("r", newline="", encoding="utf-8") as fb,
    ):
        reader_a = csv.DictReader(fa)
        reader_b = csv.DictReader(fb)

        header_a = reader_a.fieldnames or []
        header_b = reader_b.fieldnames or []

        missing_a = [c for c in columns if c not in header_a]
        missing_b = [c for c in columns if c not in header_b]
        if missing_a:
            raise KeyError(f"В {path_a} нет колонок: {missing_a}")
        if missing_b:
            raise KeyError(f"В {path_b} нет колонок: {missing_b}")

        for _, (row_a, row_b) in enumerate(zip(reader_a, reader_b, strict=False)):
            for col in columns:
                if row_a[col] != row_b[col]:
                    return False
        # Проверяем, что оба итератора исчерпаны одновременно
        extra_a = next(reader_a, None)
        extra_b = next(reader_b, None)
        if extra_a is not None or extra_b is not None:
            return False

    return True
