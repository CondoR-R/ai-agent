"""Запросы заданной длины в токенах (ЛР1, п. 3.3)"""

INSTRUCTION = "Summarize the following text in one sentence."

# число итераций подгонки длины: после decode -> encode граница токенов
# может сдвинуться на пару токенов
MAX_FIT_ITERATIONS = 10


def load_tokenizer(repo: str, revision: str | None = None):
    """Токенизатор модели (ревизию записывайте в metadata).

    params:
        repo - репозиторий на Hugging Face
        revision - коммит; None - ревизия из локального кеша / main
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(repo, revision=revision)


class PromptBuilder:
    """Строит сообщения чата с началом текста нужной длины.

    Структура запроса: уникальная метка (номер запроса) стоит в самом
    начале, затем инструкция и кусок текста. Метка не даёт кешированию
    префикса переиспользовать предыдущие запросы: общим остаётся только
    служебный префикс шаблона, он короче блока кеша.
    """

    def __init__(self, text: str, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.ids = tokenizer.encode(text, add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def _messages(request_id: int, body: str) -> list[dict]:
        content = f"[request {request_id:06d}] {INSTRUCTION}\n\n{body}"
        return [{"role": "user", "content": content}]

    def count(self, messages: list[dict]) -> int:
        """Длина запроса в токенах после шаблона диалога.

        Шаблон применяется в строку, а строка токенизируется отдельно:
        так результат не зависит от формата, который возвращает
        apply_chat_template в разных версиях transformers.
        """
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return len(self.tokenizer.encode(rendered, add_special_tokens=False))

    def spread_offset(
        self, request_id: int, n_tokens: int, stride: int = 9973
    ) -> int:
        """Детерминированное смещение начала текста для разных запросов.

        Нужно, когда запросы должны различаться текстом (опыт 3).
        """
        room = len(self.ids) - n_tokens
        if room <= 0:
            raise ValueError(
                f"Текст ({len(self.ids)} токенов) короче запроса "
                f"({n_tokens} токенов)"
            )
        return (request_id * stride) % room

    def build(
        self,
        request_id: int,
        n_tokens: int,
        offset: int = 0,
        tolerance: int = 4,
    ) -> tuple[list[dict], int]:
        """Запрос длиной n_tokens токенов после шаблона диалога.

        params:
            request_id - номер запроса (уникальная метка)
            n_tokens - целевая длина всего запроса
            offset - с какого токена текста начинать (0 - начало текста)
            tolerance - допустимое отклонение длины, токенов

        return: (сообщения, длина по локальному токенизатору). Реальную
                длину нужно сверять с usage.prompt_tokens сервера.

        raise:
            ValueError, если подогнать длину не удалось
        """
        empty = self.count(self._messages(request_id, ""))
        if n_tokens <= empty:
            raise ValueError(
                f"Запрос в {n_tokens} токенов короче служебной части "
                f"({empty} токенов)"
            )

        k = n_tokens - empty
        best: tuple[list[dict], int] | None = None
        for _ in range(MAX_FIT_ITERATIONS):
            k = max(1, min(k, len(self.ids) - offset))
            piece = self.ids[offset : offset + k]
            # срез по границе токена может разрезать многобайтный символ
            body = self.tokenizer.decode(piece).strip("\ufffd")
            messages = self._messages(request_id, body)
            n = self.count(messages)
            if best is None or abs(n - n_tokens) < abs(best[1] - n_tokens):
                best = (messages, n)
            if n == n_tokens:
                break
            k += n_tokens - n

        assert best is not None
        if abs(best[1] - n_tokens) > tolerance:
            raise ValueError(
                f"Не удалось подогнать длину: нужно {n_tokens}, "
                f"получено {best[1]} токенов"
            )
        return best
