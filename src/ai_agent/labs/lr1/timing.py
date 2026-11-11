"""Метрики времени ответа и разбор потока (Правила, п. 7; ЛР1, п. 2.3)"""


def compute_timings(
    t0: float, t1: float | None, t_last: float | None, n_tokens: int | None
) -> dict:
    """TTFT, TPOT и полное время ответа по отметкам времени.

    TTFT = t1 - t0; TPOT = (tN - t1) / (N - 1); e2e = tN - t0.
    В знаменателе N - 1: первый токен уже входит в TTFT, между первым и
    последним токенами N - 1 интервал.

    params:
        t0 - перед отправкой запроса
        t1 - пришёл первый фрагмент текста
        t_last - пришёл последний фрагмент текста
        n_tokens - число токенов ответа (из usage)

    return: {"ttft_s", "tpot_s", "e2e_s"}; None, если величину не посчитать
            (нет ответа или один токен для TPOT)
    """
    ttft = None if t1 is None else t1 - t0
    e2e = None if t_last is None else t_last - t0
    tpot = None
    if (
        t1 is not None
        and t_last is not None
        and n_tokens is not None
        and n_tokens > 1
    ):
        tpot = (t_last - t1) / (n_tokens - 1)
    return {"ttft_s": ttft, "tpot_s": tpot, "e2e_s": e2e}


class StreamRecorder:
    """Накапливает фрагменты потокового ответа и строит запись результата.

    Клиент вызывает feed(chunk, now) для каждого фрагмента, где now —
    time.perf_counter(), взятый сразу после получения фрагмента.
    """

    def __init__(self, t0: float) -> None:
        self.t0 = t0
        self.t1: float | None = None
        self.t_last: float | None = None
        self.usage = None
        self.finish_reason: str | None = None
        self.parts: list[str] = []
        self.chunk_times: list[float] = []

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def feed(self, chunk, now: float) -> None:
        """Обрабатывает один фрагмент потока."""
        if getattr(chunk, "usage", None) is not None:
            # последний фрагмент потока несёт только usage
            self.usage = chunk.usage
        if not chunk.choices:
            return
        choice = chunk.choices[0]
        if choice.finish_reason:
            self.finish_reason = choice.finish_reason
        content = getattr(choice.delta, "content", None)
        if content:
            if self.t1 is None:
                self.t1 = now
            self.t_last = now
            self.parts.append(content)
            self.chunk_times.append(now - self.t0)

    def record(
        self,
        *,
        request_id: str,
        condition: dict | None,
        warmup: bool,
        t_end: float,
        error: str | None = None,
        keep_text: bool = False,
        keep_chunk_times: bool = False,
    ) -> dict:
        """Запись для results.jsonl.

        Число токенов берётся из usage сервера, а не из числа фрагментов:
        в одном фрагменте может прийти несколько токенов.

        params:
            request_id - идентификатор запроса
            condition - условие опыта (длина запроса, число клиентов, ...)
            warmup - прогревочный запрос (в статистику не входит)
            t_end - perf_counter после конца потока
            error - ошибка вызова, если была
            keep_text - сохранить текст ответа
            keep_chunk_times - сохранить время каждого фрагмента
        """
        prompt_tokens = getattr(self.usage, "prompt_tokens", None)
        completion_tokens = getattr(self.usage, "completion_tokens", None)

        problem = error
        if problem is None and self.usage is None:
            problem = "сервер не прислал usage"
        if problem is None and self.t1 is None:
            problem = "ответ без текста"

        rec = {
            "id": request_id,
            "condition": condition or {},
            "status": "error" if problem else "ok",
            "error": problem,
            "warmup": warmup,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            **compute_timings(self.t0, self.t1, self.t_last, completion_tokens),
            "finish_reason": self.finish_reason,
            "n_chunks": len(self.chunk_times),
            # отметки одних часов perf_counter: по ним считается
            # пропускная способность уровня нагрузки
            "t_send_s": self.t0,
            "t_done_s": t_end,
        }
        if keep_text:
            rec["text"] = self.text
        if keep_chunk_times:
            rec["chunk_times_s"] = [round(t, 6) for t in self.chunk_times]
        return rec
