# ai-agent

Учебный проект по курсу «Интеллектуальные системы»: воспроизводимые
ML-эксперименты и постепенный переход к агентным системам на языковых
моделях.

Проект развивается от одного контролируемого запуска к полноценной
системе: каркас запусков и журналирования, данные и конфигурации,
эксперименты, отчётность. Каждая следующая лабораторная должна
переиспользовать этот каркас, а не собирать эксперименты заново.

Сейчас реализованы **ЛР0** — воспроизводимый эксперимент с подбором
гиперпараметров, **ЛР1** — инференс языковой модели на CPU: паспорт
сервера, замеры TTFT/TPOT, кеш префикса и нагрузочное тестирование, и
**ЛР2** — базовый RAG на SciFact: плотный поиск, сборка контекста,
структурированные ответы и разбор ошибок. Остальные части (агенты с
инструментами и их оценка) пока не реализованы.

## Технологии

| Технология | Роль |
|---|---|
| Python 3.12 | язык проекта, версия зафиксирована в `.python-version` |
| `uv` | окружение и Lock-файл зависимостей |
| PyTorch | обучение MLP в ЛР0 |
| scikit-learn | датасет `load_digits`, стратифицированное разбиение |
| NumPy | массивы признаков и меток |
| MLflow | журнал запусков: параметры, метрики, артефакты, теги |
| Optuna | поиск гиперпараметров, TPE и MedianPruner |
| `llama.cpp` | OpenAI-совместимый сервер инференса для ЛР1 (CPU) |
| OpenAI Python client | потоковые запросы, `usage`, `ignore_eos` |
| Transformers | токенизатор Qwen2.5 и параметры KV-кеша |
| sentence-transformers | плотные эмбеддинги BGE-small для поиска ЛР2 |
| matplotlib | графики отчётных артефактов |
| PyYAML | конфигурации экспериментов |
| Git | фиксация кода и проверка чистоты дерева перед запуском |

## Что сделано

### Каркас запусков

- запуск одной конфигурации одной командой;
- каталог `runs/<run_id>/` с `config.yaml`, `metadata.json`,
  `history.csv`, `predictions.csv`, `metrics.json` и `error.txt`;
- статусы `running → completed/failed`, инкрементальная запись истории;
- проверка чистого git-дерева и запись полного commit hash;
- фиксация seed, deterministic-режима и числа потоков;
- запись версий библиотек и SHA-256 `data/split.json`;
- интеграция с MLflow: параметры, метрики по эпохам, итоговые метрики,
  артефакты и теги;
- единый формат сообщений в терминале.

### ЛР0

- данные `load_digits`, разбиение 1077 / 360 / 360, `split_seed = 0`;
- MLP `64 → n_layers × hidden (ReLU, dropout) → 10`;
- эксперимент 1: повторяемость, серия seed 1–5, сравнение предсказаний;
- эксперимент 2: Optuna, study `lr0_search`, цель 40 trial, продолжение
  study после перерыва;
- эксперимент 3: base vs best на seed 101–105, оценка на test и попарный
  подсчёт исправленных/испорченных изображений;
- отчётные артефакты: JSON, два графика поиска, Markdown-отчёт.

### ЛР1

- CPU-профиль: `llama.cpp` с Qwen2.5-1.5B-Instruct в GGUF Q4_K_M,
  Apple M1 / 8 ГБ ОЗУ, температура 0; контекст 32 768, в опыте 3 —
  65 536 на 32 слота по 2048 токенов;
- `endpoint.yaml` с адресом и параметрами генерации; скрипты
  `scripts/lr1/start_llama.sh` и `stop_llama.sh`, журнал и команда
  запуска — в `/tmp/lr1-llama/` вне репозитория;
- паспорт сервера: версия, модель и ревизия, CPU/RAM, KV-кеш, проверки
  ёмкости кеша, ровно заданного числа токенов, длины запроса и
  детерминизма;
- опыт 1: TTFT от длины запроса (256–16 384 токенов, 5 повторов) и
  повтор одного запроса с кешем префикса;
- опыт 2: полное время и TPOT от длины ответа (16–1024 токена);
- опыт 3: нагрузка 1–32 клиента, `max(8, 2c)` запросов на уровень,
  очередь и KV-кеш до и после уровня;
- отчётные артефакты: JSON, Markdown-таблицы, 7 графиков, прогноз,
  паспорт и итоговый `reports/lr1/report.md`.

Ключевые результаты CPU-профиля: TTFT растёт с 0.76 с на 256 токенах до
64.9 с на 16 384; TPOT ≈30 мс (33 ток/с) против нижней границы 16 мс;
кеш префикса ускоряет повтор запроса на 8192 токена в 331–432 раза;
пропускная способность насыщается около 34 ток/с, а пределы TTFT p90 ≤ 10 с
и TPOT ≤ 150 мс выдерживают только 1–2 клиента.

### ЛР2

- официальный SciFact: 5183 документа, dev 300, split 90/210 и manifest
  с SHA-256 архива, файлов и паспортом индекса;
- плотный поиск `BAAI/bge-small-en-v1.5`, матрица `(5183, 384)` и
  соответствие `doc_id`; обрезано 8.8% аннотаций;
- бюджет 2048 токенов, сборка документов по рангу с частичным добавлением
  последнего документа; ответ по JSON Schema с `doc_id` и предложениями;
- эксперимент 1: LLM-only, RAG k=3 и oracle (240 вызовов);
- эксперимент 2: RAG k=1 и k=5 на тех же 90 утверждениях (180 вызовов);
- проверки форматов, ссылок и метрик; попарные переходы и разбор четырёх
  этапов ошибок;
- отчётные артефакты: прогноз, проверки, сводка ошибок и
  `reports/lr2/report.md`.

Ключевой результат ЛР2: поиск хорош (Hit@3 = 0.85, обоснование передано
в 85% случаев), но RAG не улучшил ответы. Accuracy LLM-only = 0.333,
RAG k=3 = 0.256, oracle = 0.667; лучший режим RAG — k=1 (0.389).
Модель 1.5B с несколькими документами почти всегда выбирает CONTRADICT
и не выбирает NOINFO, а точное короткое свидетельство oracle использует
заметно лучше.

### Пока не сделано

- агент с инструментами, мультиагентная система;
- оценка агентов и защита от prompt injection.

## Структура проекта

```
ai-agent/
├── src/ai_agent/          # каркас: запуск, MLflow, metadata, файлы, seed
│   └── labs/
│       ├── lr0/           # ЛР0: данные, модель, обучение, поиск
│       ├── lr1/           # ЛР1: паспорт, клиент, опыты 1–3, анализ
│       └── lr2/           # ЛР2: данные, индекс, контекст, RAG, оценка
├── scripts/lr0/           # CLI: подготовка данных, запуски, эксперименты, отчёт
├── scripts/lr1/           # CLI ЛР1 + start/stop llama.cpp
├── scripts/lr2/           # CLI ЛР2: данные, индекс, проверки, опыты
├── configs/lr0/           # base, best, search space
├── configs/lr1/           # паспорт и опыты 1–3
├── configs/lr2/           # индекс, эксперименты 1–2
├── data/                  # split.json, manifest.json, данные ЛР1/ЛР2
│   ├── lr1/               # Pride and Prejudice для запросов ЛР1
│   └── lr2/               # SciFact, split, индекс и doc_ids
├── reports/lr0/           # JSON, графики, шаблон и report.md
├── reports/lr1/           # паспорт, JSON, графики, прогноз и report.md
├── reports/lr2/           # прогноз, проверки, анализ ошибок и report.md
├── runs/                  # каталоги запусков (не в git)
├── mlartifacts/           # артефакты MLflow (не в git)
├── mlflow.db              # журнал MLflow (не в git)
├── optuna.db              # study Optuna (не в git)
├── pyproject.toml
└── uv.lock
```

## Как запускать

Общий порядок:

1. установить зависимости;
2. убедиться, что дерево git чистое;
3. запустить нужный скрипт;
4. результаты искать в `runs/`, `reports/` и MLflow.

```bash
uv sync
```

Итоговые запуски начинаются только из закоммиченного кода: каркас
проверяет `git status --porcelain`. Каталоги запусков, базы MLflow/Optuna
и `mlartifacts/` в git не попадают.

### ЛР0: быстрый запуск

```bash
# один запуск
uv run python scripts/lr0/run.py --config configs/lr0/base.yaml --exp 0 --series demo

# эксперименты 1–3
uv run python scripts/lr0/experiment_1.py --config configs/lr0/base.yaml
uv run python scripts/lr0/experiment_2.py --save-best configs/lr0/best.yaml
uv run python scripts/lr0/experiment_3.py

# отчёт и MLflow UI
uv run python scripts/lr0/report.py
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Подробности ЛР0, таблицы и графики появятся в `reports/lr0/report.md`
после запуска `report.py`; конфигурации — в `configs/lr0/`, разбиение — в
`data/split.json`.

### ЛР1: быстрый запуск

Для ЛР1 нужен установленный `llama-server` и скачанная модель; сервер
запускается вне каркаса, а эксперименты — через `run.py`, поэтому дерево
git должно быть чистым перед каждым запуском.

```bash
brew install llama.cpp

# текст запросов; в репозитории он уже есть, команда — для пересборки
uv run python scripts/lr1/prepare_data.py

# паспорт; кеш промпта выключен
bash scripts/lr1/start_llama.sh --no-cache-prompt
uv run python scripts/lr1/passport.py \
  --config configs/lr1/passport.yaml --series passport --exp 0
# паспорт копируется в reports/lr1/passport.json — закоммитьте его,
# иначе следующий run.py остановится из-за грязного дерева

# опыт 1: длина запроса (сервер всё ещё с --no-cache-prompt)
uv run python scripts/lr1/experiment_1.py \
  --config configs/lr1/exp1_length.yaml --series ttft_vs_length --exp 1

# опыт 1: кеш префикса; перезапуск без флага, кеш включён по умолчанию
bash scripts/lr1/stop_llama.sh
bash scripts/lr1/start_llama.sh
uv run python scripts/lr1/experiment_1.py \
  --config configs/lr1/exp1_cache.yaml --series prefix_cache --exp 1

# опыт 2: длина ответа
uv run python scripts/lr1/experiment_2.py \
  --config configs/lr1/exp2.yaml --series decode_memory --exp 2

# опыт 3: нагрузка; нужен больший общий контекст и 32 слота
bash scripts/lr1/stop_llama.sh
CTX=65536 PARALLEL=32 bash scripts/lr1/start_llama.sh
uv run python scripts/lr1/experiment_3.py \
  --config configs/lr1/exp3.yaml --series load --exp 3

# отчёты по каталогам запусков из вывода экспериментов
uv run python scripts/lr1/report_exp1.py \
  --length-run runs/<exp1-length> --cache-run runs/<exp1-cache>
uv run python scripts/lr1/report_exp2.py --run runs/<exp2>
uv run python scripts/lr1/report_exp3.py --run runs/<exp3>
uv run python scripts/lr1/report.py
```

Журнал сервера — `/tmp/lr1-llama/llama.log`, команда запуска —
`/tmp/lr1-llama/llama.cmd`. Готовый отчёт и графики лежат в
`reports/lr1/`, паспорт — в `reports/lr1/passport.json`.

### ЛР2: быстрый запуск

```bash
# данные SciFact, split и manifest
uv run python scripts/lr2/prepare_data.py

# матрица эмбеддингов и паспорт индекса
uv run python scripts/lr2/build_index.py --config configs/lr2/index.yaml

# проверки п. 3.4 без вызовов модели
uv run python scripts/lr2/checks.py

# сервер llama.cpp из ЛР1 и два эксперимента
bash scripts/lr1/start_llama.sh
uv run python scripts/lr2/experiment_1.py   --config configs/lr2/exp1.yaml --series main --exp 1
uv run python scripts/lr2/experiment_2.py   --config configs/lr2/exp2.yaml --series main --exp 2
bash scripts/lr1/stop_llama.sh
```

Итоговый отчёт — `reports/lr2/report.md`, прогноз — `forecast.md`,
машинные проверки — `checks.json`, разбор ошибок — `analysis.json`.
Каталоги запусков лежат в `runs/lr2-experiment1-*` и
`runs/lr2-experiment2-*`.

## Воспроизводимость

- код между запусками не меняется: итоговый запуск возможен только из
  закоммиченного состояния;
- фиксируются seed, число потоков и deterministic-алгоритмы PyTorch;
- в `metadata.json` сохраняются commit, версии, seed и хеш разбиения;
- в ЛР1 фиксируются модель и ревизия, версия `llama.cpp`, команда запуска,
  тип KV-кеша и его ёмкость, CPU/RAM;
- параметры, метрики и артефакты дублируются в MLflow (`lr1` и
  `lr1_report`);
- каталоги `runs/`, `mlartifacts/`, базы `mlflow.db` и `optuna.db`,
  а также `.cache/` исключены из git.

## Ограничения

- реализованы ЛР0, ЛР1 и ЛР2; агенты и их оценку ещё предстоит реализовать;
- ЛР1 выполнен в CPU-профиле на Apple M1 с 8 ГБ ОЗУ: числа TTFT/TPOT и
  предел нагрузки нельзя переносить на GPU, другую модель или квантование;
- для ЛР1 и ЛР2 нужен внешний `llama-server` и скачанная модель; серии с
  разным режимом кеша и опыт 3 требуют перезапуска сервера с другими флагами;
- качество ЛР2 ограничено маленькой моделью: при длинном многодокументном
  контексте Qwen2.5-1.5B почти всегда отвечает CONTRADICT и не выбирает
  NOINFO; это видно в отчёте и не скрывается метриками;
- `--split-seed` принимается CLI, но разбиение ЛР0 берётся из
  `data/split.json`; для изменения сплита нужно пересобрать данные.

## Автор

Dmitriy Latkin, 2026.
