# ЛР0. Воспроизводимый эксперимент: подбор гиперпараметров MLP

{{ report_meta }}

## 1. Вопрос и прогноз

**Вопрос.** Дал ли подбор гиперпараметров улучшение, которое больше случайного
разброса?

<!-- manual:forecast -->
_Впишите прогноз, зафиксированный до итоговой серии: ожидаемый разброс
`val_acc` по seed; насколько лучший trial превзойдёт базовую конфигурацию;
сохранится ли улучшение на новых seed и на test._
<!-- /manual:forecast -->

## 2. Что сделано

- Каркас запусков: конфигурация, каталог `runs/<run_id>/`, `metadata.json`,
  `history.csv`, `predictions.csv`, `metrics.json`, запись в MLflow.
- Данные: `load_digits`, признаки /16, стратифицированное разбиение
  20 % / 25 % остатка, `split_seed = 0`, хеш `split.json` в манифесте.
- Модель: MLP `64 → n_layers × hidden (ReLU, dropout) → 10`, кросс-энтропия,
  Adam, 30 эпох; seed и число потоков фиксируются.
- Эксперимент 1: повторяемость, серия `base_seeds` (seed 1–5), сравнение
  предсказаний двух seed на val.
- Эксперимент 2: Optuna, `TPESampler(seed=42)`, `MedianPruner(5, 5)`,
  пространство параметров из `configs/lr0/search_space.yaml`.
- Эксперимент 3: `base` и `best`, seed 101–105, оценка на test, попарное
  сравнение по изображениям.

{{ data_table }}

{{ configs_table }}

## 3. Результаты

### 3.1. Повторяемость и разброс (эксперимент 1)

{{ exp1_repeatability }}

{{ exp1_seeds_table }}

{{ exp1_summary }}

{{ exp1_comparison }}

{{ exp1_examples }}

### 3.2. Поиск гиперпараметров (эксперимент 2)

{{ exp2_states_table }}

{{ exp2_epochs }}

{{ exp2_time }}

{{ exp2_best_table }}

{{ exp2_importance_table }}

{{ exp2_plots }}

### 3.3. Честное сравнение (эксперимент 3)

{{ exp3_table }}

{{ exp3_summary }}

{{ exp3_three_numbers }}

## 4. Разбор примеров и ошибок

<!-- manual:error_analysis -->
_Впишите разбор: какие классы и изображения путаются, что показывают
`predictions.csv` и графики, какие ошибки общие для разных seed._
<!-- /manual:error_analysis -->

## 5. Выводы и границы

<!-- manual:conclusions -->
_Ответьте на главный вопрос: сопоставьте Δ_s на test с разбросом из
эксперимента 1; укажите, где улучшение устойчиво, а где данных недостаточно.
Опишите границы: датасет, CPU, 30 эпох, одно разбиение._
<!-- /manual:conclusions -->

## 6. Ограничения и что улучшить

<!-- manual:limitations -->
_Ограничения и возможные продолжения: другой seed семплера, RandomSampler,
n_jobs, повторные разбиения, больше seed._
<!-- /manual:limitations -->

## 7. Кодогенерация

<!-- manual:codegen -->
_Укажите, какие части кода сгенерированы ассистентом и как проверены:
тестами, пересчётом на известном примере, чтением кода._
<!-- /manual:codegen -->

## Приложение. Артефакты

- `reports/lr0/experiment_1.json`, `experiment_2.json`,
  `experiment_3.json` — числовые результаты.
- `reports/lr0/experiment_2_trials.png`,
  `experiment_2_importance.png` — графики поиска.
- `runs/` — каталоги запусков, `mlflow.db` — журнал MLflow,
  `optuna.db` — study `lr0_search`.
