# Этап 5: построение и обучение модели — комплект сдачи

## Запуск (из code/)
```bash
pip install pandas numpy scikit-learn joblib
python train.py
# -> ../metrics_by_T.csv, ../experiments_summary.csv, ../errors_log.csv,
#    ../chosen_pipeline.joblib, ../chosen_candidate.json,
#    ../validation_predictions_chosen.csv, ../validation_labels_chosen.csv

python check_saved_model.py --model ../chosen_pipeline.joblib --prepared ../model_prepared \
  --out ../reload_check_predictions.csv --expected ../validation_predictions_chosen.csv
# -> проверка: Pipeline загружается без fit и даёт идентичные validation score/rank/top-K
```
Пути `PREPARED`/`OUT` заданы в начале `train.py` — поправьте под своё расположение.

## Состав
- `code/` — `train.py` (основной скрипт), `config.json` (список X, 6 экспериментов, пояснение отклонения
  по датам), `model_factory.py`/`preprocessing_template.py`/`training_utils.py`/`check_saved_model.py`
  (материалы курса, использованы как есть).
- `model_prepared/` — копии `features.csv`/`labels.csv`/`split.csv` этапа 4 **с исключением T=2026-06-01
  из train** (см. ниже) — исходные таблицы `artefacts_4/` не изменялись.
- `metrics_by_T.csv` — N/K/ушли/найдено/Recall@K/Precision@K/AP по каждой T, для baseline и всех 6 опытов.
- `experiments_summary.csv` — сводные macro/pooled-метрики по каждому опыту на train и validation.
- `chosen_pipeline.joblib` — сохранённый Pipeline (подготовка + классификатор) лучшего ОБУЧЕННОГО кандидата.
- `chosen_candidate.json` — правило выбора и его результат.
- `validation_predictions_chosen.csv` / `validation_labels_chosen.csv` — прогнозы и истинные метки
  выбранного кандидата на validation (ключи, score, rank, флаг top-K — отдельно от меток).
- `reload_check_predictions.csv` — результат перезагрузки Pipeline без fit (сверен с validation_predictions_chosen.csv).
- `Журнал_экспериментов.md`, `Отчёт_обучение.md` — по шаблонам курса.

## Главный результат (честно, не приукрашено)
**Ни одна из 6 обученных конфигураций не превзошла baseline (сортировку по долгу) на validation.**
Baseline: Recall@K=0,403 (25/62 найдено). Лучший обученный кандидат (`lr_c01`, логрегрессия C=0,1):
Recall@K=0,371 (23/62). HistGradientBoosting сильно переобучается (train Recall@K до 0,79 при
validation 0,29) — слишком мало данных (2 358 уникальных абонентов, 411 положительных на train)
для сложных деревьев. Baseline остаётся рабочей рекомендацией; подробности и разбор — в отчёте.

## Техническое отклонение, требующее внимания команды
`training_utils.load_train_validation` (материал курса) требует, чтобы конец горизонта последней
train-даты полностью укладывался до начала validation. При горизонте 30 дней и train до 2026-06-01
это нарушается (2026-07-02 > 2026-07-01 начала validation) — T=2026-06-01 исключён из `model_prepared/`
только для этого этапа (train: 8 дат вместо 9). Подробное обоснование — `code/config.json`
(`deviation_note`) и раздел «Вход и протокол» отчёта.

## Найденный и исправленный баг
Первая версия подсчёта baseline неверно выравнивала score с метками (merge терял порядок строк) —
train Recall@K выходил ≈0,08 вместо ожидаемых ≈0,40. Исправлено и явно описано в журнале экспериментов —
это тот самый тип ошибки («позиционное присваивание без проверки порядка»), который прямо запрещён заданием.
