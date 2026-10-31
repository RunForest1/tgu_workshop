# Этап 4: признаки и выбор метода — комплект сдачи

## Запуск (из code/, рядом нужны результаты этапа 2 и 4_features/data из tgu_workshop)
```bash
pip install pandas numpy scikit-learn
python features_build.py        # -> features.csv, labels.csv, split.csv, feature_audit.csv, config.json
python baseline.py               # -> baseline_train_metrics.csv, baseline_summary.txt
python apply_preprocessing.py    # -> X_columns_and_shapes.txt
python features_checks.py        # -> checks_results.csv (8 проверок задания)
```
Пути к данным заданы в начале `features_build.py` — поправьте под своё расположение файлов.

## Состав
- `features.csv` — ключи + 20 признаков (с сохранёнными пропусками), 21 548 строк.
- `labels.csv` — ключи, `churn_target`, `target_end_exclusive`.
- `split.csv` — принадлежность train/validation/test.
- `feature_audit.csv` — служебные поля: число известных месяцев в агрегатах, максимальная использованная `available_at`, статус цифрового источника.
- `config.json` — даты, горизонт, активная база, с явным объяснением отклонения от примера задания.
- `feature_catalog.csv` — паспорт 20 признаков (смысл, формула, источник, окно, доступность, обработка пропуска/нуля, связь с EDA, риск утечки).
- `baseline_train_metrics.csv` / `baseline_summary.txt` — Recall@K/Precision@K базового правила (сортировка по долгу) на train по датам.
- `X_columns_and_shapes.txt` — список столбцов X, размеры X_train/X_validation/X_test после преобразования.
- `checks_results.csv` — результаты 8 проверок из задания, все пройдены.
- `План_моделирования.md` — план на 2–3 страницы по шаблону курса.
- `code/` — весь код, включая переданный `preprocessing_template.py`.

## Главное решение, перенесённое с этапа 3
Горизонт — 30 дней (по уставу), не календарный месяц. Сетка T и даты validation/test
сдвинуты на месяц раньше примера из задания: train 10.2025–06.2026, validation 07.2026,
test 08.2026 (вместо дек.2025–июл.2026 / авг. / сен.) — иначе T=09.2026 требовал бы данных
о расторжениях вплоть до 01.10.2026 включительно, а журнал событий полон только строго до
этой даты. Подробности — в `config.json` (`deviation_note`) и `План_моделирования.md`.
