# Этап 2: сбор и подготовка данных — комплект сдачи

## Запуск
```bash
cd code
pip install pandas
python pipeline.py --raw <папка с 1_introduction и 2_preprocessing из tgu_workshop> --out ../  # пути см. ниже
python checks.py
```
`pipeline.py` ожидает на входе папку, где лежат подпапки `1_introduction/` и `2_preprocessing/`
(как в репозитории `tgu_workshop`). Все результаты (`*_clean.csv`, `customer_month.csv`,
`quarantine.csv`, `decision_log.csv`, `quality_summary_*.csv`) воспроизводимо
пересоздаются одним запуском; `checks.py` запускает пайплайн повторно и сравнивает результат
побайтово для проверки детерминированности.

## Состав
- `subscribers_clean.csv`, `billing_clean.csv`, `contract_events_clean.csv`, `support_clean.csv`, `customer_month.csv` — чистые таблицы.
- `quarantine.csv` — 40 обращений поддержки, исключённых из `support_clean` с указанием причины.
- `decision_log.csv` — журнал решений: проблема, действие, обоснование, число затронутых записей.
- `data_dictionary_lesson2.csv` — обновлённый словарь полей, включая новые агрегаты customer_month и правила их доступности.
- `quality_summary_before.csv` / `quality_summary_after.csv` — число строк по источникам до/после.
- `checks_results.csv` — результаты проверок из задания (повтор обращения, позднее поступление, сохранность абонента без обращений, объединение биллинга, уникальность ключа, повторный запуск).
- `extract_support_used.sql` — фактический SQL-запрос извлечения из `support.sqlite`.
- `Отчёт_о_качестве.md` — отчёт по шаблону курса (2–4 страницы).
- `code/pipeline.py`, `code/checks.py` — исходный код.

Версия зависимостей на момент подготовки: Python 3.12, pandas 2.x — зафиксируйте точную версию
вашего окружения командой `pip freeze | grep -i pandas` и укажите её в отчёте.
