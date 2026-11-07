# -*- coding: utf-8 -*-
"""Этап 5: обучение baseline и обучаемых кандидатов, сравнение на train/validation."""
import json
import time
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from training_utils import load_train_validation, campaign_metrics, KEY
from model_factory import build_pipeline

ROOT = Path(__file__).parent
PREPARED = ROOT.parent / "model_prepared"  # ../model_prepared относительно code/
OUT = ROOT.parent                            # результаты пишутся в корень artefacts_5/
OUT.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config.json", encoding="utf-8") as f:
    CONFIG = json.load(f)

SEED = CONFIG["seed"]
FRACTION = CONFIG["campaign_fraction"]
NUMERIC = CONFIG["numeric_columns"]
CATEGORICAL = CONFIG["categorical_columns"]
BASELINE_FEATURE = CONFIG["baseline_feature"]

# ---------------------------------------------------------------------------
# Шаг 1: проверка входа (ключи, метки, список X, временные границы) — внутри load_train_validation
# ---------------------------------------------------------------------------
data = load_train_validation(PREPARED, NUMERIC, CATEGORICAL)
train_raw = pd.read_csv(PREPARED / "features.csv", sep=";", encoding="utf-8-sig")
print("Вход проверен (ключи, метки, список X, временные границы — через training_utils.load_train_validation).")
for part in ["train", "validation"]:
    print(f"  {part}: {len(data[part]['y'])} строк, положительных: {int(data[part]['y'].sum())}")

all_tables = []
summary_rows = []
errors_log = []

def record_metrics(exp_id, part, keys, y, score):
    summary, table, ranked = campaign_metrics(keys, y, score, FRACTION)
    table.insert(0, "part", part)
    table.insert(0, "experiment_id", exp_id)
    all_tables.append(table)
    summary_rows.append({
        "experiment_id": exp_id, "part": part,
        "macro_recall_at_K": summary["macro_recall_at_K"],
        "macro_precision_at_K": summary["macro_precision_at_K"],
        "macro_average_precision": summary["macro_average_precision"],
        "pooled_recall_at_K": summary["pooled_recall_at_K"],
        "n_rows": summary["n_rows"], "n_dates": summary["n_dates"],
    })
    return ranked

# ---------------------------------------------------------------------------
# Шаг 2: baseline — сортировка по известному долгу, неизвестный долг — после известных
# ---------------------------------------------------------------------------
for part in ["train", "validation"]:
    # ВАЖНО: берём признак прямо из data[part]['X'] — его порядок строк гарантированно совпадает
    # с data[part]['y'] и data[part]['keys'] (все три построены из одного отсортированного по KEY
    # кадра внутри load_train_validation). Объединение train_raw с keys заново рискует вернуть
    # строки в другом порядке (merge сохраняет порядок ЛЕВОЙ таблицы, а не keys) — это и есть
    # запрещённое позиционное присваивание меток без проверки порядка; исправлено.
    debt = pd.to_numeric(data[part]["X"][BASELINE_FEATURE], errors="coerce").reset_index(drop=True)
    known_min = debt[debt.notna()].min()
    sentinel = known_min - 1.0  # ниже любого известного значения; конечное число (campaign_metrics требует finite)
    score = debt.fillna(sentinel).to_numpy(dtype=float)
    record_metrics("baseline", part, data[part]["keys"], data[part]["y"], score)
print("Baseline посчитан на train и validation (неизвестный долг — ниже любого известного, конечный сентинел для сортировки).")

# ---------------------------------------------------------------------------
# Шаг 3–4: два обучаемых метода, 6 конфигураций (в пределах 3–6, максимум 8)
# ---------------------------------------------------------------------------
assert 3 <= len(CONFIG["experiments"]) <= 8, "План экспериментов должен быть 3-8 конфигураций"

candidate_predictions = {}
saved_pipelines = {}

for exp in CONFIG["experiments"]:
    exp_id = exp["id"]
    status = "ok"
    note = ""
    try:
        pipeline = build_pipeline(exp, NUMERIC, CATEGORICAL, seed=SEED)
        t0 = time.time()
        pipeline.fit(data["train"]["X"], data["train"]["y"])
        fit_seconds = time.time() - t0
    except Exception as e:  # сохраняем ошибку в журнале, не пропускаем опыт молча
        status, note = "error", str(e)
        errors_log.append({"experiment_id": exp_id, "error": note})
        print(f"[{exp_id}] ОШИБКА: {note}")
        continue

    positive_idx = int(np.flatnonzero(pipeline.classes_ == 1)[0])
    for part in ["train", "validation"]:
        score = pipeline.predict_proba(data[part]["X"])[:, positive_idx]
        ranked = record_metrics(exp_id, part, data[part]["keys"], data[part]["y"], score)
        if part == "validation":
            candidate_predictions[exp_id] = ranked
    saved_pipelines[exp_id] = pipeline
    summary_rows[-1]["fit_seconds"] = fit_seconds
    summary_rows[-2]["fit_seconds"] = fit_seconds
    summary_rows[-1]["status"] = status
    summary_rows[-2]["status"] = status
    print(f"[{exp_id}] fit={fit_seconds:.2f}с, validation macro Recall@K={summary_rows[-1]['macro_recall_at_K']:.4f}")

metrics_by_T = pd.concat(all_tables, ignore_index=True)
summary = pd.DataFrame(summary_rows)
metrics_by_T.to_csv(OUT / "metrics_by_T.csv", sep=";", encoding="utf-8-sig", index=False)
summary.to_csv(OUT / "experiments_summary.csv", sep=";", encoding="utf-8-sig", index=False)
pd.DataFrame(errors_log).to_csv(OUT / "errors_log.csv", sep=";", encoding="utf-8-sig", index=False)

# ---------------------------------------------------------------------------
# Шаг 6: выбор кандидата — заранее заданное правило
# ---------------------------------------------------------------------------
val_summary = summary[(summary.part == "validation") & (summary.experiment_id != "baseline")].copy()
val_summary = val_summary.sort_values(
    ["macro_recall_at_K", "macro_average_precision", "experiment_id"],
    ascending=[False, False, True], kind="stable",
)
chosen_id = val_summary.iloc[0]["experiment_id"]
chosen_row = val_summary.iloc[0]
baseline_val = summary[(summary.experiment_id == "baseline") & (summary.part == "validation")].iloc[0]
print(f"\nПравило выбора: максимальный validation macro Recall@K; при равенстве — максимальный validation AP; "
      f"при полном равенстве — первый опыт в порядке config.json.")
print(f"Выбранный кандидат: {chosen_id} (validation macro Recall@K={chosen_row['macro_recall_at_K']:.4f}, "
      f"baseline={baseline_val['macro_recall_at_K']:.4f})")

chosen_pipeline = saved_pipelines[chosen_id]

# ---------------------------------------------------------------------------
# Шаг 5/7: validation-прогнозы и метки выбранного кандидата отдельными файлами
# ---------------------------------------------------------------------------
ranked = candidate_predictions[chosen_id]
preds = ranked[KEY + ["score", "rank", "selected"]].copy()
preds.to_csv(OUT / "validation_predictions_chosen.csv", sep=";", encoding="utf-8-sig", index=False)
labels_out = ranked[KEY + ["target"]].rename(columns={"target": "churn_target"})
labels_out.to_csv(OUT / "validation_labels_chosen.csv", sep=";", encoding="utf-8-sig", index=False)

additional_found = int(ranked.loc[ranked.selected, "target"].sum()) - int(
    candidate_predictions.get("baseline", ranked).loc[ranked.selected, "target"].sum()
    if "baseline" in candidate_predictions else 0
)
# пересчитываем честно через сохранённые validation-таблицы baseline и кандидата
base_hits = int(metrics_by_T[(metrics_by_T.experiment_id == "baseline") & (metrics_by_T.part == "validation")].hits.sum())
chosen_hits = int(metrics_by_T[(metrics_by_T.experiment_id == chosen_id) & (metrics_by_T.part == "validation")].hits.sum())
additional_found = chosen_hits - base_hits
print(f"Доп. найденных уходов на validation против baseline (суммарно по датам): {additional_found} "
      f"({chosen_hits} против {base_hits})")

# ---------------------------------------------------------------------------
# Шаг 7: сохранение Pipeline целиком
# ---------------------------------------------------------------------------
bundle = {
    "pipeline": chosen_pipeline,
    "numeric_columns": NUMERIC,
    "categorical_columns": CATEGORICAL,
    "config": CONFIG,
    "chosen_experiment_id": chosen_id,
    "seed": SEED,
    "library_versions": {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    },
    "feature_version": "artefacts_4 (этап 4) + исключение T=2026-06-01 из train для этапа 5 (см. config.json.deviation_note)",
    "prepared_dir_used": str(PREPARED),
}
joblib.dump(bundle, OUT / "chosen_pipeline.joblib")
print(f"\nСохранён Pipeline: {OUT / 'chosen_pipeline.joblib'} (выбран: {chosen_id})")

with open(OUT / "chosen_candidate.json", "w", encoding="utf-8") as f:
    json.dump({
        "chosen_experiment_id": chosen_id,
        "selection_rule": "max validation macro Recall@K; tie -> max validation macro AP; tie -> first in config.json order",
        "validation_macro_recall_at_K": chosen_row["macro_recall_at_K"],
        "validation_macro_average_precision": chosen_row["macro_average_precision"],
        "baseline_validation_macro_recall_at_K": baseline_val["macro_recall_at_K"],
        "additional_churners_found_vs_baseline_validation_pooled": additional_found,
    }, f, ensure_ascii=False, indent=2)

print("\nГотово.")
