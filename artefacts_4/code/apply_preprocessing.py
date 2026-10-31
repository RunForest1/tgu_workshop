# -*- coding: utf-8 -*-
"""Применение preprocessing_template.py: fit на train, transform на validation/test."""
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, "/home/claude/work")
from preprocessing_template import make_preprocessor

OUT = Path("/home/claude/work/features_out")
features = pd.read_csv(OUT / "features.csv", sep=";", encoding="utf-8-sig", parse_dates=["snapshot_at"])
split = pd.read_csv(OUT / "split.csv", sep=";", encoding="utf-8-sig", parse_dates=["snapshot_at"])
df = features.merge(split, on=["customer_id", "snapshot_at"])

NUMERIC = ["tenure_days", "debt_prev_month", "payment_delay_days_prev_month", "payment_ratio_prev_month",
           "avg_debt_last3", "months_with_debt_last3", "traffic_prev_month", "traffic_avg_last3",
           "traffic_delta_prev2", "traffic_per_active_day_prev_month", "support_count_prev_month",
           "support_count_last3", "digital_sessions_prev_month", "digital_payment_failure_rate_prev_month",
           "digital_selfservice_actions_prev_month"]
CATEGORICAL = ["region", "customer_segment", "acquisition_channel", "tariff_prev_month", "digital_status_prev_month"]

X_cols = NUMERIC + CATEGORICAL
print("Список столбцов X (", len(X_cols), "):", X_cols)

prep = make_preprocessor(NUMERIC, CATEGORICAL)

train_df = df[df.split == "train"].reset_index(drop=True)
val_df = df[df.split == "validation"].reset_index(drop=True)
test_df = df[df.split == "test"].reset_index(drop=True)

X_train = prep.fit_transform(train_df[X_cols])
X_val = prep.transform(val_df[X_cols])
X_test = prep.transform(test_df[X_cols])

print("\nX_train:", X_train.shape, " ключи:", train_df[["customer_id", "snapshot_at"]].shape)
print("X_validation:", X_val.shape, " ключи:", val_df[["customer_id", "snapshot_at"]].shape)
print("X_test:", X_test.shape, " ключи:", test_df[["customer_id", "snapshot_at"]].shape)

feature_names = prep.get_feature_names_out()
print("\nЧисло столбцов после преобразования:", len(feature_names))

# новая категория на validation, которой не было на train (проверка transform без fit)
val_regions = set(val_df.region.unique())
train_regions = set(train_df.region.unique())
print("\nРегионы validation, отсутствующие в train:", val_regions - train_regions)

# сохраняем список столбцов и размеры — не обязаны сохранять саму матрицу в CSV
with open(OUT / "X_columns_and_shapes.txt", "w", encoding="utf-8") as f:
    f.write("Числовые столбцы:\n  " + "\n  ".join(NUMERIC) + "\n\n")
    f.write("Категориальные столбцы:\n  " + "\n  ".join(CATEGORICAL) + "\n\n")
    f.write(f"X_train shape: {X_train.shape}\n")
    f.write(f"X_validation shape: {X_val.shape}\n")
    f.write(f"X_test shape: {X_test.shape}\n")
    f.write(f"Число столбцов после OneHot/масштабирования: {len(feature_names)}\n")
    f.write("Порядок строк в X совпадает с порядком строк train_df/val_df/test_df "
            "(соответствующие ключи customer_id, snapshot_at сохранены в том же порядке в split.csv "
            "после фильтрации по split и сброса индекса).\n")
print("\nСохранено в X_columns_and_shapes.txt")
