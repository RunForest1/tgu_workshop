# -*- coding: utf-8 -*-
"""Базовое правило: сортировка по долгу предыдущего месяца, K = floor(0.10 * N_active)."""
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("/home/claude/work/features_out")
feat = pd.read_csv(OUT / "features.csv", sep=";", encoding="utf-8-sig", parse_dates=["snapshot_at"])
labels = pd.read_csv(OUT / "labels.csv", sep=";", encoding="utf-8-sig", parse_dates=["snapshot_at"])
split = pd.read_csv(OUT / "split.csv", sep=";", encoding="utf-8-sig", parse_dates=["snapshot_at"])

df = feat.merge(labels, on=["customer_id", "snapshot_at"]).merge(split, on=["customer_id", "snapshot_at"])
train = df[df.split == "train"].copy()

rows = []
for T, g in train.groupby("snapshot_at"):
    g = g.copy()
    # неизвестный долг — после всех известных (не считаем неизвестное нулём); сортировка по убыванию долга, при равенстве — customer_id по возрастанию
    g["debt_known"] = g["debt_prev_month"].notna()
    g_sorted = g.sort_values(["debt_known", "debt_prev_month", "customer_id"], ascending=[False, False, True])
    N = len(g_sorted)
    K = int(np.floor(0.10 * N))
    selected = g_sorted.iloc[:K]
    n_pos_total = int(g.churn_target.sum())
    n_pos_selected = int(selected.churn_target.sum())
    recall = (n_pos_selected / n_pos_total) if n_pos_total > 0 else np.nan
    precision = (n_pos_selected / K) if K > 0 else np.nan
    rows.append({
        "snapshot_at": T, "N_active": N, "K": K,
        "n_pos_total": n_pos_total, "n_pos_selected_numerator": n_pos_selected,
        "recall_at_k": recall, "precision_at_k_denominator": K, "precision_at_k": precision,
    })

res = pd.DataFrame(rows)
res.to_csv(OUT / "baseline_train_metrics.csv", sep=";", encoding="utf-8-sig", index=False)
print(res.to_string(index=False))

valid_recall = res.dropna(subset=["recall_at_k"])
valid_precision = res.dropna(subset=["precision_at_k"])
macro_recall = valid_recall.recall_at_k.mean()
micro_recall = valid_recall.n_pos_selected_numerator.sum() / valid_recall.n_pos_total.sum()
macro_precision = valid_precision.precision_at_k.mean()
micro_precision = valid_precision.n_pos_selected_numerator.sum() / valid_precision.precision_at_k_denominator.sum()
print(f"\nМакро-среднее Recall@K (среднее по датам T): {macro_recall:.4f}")
print(f"Микро-среднее Recall@K (сумма числителей / сумма знаменателей): {micro_recall:.4f}")
print(f"Макро-среднее Precision@K: {macro_precision:.4f}")
print(f"Микро-среднее Precision@K: {micro_precision:.4f}")

with open(OUT / "baseline_summary.txt", "w", encoding="utf-8") as fh:
    fh.write(f"Макро-среднее Recall@K (среднее по датам T): {macro_recall:.4f}\n")
    fh.write(f"Микро-среднее Recall@K (сумма числителей / сумма знаменателей): {micro_recall:.4f}\n")
    fh.write(f"Макро-среднее Precision@K: {macro_precision:.4f}\n")
    fh.write(f"Микро-среднее Precision@K: {micro_precision:.4f}\n")
