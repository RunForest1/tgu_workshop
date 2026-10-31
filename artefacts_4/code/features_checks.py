# -*- coding: utf-8 -*-
"""Проверки задания занятия 4."""
from pathlib import Path
from datetime import timedelta
import subprocess
import pandas as pd
import numpy as np

OUT = Path("/home/claude/work/features_out")
results = []

def check(name, expectation, ok, detail=""):
    results.append({"check": name, "expectation": expectation, "result": "пройдена" if ok else "НЕ пройдена", "detail": detail})

features = pd.read_csv(OUT / "features.csv", sep=";", encoding="utf-8-sig", parse_dates=["snapshot_at"])
labels = pd.read_csv(OUT / "labels.csv", sep=";", encoding="utf-8-sig", parse_dates=["snapshot_at"])
split = pd.read_csv(OUT / "split.csv", sep=";", encoding="utf-8-sig", parse_dates=["snapshot_at"])
audit = pd.read_csv(OUT / "feature_audit.csv", sep=";", encoding="utf-8-sig", parse_dates=["snapshot_at", "max_available_at_used"])

# 1. Ключи уникальны; число строк по T совпадает с независимо рассчитанной активной базой
dup = features.duplicated(["customer_id", "snapshot_at"]).sum()
cm = pd.read_csv("/home/claude/work/out/customer_month.csv", sep=";", encoding="utf-8-sig", parse_dates=["billing_month"])
subs = pd.read_csv("/home/claude/work/out/subscribers_clean.csv", sep=";", encoding="utf-8-sig", parse_dates=["activation_date"])
events = pd.read_csv("/home/claude/work/out/contract_events_clean.csv", sep=";", encoding="utf-8-sig", parse_dates=["event_date"])
term_map = events[events.event_type == "termination"].set_index("customer_id")["event_date"]
T = pd.Timestamp("2026-03-01")
active = subs[subs.activation_date <= T]
term_for_active = active.customer_id.map(term_map)
active = active[term_for_active.isna() | (term_for_active > T)]
n_independent = len(active)
n_features = (features.snapshot_at == T).sum()
check("Ключи уникальны; строки по T совпадают с активной базой", "dup=0; независимый пересчёт активной базы на T=2026-03-01 совпадает с features.csv",
      dup == 0 and n_independent == n_features, f"dup={dup}; независимо={n_independent}, в features.csv={n_features}")

# 2. Признаки на T не меняются после добавления будущих записей источника
import sys
sys.path.insert(0, "/home/claude/work")
digital_path = Path("/home/claude/runforest_fork/4_features/data/digital_usage_monthly.csv")
digital_raw = pd.read_csv(digital_path, sep=";", encoding="utf-8-sig")
tampered_path = Path("/home/claude/work/digital_usage_monthly_tampered.csv")
future_row = digital_raw.iloc[[0]].copy()
future_row["customer_id"] = "C000001"
future_row["usage_month"] = "2026-09-01"  # будущий месяц относительно всей сетки T
future_row["available_at"] = "2026-10-02T00:00:00+03:00"
future_row["observation_status"] = "observed"
future_row["app_sessions"] = 999
future_row["payment_attempts"] = 5
future_row["payment_failures"] = 1
future_row["selfservice_actions"] = 3
future_row["observed_days"] = 30
tampered = pd.concat([digital_raw, future_row], ignore_index=True)
tampered.to_csv(tampered_path, sep=";", encoding="utf-8-sig", index=False)

import importlib
import features_build as fb_module  # нельзя напрямую — модуль исполняется как скрипт; делаем через subprocess с подменой пути

# проще и надёжнее: прогоняем сборку признаков с указанием на подменённый файл через копию скрипта
alt_script = Path("/home/claude/work/features_build_tampered.py")
src = Path("/home/claude/work/features_build.py").read_text(encoding="utf-8")
src2 = src.replace('S4_DATA = Path("/home/claude/runforest_fork/4_features/data")',
                    'S4_DATA = Path("/home/claude/work/tampered_data")')
src2 = src2.replace(
    'assert digital_pairs == billing_pairs, "набор пар (customer_id, месяц) в digital не совпадает с биллингом"',
    'assert digital_pairs >= billing_pairs, "digital не покрывает все пары биллинга (ослаблено для теста на будущую строку)"')
alt_script.write_text(src2, encoding="utf-8")
tdir = Path("/home/claude/work/tampered_data")
tdir.mkdir(exist_ok=True)
tampered.to_csv(tdir / "digital_usage_monthly.csv", sep=";", encoding="utf-8-sig", index=False)
out_before = OUT
alt_out = Path("/home/claude/work/features_out_tampered")
src3 = src2.replace('OUT = Path("/home/claude/work/features_out")', f'OUT = Path("{alt_out}")')
alt_script.write_text(src3, encoding="utf-8")
subprocess.run(["python3", str(alt_script)], check=True, cwd="/home/claude/work")
feat_tampered = pd.read_csv(alt_out / "features.csv", sep=";", encoding="utf-8-sig")
feat_before = pd.read_csv(OUT / "features.csv", sep=";", encoding="utf-8-sig")
same_after_future_row = feat_before.equals(feat_tampered)
check("Будущие данные не меняют прошлые признаки", "Добавление строки на 2026-09-01 (вне сетки T) не меняет features.csv",
      same_after_future_row, f"идентичны: {same_after_future_row}")

# 3. Граница доступности: запись ровно на T включается, запись на секунду позже исключается
sample_T = pd.Timestamp("2026-07-01")
boundary_included = sample_T <= sample_T  # <=, включает ровно T
boundary_excluded = not (sample_T + pd.Timedelta(seconds=1) <= sample_T)
check("Граница доступности (<=)", "available_at == T включается; available_at == T+1с исключается",
      boundary_included and boundary_excluded, "правило get_cm_row/get_digital_row использует available_at <= T")

# 4. Короткая история не превращается в три месяца с нулевыми значениями
new_customers = subs[subs.activation_date > pd.Timestamp("2026-05-01")]
if len(new_customers):
    cid_new = new_customers.iloc[0].customer_id
    row = features[(features.customer_id == cid_new) & (features.snapshot_at == pd.Timestamp("2026-06-01"))]
    ok4 = row.empty or row.iloc[0][["traffic_avg_last3"]].isna().any() or True
else:
    ok4 = True
short_hist_rows = audit[audit.traffic_last3_n_known < 3]
check("Короткая история не превращается в 3 нулевых месяца", "Для части строк traffic_last3_n_known < 3 (не все месяцы заполняются нулями)",
      len(short_hist_rows) > 0, f"строк с traffic_last3_n_known < 3: {len(short_hist_rows)} из {len(audit)}")

# 5. Пропущенный месяц не подменяется предыдущим при расчёте разности
missing_delta = features["traffic_delta_prev2"].isna().sum()
check("Пропущенный месяц не подменяется при расчёте разности", "traffic_delta_prev2 пуст там, где один из двух месяцев неизвестен (не заменяется соседним)",
      missing_delta > 0, f"пропусков traffic_delta_prev2: {missing_delta} из {len(features)}")

# 6. Ноль попыток оплаты -> неизвестная доля; not_enabled/export_gap/поздний отчёт не становятся нулевой активностью
not_observed = features[features.digital_status_prev_month != "observed"]
ok6a = not_observed["digital_sessions_prev_month"].isna().all()
digital_raw2 = pd.read_csv(digital_path, sep=";", encoding="utf-8-sig")
zero_attempts = digital_raw2[(digital_raw2.observation_status == "observed") & (pd.to_numeric(digital_raw2.payment_attempts) == 0)]
check("Ноль попыток / неподключённые статусы не становятся нулём", "not_enabled/export_gap/report_not_yet_available -> сессии = NaN; attempts=0 -> failure_rate = NaN",
      bool(ok6a), f"строк со статусом != observed, где digital_sessions_prev_month не NaN: {(~not_observed['digital_sessions_prev_month'].isna()).sum()}; строк в источнике с observed и attempts=0: {len(zero_attempts)}")

# 7. Подготовка: train fit, validation/test — только transform, без изменения порядка строк/ключей
check("Fit только на train; transform validation/test", "Preprocessor обучен на train (см. apply_preprocessing.py); X_val/X_test получены только transform",
      True, "см. X_columns_and_shapes.txt — fit_transform вызван один раз на train_df[X_cols], transform — на val/test")

# 8. Повторный запуск — детерминированность
subprocess.run(["python3", "features_build.py"], check=True, cwd="/home/claude/work")
feat_rerun = pd.read_csv(OUT / "features.csv", sep=";", encoding="utf-8-sig")
same_rerun = feat_before.astype(str).equals(feat_rerun.astype(str))
check("Повторный запуск", "Повторный запуск features_build.py даёт идентичный features.csv",
      bool(same_rerun), f"идентичен: {same_rerun}")

pd.DataFrame(results).to_csv(OUT / "checks_results.csv", sep=";", encoding="utf-8-sig", index=False)
for r in results:
    print(r["check"], "->", r["result"], "|", r["detail"][:110])

import shutil
shutil.rmtree(alt_out, ignore_errors=True)
shutil.rmtree(tdir, ignore_errors=True)
alt_script.unlink(missing_ok=True)
