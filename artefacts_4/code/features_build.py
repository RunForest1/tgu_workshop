# -*- coding: utf-8 -*-
"""
Этап 4: формирование признаков и выбор метода.
Горизонт/активная база/разбиение — из устава команды и решения, принятого на этапе 3
(горизонт 30 дней, activation <= T, termination > T; сетка T и отложенные периоды
сдвинуты на месяц раньше примера из задания — обоснование в config.json и отчёте).
"""
from pathlib import Path
from datetime import timedelta
import json
import pandas as pd
import numpy as np

S2 = Path("/home/claude/work/out")
S3 = Path("/home/claude/work/eda_out")
S4_DATA = Path("/home/claude/runforest_fork/4_features/data")
OUT = Path("/home/claude/work/features_out")
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "horizon": "30_days",
    "active_mode": "after_events",  # activation <= T; нет termination или termination > T (дословно по уставу)
    "window_months": 3,
    "train_start": "2025-10-01",
    "train_end_inclusive": "2026-06-01",
    "validation_dates": ["2026-07-01"],
    "test_dates": ["2026-08-01"],
    "observation_end_exclusive": "2026-10-01",
    "campaign_fraction": 0.1,
    "deviation_note": (
        "Пример из задания (train дек.2025-июл.2026, validation авг.2026, test сен.2026) "
        "рассчитан на горизонт \"календарный месяц\". Устав команды фиксирует горизонт ровно "
        "30 дней: T=2026-09-01 потребовал бы данных о расторжениях вплоть до 2026-10-01 "
        "включительно, а журнал событий полон только строго до этой даты. Решение принято "
        "на этапе 3 (см. artefacts_3) и сохраняется здесь без изменений в середине проекта."
    ),
}
with open(OUT / "config.json", "w", encoding="utf-8") as f:
    json.dump(CONFIG, f, ensure_ascii=False, indent=2)

EVENTS_CUTOFF = pd.Timestamp(CONFIG["observation_end_exclusive"])
HORIZON_DAYS = 30

# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------
subs = pd.read_csv(S2 / "subscribers_clean.csv", sep=";", encoding="utf-8-sig", parse_dates=["activation_date"])
events = pd.read_csv(S2 / "contract_events_clean.csv", sep=";", encoding="utf-8-sig", parse_dates=["event_date"])
cm = pd.read_csv(S2 / "customer_month.csv", sep=";", encoding="utf-8-sig", parse_dates=["billing_month"])
cm["billing_available_at"] = pd.to_datetime(cm["billing_available_at"], utc=True).dt.tz_convert("Etc/GMT-3").dt.tz_localize(None)

digital = pd.read_csv(S4_DATA / "digital_usage_monthly.csv", sep=";", encoding="utf-8-sig", parse_dates=["usage_month"])
digital["available_at"] = pd.to_datetime(digital["available_at"], utc=True).dt.tz_convert("Etc/GMT-3").dt.tz_localize(None)

# --- входные проверки нового источника (из задания и стартового ноутбука) ---
assert not digital.duplicated(["customer_id", "usage_month"]).any(), "дублирующийся ключ digital_usage_monthly"
assert set(digital.customer_id) <= set(subs.customer_id), "digital ссылается на неизвестных абонентов"
assert set(digital.observation_status) <= {"observed", "not_enabled", "export_gap"}, "неизвестный статус наблюдения"
billing_pairs = set(zip(cm.customer_id, cm.billing_month))
digital_pairs = set(zip(digital.customer_id, digital.usage_month))
assert digital_pairs == billing_pairs, "набор пар (customer_id, месяц) в digital не совпадает с биллингом"
att = pd.to_numeric(digital.loc[digital.observation_status == "observed", "payment_attempts"])
fail = pd.to_numeric(digital.loc[digital.observation_status == "observed", "payment_failures"])
assert ((fail >= 0) & (fail <= att)).all(), "payment_failures вне диапазона [0, payment_attempts]"
print("Входные проверки digital_usage_monthly пройдены.")

termination = events[events.event_type == "termination"][["customer_id", "event_date"]].rename(columns={"event_date": "termination_date"})
assert not termination.customer_id.duplicated().any()
term_map = termination.set_index("customer_id")["termination_date"]

cm_idx = cm.set_index(["customer_id", "billing_month"])
digital_idx = digital.set_index(["customer_id", "usage_month"])

T_GRID = pd.date_range(CONFIG["train_start"], CONFIG["test_dates"][-1], freq="MS")
TRAIN_T = pd.date_range(CONFIG["train_start"], CONFIG["train_end_inclusive"], freq="MS")
VAL_T = [pd.Timestamp(d) for d in CONFIG["validation_dates"]]
TEST_T = [pd.Timestamp(d) for d in CONFIG["test_dates"]]
print("Сетка T:", list(T_GRID.date))
print("train:", len(TRAIN_T), "validation:", VAL_T, "test:", TEST_T)

# ---------------------------------------------------------------------------
# Активная база, метки, служебные даты окна
# ---------------------------------------------------------------------------
def month_minus(T, k):
    return T - pd.offsets.MonthBegin(k)

base_rows = []
for T in T_GRID:
    active = subs[subs.activation_date <= T].copy()
    term_for_active = active["customer_id"].map(term_map)
    active = active[term_for_active.isna() | (term_for_active > T)]

    target_end = T + timedelta(days=HORIZON_DAYS)  # последняя дата окна (включительно)
    fully_observed = target_end <= (EVENTS_CUTOFF - timedelta(days=1))
    term_dates = active["customer_id"].map(term_map)
    churned = term_dates.notna() & (term_dates > T) & (term_dates <= target_end)

    snap = active[["customer_id", "activation_date", "region", "acquisition_channel", "customer_segment"]].copy()
    snap["snapshot_at"] = T
    snap["tenure_days"] = (T - snap["activation_date"]).dt.days
    snap["churn_target"] = churned.astype(int).values if fully_observed else pd.NA
    snap["target_end_exclusive"] = target_end + pd.Timedelta(days=1)
    base_rows.append(snap)

base = pd.concat(base_rows, ignore_index=True)
print("\nСтрок активной базы:", len(base), "уникальных абонентов:", base.customer_id.nunique())
assert base.churn_target.isna().sum() == 0, "в выбранной сетке T не должно быть неполных наблюдений"

split = pd.Series("не используется", index=base.index)
split[base.snapshot_at.isin(TRAIN_T)] = "train"
split[base.snapshot_at.isin(VAL_T)] = "validation"
split[base.snapshot_at.isin(TEST_T)] = "test"
base["split"] = split
assert (base.split != "не используется").all()
print(base.groupby("split").agg(n_rows=("customer_id", "size"), n_customers=("customer_id", "nunique")))

# ---------------------------------------------------------------------------
# Безопасная выборка одной строки billing/customer_month месяца m для абонента на дату T
# (доступность: billing_available_at <= T и месяц == m)
# ---------------------------------------------------------------------------
def get_cm_row(customer_id, month, T):
    key = (customer_id, month)
    if key not in cm_idx.index:
        return None
    row = cm_idx.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    if row["billing_available_at"] > T:
        return None
    return row

def get_digital_row(customer_id, month, T):
    key = (customer_id, month)
    if key not in digital_idx.index:
        return None
    row = digital_idx.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    if row["available_at"] > T:
        return None
    return row

feature_rows = []
audit_rows = []

for _, r in base.iterrows():
    cid, T = r.customer_id, r.snapshot_at
    m1, m2, m3 = month_minus(T, 1), month_minus(T, 2), month_minus(T, 3)

    b1 = get_cm_row(cid, m1, T)
    b2 = get_cm_row(cid, m2, T)
    b3 = get_cm_row(cid, m3, T)
    last3 = [b for b in (b1, b2, b3) if b is not None]

    feat = {"customer_id": cid, "snapshot_at": T}
    audit = {"customer_id": cid, "snapshot_at": T}

    # --- договор/сегмент ---
    feat["tenure_days"] = r.tenure_days
    feat["region"] = r.region
    feat["customer_segment"] = r.customer_segment
    feat["acquisition_channel"] = r.acquisition_channel

    # --- биллинг: предыдущий конкретный месяц ---
    feat["debt_prev_month"] = b1["closing_debt_rub"] if b1 is not None else np.nan
    feat["payment_delay_days_prev_month"] = b1["payment_delay_days"] if b1 is not None else np.nan
    feat["tariff_prev_month"] = b1["tariff"] if b1 is not None else np.nan

    # --- биллинг: нормированное отношение с обработкой нулевого знаменателя ---
    if b1 is not None and b1["charged_rub"] and b1["charged_rub"] > 0:
        feat["payment_ratio_prev_month"] = b1["paid_rub"] / b1["charged_rub"]
    else:
        feat["payment_ratio_prev_month"] = np.nan  # начислений не было или месяц недоступен — отношение неизвестно

    # --- биллинг: агрегация за 3 месяца (долг) ---
    debts3 = [b["closing_debt_rub"] for b in last3]
    feat["avg_debt_last3"] = float(np.mean(debts3)) if debts3 else np.nan
    feat["months_with_debt_last3"] = int(sum(1 for d in debts3 if d > 0)) if debts3 else np.nan
    audit["debt_last3_n_known"] = len(debts3)

    # --- история использования: предыдущий месяц, 3-месячная агрегация, разница, отношение ---
    feat["traffic_prev_month"] = b1["traffic_gb"] if b1 is not None else np.nan
    traffics3 = [b["traffic_gb"] for b in last3 if pd.notna(b["traffic_gb"])]
    feat["traffic_avg_last3"] = float(np.mean(traffics3)) if traffics3 else np.nan
    audit["traffic_last3_n_known"] = len(traffics3)

    t1 = b1["traffic_gb"] if b1 is not None else np.nan
    t2 = b2["traffic_gb"] if b2 is not None else np.nan
    feat["traffic_delta_prev2"] = (t1 - t2) if (pd.notna(t1) and pd.notna(t2)) else np.nan

    if b1 is not None and pd.notna(b1["traffic_gb"]) and b1["active_days"] and b1["active_days"] > 0:
        feat["traffic_per_active_day_prev_month"] = b1["traffic_gb"] / b1["active_days"]
    else:
        feat["traffic_per_active_day_prev_month"] = np.nan

    # --- поддержка: предыдущий месяц и сумма сохранённых месячных снимков за 3 месяца ---
    feat["support_count_prev_month"] = b1["support_count"] if b1 is not None else np.nan
    sup3 = [b["support_count"] for b in last3]
    feat["support_count_last3"] = float(np.sum(sup3)) if sup3 else np.nan
    audit["support_last3_n_known"] = len(sup3)

    # --- новый цифровой источник ---
    d1 = get_digital_row(cid, m1, T)
    if d1 is None:
        feat["digital_status_prev_month"] = "report_not_yet_available"
        feat["digital_sessions_prev_month"] = np.nan
        feat["digital_payment_failure_rate_prev_month"] = np.nan
        feat["digital_selfservice_actions_prev_month"] = np.nan
    else:
        feat["digital_status_prev_month"] = d1["observation_status"]
        if d1["observation_status"] == "observed":
            feat["digital_sessions_prev_month"] = d1["app_sessions"]
            feat["digital_selfservice_actions_prev_month"] = d1["selfservice_actions"]
            att_v, fail_v = d1["payment_attempts"], d1["payment_failures"]
            feat["digital_payment_failure_rate_prev_month"] = (fail_v / att_v) if att_v > 0 else np.nan
        else:
            feat["digital_sessions_prev_month"] = np.nan
            feat["digital_selfservice_actions_prev_month"] = np.nan
            feat["digital_payment_failure_rate_prev_month"] = np.nan
    audit["digital_available_flag"] = d1 is not None
    audit["digital_status_prev_month"] = feat["digital_status_prev_month"]

    # --- аудит доступности использованных источников ---
    avail_used = [b["billing_available_at"] for b in last3] + ([d1["available_at"]] if d1 is not None else [])
    audit["max_available_at_used"] = max(avail_used) if avail_used else pd.NaT
    audit["billing_prev_month_available"] = b1 is not None

    feature_rows.append(feat)
    audit_rows.append(audit)

features = pd.DataFrame(feature_rows)
audit = pd.DataFrame(audit_rows)
print("\nПостроено признаков:", features.shape[1] - 2, "столбцов (без ключей)")

# ---------------------------------------------------------------------------
# Сохранение четырёх слоёв
# ---------------------------------------------------------------------------
labels = base[["customer_id", "snapshot_at", "churn_target", "target_end_exclusive"]].copy()
split_tbl = base[["customer_id", "snapshot_at", "split"]].copy()

def w(df, name):
    df.to_csv(OUT / name, sep=";", encoding="utf-8-sig", index=False)

w(features, "features.csv")
w(labels, "labels.csv")
w(split_tbl, "split.csv")
w(audit, "feature_audit.csv")

print("\nГотово:")
print(" features:", features.shape)
print(" labels:", labels.shape, "положительных меток:", int(labels.churn_target.sum()))
print(" split:", split_tbl.shape)
print(" audit:", audit.shape)
