# -*- coding: utf-8 -*-
"""
Построение исследовательской таблицы снимков eda_sample.csv — Этап 3 (EDA).
Правило активной базы и горизонт — из устава проекта (activation <= T; termination > T; горизонт 30 дней).
"""
from pathlib import Path
from datetime import timedelta
import pandas as pd

S2 = Path("/home/claude/work/out")              # результаты этапа 2
S3_DATA = Path("/home/claude/runforest_fork/3_eda/data")  # новые источники занятия 3
OUT = Path("/home/claude/work/eda_out")
OUT.mkdir(parents=True, exist_ok=True)

EVENTS_CUTOFF = pd.Timestamp("2026-10-01")  # журнал событий полон строго до этой даты (не включая)
HORIZON_DAYS = 30

# ---------------------------------------------------------------------------
# Загрузка этапа 2
# ---------------------------------------------------------------------------
subs = pd.read_csv(S2 / "subscribers_clean.csv", sep=";", encoding="utf-8-sig", parse_dates=["activation_date"])
events = pd.read_csv(S2 / "contract_events_clean.csv", sep=";", encoding="utf-8-sig", parse_dates=["event_date"])
cm = pd.read_csv(S2 / "customer_month.csv", sep=";", encoding="utf-8-sig", parse_dates=["billing_month"])
cm["billing_available_at"] = pd.to_datetime(cm["billing_available_at"], utc=True).dt.tz_convert("Etc/GMT-3").dt.tz_localize(None)

termination = events[events.event_type == "termination"][["customer_id", "event_date"]].rename(columns={"event_date": "termination_date"})
# по уставу и проверке этапа 2 — не более одного termination на абонента
assert not termination.customer_id.duplicated().any()
term_map = termination.set_index("customer_id")["termination_date"]

network = pd.read_csv(S3_DATA / "network_quality_monthly.csv", sep=";", encoding="utf-8-sig", parse_dates=["measurement_month"])
network["report_available_at"] = pd.to_datetime(network["report_available_at"], utc=True).dt.tz_convert("Etc/GMT-3").dt.tz_localize(None)

# ---------------------------------------------------------------------------
# Сетка дат T: 30-дневный горизонт (по уставу), первые числа месяцев
# Нижняя граница: нужен биллинг предыдущего месяца -> первый billing_month = 2025-09-01 -> T >= 2025-10-01
# Верхняя граница: T+30 должно укладываться в полноту журнала событий (строго до 2026-10-01)
#   -> T <= 2026-08-01 (2026-08-01 + 30 = 2026-08-31, последняя дата с полным 30-дневным окном)
# Предложенное в задании разбиение (дек2025-июл2026 / авг / сен) рассчитано на горизонт "календарный месяц"
# и не умещается в полноту журнала при горизонте 30 дней (T=2026-09-01 требует данные по 2026-10-01 включительно).
# Поэтому сетка и валидация/тест сдвинуты на один месяц раньше при сохранении той же логики: 2 последних месяца отложены.
T_GRID = pd.date_range("2025-10-01", "2026-08-01", freq="MS")
TRAIN_T = pd.date_range("2025-10-01", "2026-06-01", freq="MS")   # основной EDA
VAL_T = pd.Timestamp("2026-07-01")
TEST_T = pd.Timestamp("2026-08-01")

print("Сетка T:", len(T_GRID), "дат, с", T_GRID.min().date(), "по", T_GRID.max().date())
print("Обучение:", len(TRAIN_T), "дат; валидация:", VAL_T.date(), "; тест:", TEST_T.date())

# ---------------------------------------------------------------------------
# Активная база на T и churn_target
# ---------------------------------------------------------------------------
rows = []
for T in T_GRID:
    active = subs[subs.activation_date <= T].copy()
    term_for_active = active["customer_id"].map(term_map)
    active = active[term_for_active.isna() | (term_for_active > T)]

    target_end = T + timedelta(days=HORIZON_DAYS)
    fully_observed = target_end <= (EVENTS_CUTOFF - timedelta(days=1))  # нужны события вплоть до target_end включительно

    term_dates = active["customer_id"].map(term_map)
    churned_in_window = term_dates.notna() & (term_dates > T) & (term_dates <= target_end)

    prev_month = (T - pd.offsets.MonthBegin(1))
    snap = active[["customer_id", "activation_date", "region", "acquisition_channel", "customer_segment"]].copy()
    snap["snapshot_at"] = T
    snap["source_billing_month"] = prev_month
    snap["target_end_exclusive"] = target_end + pd.Timedelta(days=1)  # конец интервала исключён -> первая дата ВНЕ окна
    snap["tenure_days"] = (T - snap["activation_date"]).dt.days
    if fully_observed:
        snap["churn_target"] = churned_in_window.astype(int).values
    else:
        snap["churn_target"] = pd.NA
    rows.append(snap)

base = pd.concat(rows, ignore_index=True)
print("\nСтрок активной базы по всем T:", len(base), "; уникальных абонентов:", base.customer_id.nunique())
print("Без метки (неполное наблюдение):", base.churn_target.isna().sum())

# ---------------------------------------------------------------------------
# Признаки предыдущего месяца из customer_month (billing_available_at <= T гарантируется по построению)
# ---------------------------------------------------------------------------
feat_cols = ["customer_id", "billing_month", "billing_available_at", "tariff", "monthly_fee_rub",
             "charged_rub", "paid_rub", "opening_debt_rub", "closing_debt_rub", "traffic_gb",
             "voice_minutes", "active_days", "payment_delay_days", "tariff_changes_count",
             "support_count", "support_network_count", "support_billing_count", "support_unknown_topic_count"]
cm_feat = cm[feat_cols].rename(columns={"billing_month": "source_billing_month"})

eda = base.merge(cm_feat, on=["customer_id", "source_billing_month"], how="left")
assert len(eda) == len(base), "join с billing/customer_month размножил строки"

has_billing = eda["billing_available_at"].notna()
check_avail = (eda.loc[has_billing, "billing_available_at"] <= eda.loc[has_billing, "snapshot_at"]).all()
print("billing_available_at <= snapshot_at для всех строк с биллингом:", bool(check_avail))

missing_billing = (~has_billing).sum()
print("Строк без данных биллинга за предыдущий месяц (новые/с пропуском истории):", missing_billing)

eda.to_csv(OUT / "eda_sample.csv", sep=";", encoding="utf-8-sig", index=False)

# ---------------------------------------------------------------------------
# Доступность сетевых отчётов — отдельная безопасная таблица (region, measurement_month, snapshot_at)
# ---------------------------------------------------------------------------
regions = subs.region.unique()
avail_rows = []
for T in T_GRID:
    prev_month = (T - pd.offsets.MonthBegin(1))
    for r in regions:
        rec = network[(network.region == r) & (network.measurement_month == prev_month)]
        if len(rec) == 0:
            avail_rows.append({"region": r, "measurement_month": prev_month, "snapshot_at": T,
                                "available": False, "report_available_at": pd.NaT,
                                "probe_count": pd.NA, "failed_probe_count": pd.NA,
                                "download_mbps_p50": pd.NA, "download_mbps_p90": pd.NA})
            continue
        rec = rec.iloc[0]
        available = rec.report_available_at <= T
        avail_rows.append({"region": r, "measurement_month": prev_month, "snapshot_at": T,
                            "available": available, "report_available_at": rec.report_available_at,
                            "probe_count": rec.probe_count if available else pd.NA,
                            "failed_probe_count": rec.failed_probe_count if available else pd.NA,
                            "download_mbps_p50": rec.download_mbps_p50 if available else pd.NA,
                            "download_mbps_p90": rec.download_mbps_p90 if available else pd.NA})
network_avail = pd.DataFrame(avail_rows)
network_avail.to_csv(OUT / "network_availability.csv", sep=";", encoding="utf-8-sig", index=False)
print("\nСетевая доступность: всего", len(network_avail), "строк; доступно на T:", network_avail.available.sum())

print("\nГотово.")
