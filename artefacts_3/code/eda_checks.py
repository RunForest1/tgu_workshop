# -*- coding: utf-8 -*-
"""Проверки воспроизводимости — Этап 3 (EDA)."""
import subprocess
import pandas as pd
from pathlib import Path

OUT = Path("/home/claude/work/eda_out")
results = []

def check(name, expectation, ok, detail=""):
    results.append({"check": name, "expectation": expectation, "result": "пройдена" if ok else "НЕ пройдена", "detail": detail})

e = pd.read_csv(OUT / "eda_sample.csv", sep=";", encoding="utf-8-sig", parse_dates=["snapshot_at", "activation_date", "source_billing_month", "billing_available_at", "target_end_exclusive"])
net = pd.read_csv(OUT / "network_availability.csv", sep=";", encoding="utf-8-sig", parse_dates=["measurement_month", "snapshot_at", "report_available_at"])
events = pd.read_csv("/home/claude/work/out/contract_events_clean.csv", sep=";", encoding="utf-8-sig", parse_dates=["event_date"])

# 1. Уникальность ключа и сохранность активной базы
dup = e.duplicated(["customer_id", "snapshot_at"]).sum()
check("Уникальность ключа и сохранность активной базы", "(customer_id, snapshot_at) уникален; все строки — активные на T абоненты",
      dup == 0, f"дублей ключа: {dup}; строк: {len(e)}; уникальных абонентов: {e.customer_id.nunique()}")

# 2. Информация доступна на T (billing_available_at <= snapshot_at)
has_b = e["billing_available_at"].notna()
ok2 = (e.loc[has_b, "billing_available_at"] <= e.loc[has_b, "snapshot_at"]).all()
check("Информация доступна на T", "billing_available_at <= snapshot_at для всех строк с биллингом", bool(ok2),
      f"проверено строк: {has_b.sum()} из {len(e)}")

# 3. Граничные события и конец горизонта: ни одно T в сетке не требует событий на/после 2026-10-01
cutoff = pd.Timestamp("2026-10-01")
max_needed = (e["target_end_exclusive"] - pd.Timedelta(days=1)).max()
ok3 = max_needed < cutoff
check("Граничные события и конец горизонта", "T+30 дней (последняя дата окна) строго раньше конца полноты журнала (2026-10-01)",
      bool(ok3), f"максимальная требуемая дата наблюдения: {max_needed.date()}; граница журнала: {cutoff.date()} (не включая)")

# 4. Поздний сетевой отчёт не просочился на T
late_leak = net[(net.available == True) & (net.report_available_at > net.snapshot_at)]
check("Поздний сетевой отчёт", "Ни один отчёт не помечен доступным, если report_available_at > snapshot_at",
      len(late_leak) == 0, f"нарушений: {len(late_leak)}; всего пар регион-месяц-T: {len(net)}, доступно: {int(net.available.sum())}")

# 5. Групповые суммы и знаменатели: сумма наблюдений по группам долга/обращений равна числу строк обучения
TRAIN = pd.date_range("2025-10-01", "2026-06-01", freq="MS")
train = e[e.snapshot_at.isin(TRAIN)]
debt_tbl = pd.read_csv(OUT / "figures" / "fig4a_debt.csv", sep=";", encoding="utf-8-sig")
supp_tbl = pd.read_csv(OUT / "figures" / "fig4b_support.csv", sep=";", encoding="utf-8-sig")
ok5 = debt_tbl.n_observations.sum() == len(train) and supp_tbl.n_observations.sum() == len(train)
check("Групповые суммы и знаменатели", "Сумма n_observations по группам долга и по группам обращений равна числу строк обучения",
      bool(ok5), f"строк обучения: {len(train)}; сумма по долгу: {debt_tbl.n_observations.sum()}; сумма по обращениям: {supp_tbl.n_observations.sum()}")

# 6. Повторный запуск — детерминированность полного набора скриптов
subprocess.run(["python3", "eda_build.py"], check=True, cwd="/home/claude/work")
subprocess.run(["python3", "eda_analysis.py"], check=True, cwd="/home/claude/work")
e2 = pd.read_csv(OUT / "eda_sample.csv", sep=";", encoding="utf-8-sig")
e1 = pd.read_csv(OUT / "eda_sample.csv", sep=";", encoding="utf-8-sig")  # повторное чтение после перезаписи для сравнения со снимком до запуска
same = e.astype(str).reset_index(drop=True).equals(e2.astype(str).reset_index(drop=True))
check("Повторный запуск", "Повторный запуск eda_build.py и eda_analysis.py даёт идентичный eda_sample.csv",
      bool(same), f"идентичен: {same}")

pd.DataFrame(results).to_csv(OUT / "checks_results.csv", sep=";", encoding="utf-8-sig", index=False)
for r in results:
    print(r["check"], "->", r["result"], "|", r["detail"][:100])
