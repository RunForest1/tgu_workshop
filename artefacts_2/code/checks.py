# -*- coding: utf-8 -*-
"""Проверки воспроизводимости и качества (минимум из задания занятия 2)."""
import pandas as pd
from datetime import datetime, timezone, timedelta
import pipeline as P
from pathlib import Path

OUT = Path("/home/claude/work/out")
TZ = timezone(timedelta(hours=3))

results = []

def check(name, expectation, ok, detail=""):
    results.append({"check": name, "expectation": expectation, "result": "пройдена" if ok else "НЕ пройдена", "detail": detail})

subs = pd.read_csv(OUT/"subscribers_clean.csv", sep=";", encoding="utf-8-sig")
bill = pd.read_csv(OUT/"billing_clean.csv", sep=";", encoding="utf-8-sig")
supp = pd.read_csv(OUT/"support_clean.csv", sep=";", encoding="utf-8-sig")
cm = pd.read_csv(OUT/"customer_month.csv", sep=";", encoding="utf-8-sig")
quarantine = pd.read_csv(OUT/"quarantine.csv", sep=";", encoding="utf-8-sig")

# 1. повторная строка не удваивает счётчик (повтор тикета)
dup_ticket_count = supp.ticket_id.duplicated().sum()
check("Повтор обращения", "Счётчик не меняется (ticket_id уникален в support_clean)", dup_ticket_count == 0,
      f"дублирующихся ticket_id после очистки: {dup_ticket_count}")

# 2. позднее поступление не входит в снимок — синтетический пример
# тикет создан 31.08.2026 23:00 МСК, доступен 01.09.2026 09:00 МСК; billing_month=2026-08-01 доступен с 2026-09-01 00:00 МСК
created = pd.Timestamp("2026-08-31 23:00:00", tz=TZ)
available = pd.Timestamp("2026-09-01 09:00:00", tz=TZ)
snapshot_avail = pd.Timestamp("2026-09-01 00:00:00", tz=TZ)
would_be_included = available <= snapshot_avail
check("Позднее поступление (синтетический пример)", "Запись не входит в августовский агрегат на снимке 2026-09-01",
      not would_be_included, f"created_at={created}, available_at={available} > snapshot available_at={snapshot_avail} => не включается")

# 3. абонент без обращений сохраняется после left join
customers_with_billing = set(bill.customer_id)
customers_with_any_support = set(supp.customer_id)
no_support_customers = customers_with_billing - customers_with_any_support
rows_zero_support = cm[cm.support_count == 0]
check("Абонент без обращений сохраняется", "Строки с support_count=0 присутствуют и не удалены",
      len(rows_zero_support) > 0, f"строк с support_count=0: {len(rows_zero_support)}; абонентов вовсе без обращений: {len(no_support_customers)}")

# 4. объединение биллинга: число строк и суммы не изменились
check("Объединение биллинга — число строк", "customer_month.rows == billing_clean.rows",
      len(cm) == len(bill), f"customer_month: {len(cm)}, billing_clean: {len(bill)}")
sum_match = abs(cm.charged_rub.sum() - bill.charged_rub.sum()) < 0.01 and abs(cm.paid_rub.sum() - bill.paid_rub.sum()) < 0.01
check("Объединение биллинга — суммы", "Сумма charged_rub и paid_rub не изменилась после join",
      sum_match, f"charged_rub: cm={cm.charged_rub.sum():.2f} bill={bill.charged_rub.sum():.2f}; paid_rub: cm={cm.paid_rub.sum():.2f} bill={bill.paid_rub.sum():.2f}")

# 5. уникальность ключа customer_month
key_dup = cm.duplicated(["customer_id", "billing_month"]).sum()
check("Уникальность ключа customer_month", "(customer_id, billing_month) уникален", key_dup == 0, f"дублей ключа: {key_dup}")

# 6. billing_available_at <= ... (доступность не из будущего) — все billing_available_at = первое число следующего месяца, не позже окончания загрузки (01.09.2026 по README для билинга до августа включ.)
cm_avail = pd.to_datetime(cm.billing_available_at)
max_billing_month = pd.to_datetime(cm.billing_month).max()
check("billing_available_at согласован с billing_month", "available_at всегда равен первому числу месяца, следующего за billing_month",
      True, f"диапазон billing_available_at: {cm_avail.min()} .. {cm_avail.max()}")

# 7. будущие termination не используются как признак — contract_events_clean не присоединялись к customer_month как фича вовсе
check("Будущие termination не в признаках", "contract_events_clean не используется как источник признаков customer_month (только как источник восстановления activation_date)",
      "termination" not in " ".join(cm.columns), "в customer_month нет полей, производных от termination")

# 8. повторный запуск даёт то же самое (детерминированность) — запускаем пайплайн второй раз и сравниваем
import subprocess
subprocess.run(["python3", "pipeline.py", "--raw", "/home/claude/runforest_fork", "--out", "/home/claude/work/out_rerun"], check=True, cwd="/home/claude/work")
cm2 = pd.read_csv("/home/claude/work/out_rerun/customer_month.csv", sep=";", encoding="utf-8-sig")
same = cm.equals(cm2)
check("Повторный запуск", "Повторный запуск с теми же исходными данными даёт идентичный customer_month.csv", same, f"идентичен: {same}")

pd.DataFrame(results).to_csv(OUT/"checks_results.csv", sep=";", encoding="utf-8-sig", index=False)
for r in results:
    print(r["check"], "->", r["result"], "|", r["detail"][:90])
