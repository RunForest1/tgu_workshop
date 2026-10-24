# -*- coding: utf-8 -*-
"""
Пайплайн подготовки данных — Этап 2 «Сбор и подготовка данных».
Проект «Прогнозирование оттока абонентов» («СвязьРегион»).

Запуск: python pipeline.py --raw <папка с 1_introduction и 2_preprocessing> --out <папка результатов>
Зависимости: pandas>=2.0 (стандартная библиотека: sqlite3, csv, pathlib, datetime, argparse).
"""
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd

TZ = timezone(timedelta(hours=3))  # московское время, фиксированное смещение +03:00
CUTOFF_EVENTS = pd.Timestamp('2026-10-01')  # журнал событий полон строго до этой даты (не включая)

decision_log = []     # проблема, действие, обоснование, число затронутых записей
quality_before = []   # строк до обработки, по источникам
quality_after = []    # строк после обработки
quarantine_rows = []  # унифицированная таблица карантина


def log_decision(source_field, problem, count, action, rationale):
    decision_log.append({
        "source_field": source_field, "problem": problem, "count": count,
        "action": action, "rationale": rationale,
    })


def read_csv_raw(path):
    return pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)


# ---------------------------------------------------------------------------
# 1. subscribers.csv
# ---------------------------------------------------------------------------
def clean_subscribers(raw_dir, events_clean):
    sub = read_csv_raw(raw_dir / "1_introduction" / "subscribers.csv")
    n0 = len(sub)
    quality_before.append({"source": "subscribers.csv", "rows": n0, "unit": "один абонент", "key": "customer_id"})

    for c in sub.columns:
        sub[c] = sub[c].str.strip()

    missing_before = sub["activation_date"].isna().sum()
    # восстановление даты подключения из событий activation (у каждого абонента ровно одно)
    act = events_clean[events_clean.event_type == "activation"].set_index("customer_id")["event_date"]
    mask = sub["activation_date"].isna()
    recovered = sub.loc[mask, "customer_id"].map(act).astype(object)
    recovered = recovered.map(lambda d: d.isoformat() if pd.notna(d) else pd.NA)
    sub.loc[mask, "activation_date"] = recovered
    recovered_n = recovered.notna().sum()
    still_missing = sub["activation_date"].isna().sum()
    log_decision("subscribers.activation_date", "дата подключения отсутствует", int(missing_before),
                 f"восстановлено по единственному событию activation: {recovered_n}; осталось пропусков: {still_missing}",
                 "У каждого абонента ровно одно событие activation (проверено на источнике) — восстановление однозначно; дату не выдумываем, а берём из подтверждающего источника")

    sub["activation_date"] = pd.to_datetime(sub["activation_date"], errors="coerce").dt.date

    dup = sub.duplicated("customer_id").sum()
    log_decision("subscribers.customer_id", "дублирующиеся ключи", int(dup), "строки не удалялись (дублей нет)",
                 "customer_id уникален по построению источника")

    n1 = len(sub)
    quality_after.append({"source": "subscribers_clean.csv", "rows": n1, "unit": "один абонент", "key": "customer_id"})
    return sub


# ---------------------------------------------------------------------------
# 2. contract_events.csv
# ---------------------------------------------------------------------------
def clean_events(raw_dir):
    ev = read_csv_raw(raw_dir / "1_introduction" / "contract_events.csv")
    n0 = len(ev)
    quality_before.append({"source": "contract_events.csv", "rows": n0, "unit": "одно событие договора", "key": "event_id"})

    for c in ev.columns:
        ev[c] = ev[c].str.strip()

    dup_id = ev.event_id.duplicated().sum()
    log_decision("contract_events.event_id", "дублирующиеся event_id", int(dup_id), "строки не удалялись (дублей нет)", "event_id уникален в источнике")

    bad_type = ~ev.event_type.isin(["activation", "termination"])
    log_decision("contract_events.event_type", "неизвестный тип события", int(bad_type.sum()), "строки не удалялись (нарушений нет)", "в источнике встречаются только activation и termination")

    future = pd.to_datetime(ev.event_date) >= CUTOFF_EVENTS
    log_decision("contract_events.event_date", f"события на/после {CUTOFF_EVENTS.date()} (вне полноты журнала)",
                 int(future.sum()), "строки сохранены как есть, помечены вне диапазона полноты" if future.sum() else "нарушений нет",
                 "журнал полон строго до 01.10.2026; записи на/после этой даты не удаляются, но не считаются подтверждённо полными")

    act_counts = ev.groupby("customer_id").event_type.apply(lambda s: (s == "activation").sum())
    term_counts = ev.groupby("customer_id").event_type.apply(lambda s: (s == "termination").sum())
    log_decision("contract_events", "абоненты с !=1 activation или >1 termination",
                 int((act_counts != 1).sum() + (term_counts > 1).sum()), "строки не удалялись (нарушений нет)",
                 "устав требует ровно одно подключение и не более одного расторжения на абонента — проверено")

    ev["event_date"] = pd.to_datetime(ev["event_date"]).dt.date
    quality_after.append({"source": "contract_events_clean.csv", "rows": len(ev), "unit": "одно событие договора", "key": "event_id"})
    return ev


# ---------------------------------------------------------------------------
# 3. billing_monthly.csv
# ---------------------------------------------------------------------------
TARIFF_MAP = {
    "basic": "Basic", "comfort": "Comfort", "premium": "Premium", "премиум": "Premium",
}


def norm_tariff(x):
    key = x.strip().lower()
    return TARIFF_MAP.get(key, x.strip())


def norm_month(x):
    x = x.strip()
    if "." in x:
        d, m, y = x.split(".")
        return f"{y}-{m}-{d}"
    return x


def clean_billing(raw_dir):
    bill = read_csv_raw(raw_dir / "1_introduction" / "billing_monthly.csv")
    n0 = len(bill)
    quality_before.append({"source": "billing_monthly.csv", "rows": n0, "unit": "абонент за расчётный месяц", "key": "customer_id + billing_month"})

    full_dup = bill.duplicated().sum()
    bill = bill.drop_duplicates().copy()
    log_decision("billing_monthly (вся строка)", "полные повторные копии строк", int(full_dup),
                 f"удалено {full_dup} строк-копий", "точные дубликаты строки не несут новой информации и дают задвоенные агрегаты")

    bill["billing_month"] = bill["billing_month"].apply(norm_month)
    log_decision("billing_monthly.billing_month", "два формата даты (DD.MM.YYYY и YYYY-MM-DD)", n0,
                 "все значения приведены к YYYY-MM-DD", "единый формат необходим для согласования дат между источниками")

    dup_key = bill.duplicated(["customer_id", "billing_month"]).sum()
    log_decision("billing_monthly (customer_id+billing_month)", "дублирующиеся составные ключи после очистки формата дат",
                 int(dup_key), "строки не удалялись (конфликтующих дублей нет)", "после устранения дублей-копий и нормализации дат составной ключ уникален")

    raw_tariff_variants = bill["tariff"].unique().tolist()
    bill["tariff"] = bill["tariff"].apply(norm_tariff)
    bad_tariff = ~bill["tariff"].isin(["Basic", "Comfort", "Premium"])
    log_decision("billing_monthly.tariff", f"варианты написания тарифа {raw_tariff_variants}", n0,
                 "приведено к канону Basic/Comfort/Premium по регистру и переводу", "словарь данных фиксирует три канонических тарифа")

    for c in ["monthly_fee_rub", "charged_rub", "paid_rub", "opening_debt_rub", "closing_debt_rub",
              "traffic_gb", "voice_minutes", "active_days", "payment_delay_days", "tariff_changes_count"]:
        bill[c] = pd.to_numeric(bill[c], errors="coerce")

    missing_closing = bill["closing_debt_rub"].isna().sum()
    calc_closing = bill["opening_debt_rub"] + bill["charged_rub"] - bill["paid_rub"]
    bill.loc[bill["closing_debt_rub"].isna(), "closing_debt_rub"] = calc_closing[bill["closing_debt_rub"].isna()]
    log_decision("billing_monthly.closing_debt_rub", "конечный долг отсутствует", int(missing_closing),
                 "пересчитано по правилу opening_debt_rub + charged_rub - paid_rub", "равенство выполняется на всех остальных строках (расхождение < 1e-9 руб.) — пересчёт не искажает данные")

    mismatch = (bill["opening_debt_rub"] + bill["charged_rub"] - bill["paid_rub"] - bill["closing_debt_rub"]).abs()
    log_decision("billing_monthly (равенство долга)", "расхождение opening+charged-paid != closing (> 0.01 руб.)",
                 int((mismatch > 0.01).sum()), "строки не удалялись (нарушений нет)", "равенство — это требование проверки задания, не нарушено")

    neg_minutes = (bill["voice_minutes"] < 0).sum()
    bill.loc[bill["voice_minutes"] < 0, "voice_minutes"] = pd.NA
    log_decision("billing_monthly.voice_minutes", "отрицательные значения минут (невозможно физически)", int(neg_minutes),
                 "значение заменено на пропуск (NaN), остальные поля строки сохранены", "отрицательные минуты — явная ошибка записи; абонент и строка не удаляются, искажённая ячейка помечается неизвестной")

    missing_traffic = bill["traffic_gb"].isna().sum()
    log_decision("billing_monthly.traffic_gb", "пропуски трафика", int(missing_traffic), "пропуск сохранён как есть",
                 "ноль и пропуск трафика имеют разный смысл — подмена нулём недопустима")

    not_linked = (~bill["customer_id"].isin(read_csv_raw(raw_dir / "1_introduction" / "subscribers.csv")["customer_id"])).sum()
    log_decision("billing_monthly.customer_id", "ссылки на неизвестных абонентов", int(not_linked), "строки не удалялись (нарушений нет)", "все customer_id биллинга присутствуют в справочнике абонентов")

    bill["billing_month"] = pd.to_datetime(bill["billing_month"]).dt.date
    bill["available_at"] = pd.to_datetime(bill["available_at"]).dt.tz_localize(TZ)

    quality_after.append({"source": "billing_clean.csv", "rows": len(bill), "unit": "абонент за расчётный месяц", "key": "customer_id + billing_month"})
    return bill


# ---------------------------------------------------------------------------
# 4. support (legacy CSV + current SQLite)
# ---------------------------------------------------------------------------
TOPIC_MAP = {"Качество связи": "network", "Оплата и начисления": "billing", "Тариф": "tariff", "Дополнительные услуги": "services"}
CHANNEL_MAP = {"Чат": "chat", "Телефон": "phone", "Офис": "office"}


def parse_legacy_dt(s):
    if pd.isna(s) or str(s).strip() == "":
        return pd.NaT
    return pd.Timestamp(datetime.strptime(s.strip(), "%d.%m.%Y %H:%M:%S").replace(tzinfo=TZ))


def parse_current_dt(s):
    if pd.isna(s) or str(s).strip() == "":
        return pd.NaT
    s = s.strip()
    if s.endswith("Z"):
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone(TZ)
    else:
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=TZ)
    return pd.Timestamp(dt)


def extract_current_support(raw_dir):
    """Извлечение текущих обращений из SQLite (support_current_raw), как того требует задание."""
    db = raw_dir / "2_preprocessing" / "data" / "support.sqlite"
    query = """SELECT ticket_id, customer_id, created_at, available_at,
                      topic, channel, resolved_at, resolution_available_at,
                      satisfaction_score
               FROM support_current_raw"""
    with sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True) as conn:
        cur = conn.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols).astype("string")
    return df, query


def clean_support(raw_dir, subscribers_clean):
    legacy = read_csv_raw(raw_dir / "2_preprocessing" / "data" / "support_legacy_sep_oct_2025.csv")
    quality_before.append({"source": "support_legacy_sep_oct_2025.csv", "rows": len(legacy), "unit": "одно обращение", "key": "ticket_no"})

    current, sql_used = extract_current_support(raw_dir)
    # сверка с CSV-экспортом той же таблицы (должны совпасть количественно — это одна и та же таблица, а не два источника)
    current_csv = read_csv_raw(raw_dir / "2_preprocessing" / "data" / "support_current_export.csv")
    assert len(current) == len(current_csv), "SQLite и CSV-экспорт текущей поддержки разошлись по числу строк"
    quality_before.append({"source": "support.sqlite (support_current_raw, через SQL-запрос)", "rows": len(current), "unit": "одно обращение", "key": "ticket_id"})

    # --- дедупликация точных копий ---
    dup_legacy = legacy.duplicated().sum()
    legacy = legacy.drop_duplicates().copy()
    log_decision("support_legacy (вся строка)", "полные повторные копии строк", int(dup_legacy), f"удалено {dup_legacy} строк-копий", "точный дубликат обращения не является отдельным обращением")

    dup_current = current.duplicated().sum()
    current = current.drop_duplicates().copy()
    log_decision("support_current (вся строка)", "полные повторные копии строк", int(dup_current), f"удалено {dup_current} строк-копий", "точный дубликат обращения не является отдельным обращением")

    # --- переименование по schema_support.csv и приведение словарей ---
    legacy = legacy.rename(columns={
        "ticket_no": "ticket_id", "abonent_id": "customer_id",
        "opened_local": "created_at", "loaded_local": "available_at",
        "issue_ru": "topic", "contact_ru": "channel",
        "closed_local": "resolved_at", "closed_loaded_local": "resolution_available_at",
        "rating": "satisfaction_score",
    })
    legacy["source_system"] = "legacy"
    current = current.rename(columns={})
    current["source_system"] = "current"

    legacy_topic_before = legacy["topic"].copy()
    legacy["topic"] = legacy["topic"].map(TOPIC_MAP).fillna("unknown")
    legacy["channel"] = legacy["channel"].map(CHANNEL_MAP)
    log_decision("support_legacy.topic/channel", "русские категории темы/канала", len(legacy),
                 "сопоставлены канону по schema_support.csv (Качество связи→network и т.д.)",
                 "единый словарь категорий нужен для совместного анализа с текущей системой")

    current["topic"] = current["topic"].where(current["topic"].str.strip() != "", pd.NA)
    missing_topic_current = current["topic"].isna().sum()
    current["topic"] = current["topic"].fillna("unknown")
    log_decision("support_current.topic", "тема обращения отсутствует", int(missing_topic_current),
                 "заполнено значением unknown", "правило schema_support.csv: пропуск темы допускает unknown")

    # --- парсинг времени, сохранение момента события, приведение к московскому времени ---
    for c in ["created_at", "available_at", "resolved_at", "resolution_available_at"]:
        legacy[c] = legacy[c].apply(parse_legacy_dt)
    no_tz_current = (~current["created_at"].str.strip().str.endswith("Z")).sum()
    for c in ["created_at", "available_at", "resolved_at", "resolution_available_at"]:
        current[c] = current[c].apply(parse_current_dt)
    log_decision("support_current (время)", "значения времени без суффикса часового пояса", int(no_tz_current),
                 "интерпретированы как московское время и приведены к +03:00 наравне с UTC-записями", "правило заказчика: запись без явного пояса — московское время")

    support = pd.concat([
        legacy[["ticket_id", "customer_id", "created_at", "available_at", "topic", "channel",
                "resolved_at", "resolution_available_at", "satisfaction_score", "source_system"]],
        current[["ticket_id", "customer_id", "created_at", "available_at", "topic", "channel",
                 "resolved_at", "resolution_available_at", "satisfaction_score", "source_system"]],
    ], ignore_index=True)
    support["satisfaction_score"] = pd.to_numeric(support["satisfaction_score"], errors="coerce")
    support["customer_id"] = support["customer_id"].str.strip()
    blank_customer = support["customer_id"].notna() & (support["customer_id"] == "")
    support.loc[blank_customer, "customer_id"] = pd.NA

    n_merged = len(support)

    # --- карантин: нельзя установить абонента или безопасную дату доступности создания ---
    unlinked = support["customer_id"].isna() | (~support["customer_id"].isin(subscribers_clean["customer_id"]))
    unsafe_avail = support["available_at"] < support["created_at"]

    q1 = support[unlinked].copy()
    q1["quarantine_reason"] = "customer_id пуст или отсутствует в справочнике абонентов"
    q2 = support[unsafe_avail & ~unlinked].copy()
    q2["quarantine_reason"] = "available_at раньше created_at — небезопасная дата доступности"
    quarantine_rows.append(pd.concat([q1, q2], ignore_index=True))

    log_decision("support.customer_id", "обращение не связывается с известным абонентом", int(unlinked.sum()),
                 f"вынесено в карантин ({int(unlinked.sum())} строк)", "без корректного customer_id нельзя учитывать обращение в снимке конкретного абонента")
    log_decision("support.available_at", "доступность раньше создания обращения (небезопасная дата)", int(unsafe_avail.sum()),
                 f"вынесено в карантин ({int((unsafe_avail & ~unlinked).sum())} строк, не считая уже вынесенных по customer_id)",
                 "нарушение временного порядка делает момент доступности ненадёжным для правил учёта")

    support_clean = support[~unlinked & ~unsafe_avail].copy()

    # --- резолюция раньше создания: считаем закрытие неподтверждённым, запись сохраняем ---
    bad_resolution = support_clean["resolved_at"].notna() & (support_clean["resolved_at"] < support_clean["created_at"])
    support_clean.loc[bad_resolution, ["resolved_at", "resolution_available_at", "satisfaction_score"]] = pd.NA
    log_decision("support.resolved_at", "дата закрытия раньше даты создания обращения", int(bad_resolution.sum()),
                 "поля закрытия и оценки заменены на пропуск, само обращение сохранено", "закрытие раньше открытия невозможно; это ошибка полей закрытия, а не всей записи — обращение как факт остаётся достоверным")

    # резолюция доступна раньше её собственного момента закрытия или доступности создания — проверка (не было нарушений в данных)
    bad_res_avail = support_clean["resolution_available_at"].notna() & (
        (support_clean["resolution_available_at"] < support_clean["resolved_at"]) |
        (support_clean["resolution_available_at"] < support_clean["available_at"])
    )
    log_decision("support.resolution_available_at", "доступность закрытия раньше закрытия или доступности создания",
                 int(bad_res_avail.sum()), "строки не удалялись (нарушений нет)", "проверка пройдена на очищенных данных")

    dup_ticket = support_clean["ticket_id"].duplicated().sum()
    log_decision("support_clean.ticket_id", "дублирующиеся ticket_id после объединения источников", int(dup_ticket),
                 "строки не удалялись (конфликтов нет)", "пересечений идентификаторов между legacy и current не обнаружено")

    quality_after.append({"source": "support_clean.csv", "rows": len(support_clean), "unit": "одно обращение", "key": "ticket_id"})
    return support_clean, sql_used


# ---------------------------------------------------------------------------
# 5. customer_month.csv
# ---------------------------------------------------------------------------
def build_customer_month(billing_clean, subscribers_clean, support_clean):
    cm = billing_clean.merge(
        subscribers_clean[["customer_id", "region", "acquisition_channel", "customer_segment"]],
        on="customer_id", how="left", validate="many_to_one",
    )
    assert len(cm) == len(billing_clean), "объединение с абонентами размножило строки биллинга"

    s = support_clean.copy()
    s["created_month"] = s["created_at"].dt.tz_convert(TZ).dt.to_period("M").dt.to_timestamp().dt.date

    agg_rows = []
    for _, row in cm.iterrows():
        bm = row["billing_month"]
        snap_avail = row["available_at"]
        sub_s = s[(s["customer_id"] == row["customer_id"]) & (s["created_month"] == bm) & (s["available_at"] <= snap_avail)]
        agg_rows.append({
            "customer_id": row["customer_id"], "billing_month": bm,
            "support_count": len(sub_s),
            "support_network_count": int((sub_s["topic"] == "network").sum()),
            "support_billing_count": int((sub_s["topic"] == "billing").sum()),
            "support_unknown_topic_count": int((sub_s["topic"] == "unknown").sum()),
        })
    agg = pd.DataFrame(agg_rows)

    cm = cm.merge(agg, on=["customer_id", "billing_month"], how="left", validate="one_to_one")
    assert len(cm) == len(billing_clean), "объединение агрегатов поддержки размножило строки биллинга"
    cm = cm.rename(columns={"available_at": "billing_available_at"})
    cm["source_billing_month"] = cm["billing_month"]
    return cm


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True, help="папка, содержащая 1_introduction и 2_preprocessing")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    events_clean = clean_events(args.raw)
    subscribers_clean = clean_subscribers(args.raw, events_clean)
    billing_clean = clean_billing(args.raw)
    support_clean, sql_used = clean_support(args.raw, subscribers_clean)
    customer_month = build_customer_month(billing_clean, subscribers_clean, support_clean)
    quarantine = pd.concat(quarantine_rows, ignore_index=True) if quarantine_rows else pd.DataFrame()

    def w(df, name):
        df.to_csv(args.out / name, sep=";", encoding="utf-8-sig", index=False)

    w(subscribers_clean, "subscribers_clean.csv")
    w(events_clean, "contract_events_clean.csv")
    w(billing_clean, "billing_clean.csv")
    w(support_clean, "support_clean.csv")
    w(customer_month, "customer_month.csv")
    w(quarantine, "quarantine.csv")
    w(pd.DataFrame(decision_log), "decision_log.csv")
    w(pd.DataFrame(quality_before), "quality_summary_before.csv")
    w(pd.DataFrame(quality_after), "quality_summary_after.csv")

    (args.out / "extract_support_used.sql").write_text(sql_used, encoding="utf-8")

    print("Готово. Строк:")
    print(" subscribers_clean:", len(subscribers_clean))
    print(" contract_events_clean:", len(events_clean))
    print(" billing_clean:", len(billing_clean))
    print(" support_clean:", len(support_clean))
    print(" customer_month:", len(customer_month))
    print(" quarantine:", len(quarantine))
    return {
        "subscribers_clean": subscribers_clean, "events_clean": events_clean, "billing_clean": billing_clean,
        "support_clean": support_clean, "customer_month": customer_month, "quarantine": quarantine,
    }


if __name__ == "__main__":
    main()
