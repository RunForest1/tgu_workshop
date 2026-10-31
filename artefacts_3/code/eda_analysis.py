# -*- coding: utf-8 -*-
"""Расчётные таблицы и шесть обязательных рисунков для отчёта занятия 3."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/claude/work/eda_out")
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

e = pd.read_csv(OUT / "eda_sample.csv", sep=";", encoding="utf-8-sig", parse_dates=["snapshot_at", "source_billing_month"])
net_avail = pd.read_csv(OUT / "network_availability.csv", sep=";", encoding="utf-8-sig", parse_dates=["measurement_month", "snapshot_at"])

TRAIN = pd.date_range("2025-10-01", "2026-06-01", freq="MS")
train = e[e.snapshot_at.isin(TRAIN)].copy()
val = e[e.snapshot_at == pd.Timestamp("2026-07-01")].copy()
test = e[e.snapshot_at == pd.Timestamp("2026-08-01")].copy()
print("train/val/test rows:", len(train), len(val), len(test))

plt.rcParams.update({"figure.autolayout": True, "font.size": 10})

# ---------------------------------------------------------------------------
# Рисунок 1: доля пропусков исследуемых показателей (train)
# ---------------------------------------------------------------------------
cols_for_missing = ["closing_debt_rub", "traffic_gb", "voice_minutes", "support_count",
                     "payment_delay_days", "active_days", "tariff"]
miss = train[cols_for_missing].isna().mean().sort_values(ascending=False) * 100
miss_tbl = miss.reset_index()
miss_tbl.columns = ["field", "missing_pct"]
miss_tbl["n_rows"] = len(train)
miss_tbl.to_csv(FIG / "fig1_missing.csv", sep=";", encoding="utf-8-sig", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(miss_tbl.field, miss_tbl.missing_pct, color="#4C72B0")
ax.set_xlabel("Доля пропусков, %")
ax.set_title(f"Рис. 1. Доля пропусков по показателям (обучение, T=10.2025–06.2026, n={len(train)})")
for i, v in enumerate(miss_tbl.missing_pct):
    ax.text(v + 0.3, i, f"{v:.2f}%", va="center", fontsize=9)
ax.set_xlim(0, max(5, miss_tbl.missing_pct.max() * 1.4))
fig.savefig(FIG / "fig1_missing.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Рисунок 2: распределения трафика и долга (две панели)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
traf = train["traffic_gb"].dropna()
axes[0].hist(traf, bins=40, color="#55A868")
axes[0].set_title(f"Трафик, ГБ (известно n={len(traf)} из {len(train)})")
axes[0].set_xlabel("ГБ за месяц")
axes[0].axvline(traf.median(), color="black", linestyle="--", linewidth=1)
axes[0].text(traf.median(), axes[0].get_ylim()[1]*0.9, f"медиана={traf.median():.1f}", rotation=90, va="top")

debt = train["closing_debt_rub"]
axes[1].boxplot(debt, vert=True, showfliers=True)
axes[1].set_title(f"Долг на конец месяца, руб. (n={len(debt)})")
axes[1].set_ylabel("руб.")
q = debt.quantile([0.5, 0.9, 0.99])
axes[1].text(1.1, q[0.9], f"p50={q[0.5]:.0f}\np90={q[0.9]:.0f}\np99={q[0.99]:.0f}", va="center")
fig.suptitle("Рис. 2. Распределения трафика и долга (обучение)")
fig.savefig(FIG / "fig2_distributions.png", dpi=150)
plt.close(fig)

dist_tbl = pd.DataFrame({
    "показатель": ["traffic_gb", "closing_debt_rub"],
    "n_known": [traf.shape[0], debt.shape[0]],
    "n_missing": [train["traffic_gb"].isna().sum(), train["closing_debt_rub"].isna().sum()],
    "median": [traf.median(), debt.median()],
    "p10": [traf.quantile(.1), debt.quantile(.1)],
    "p90": [traf.quantile(.9), debt.quantile(.9)],
    "max": [traf.max(), debt.max()],
})
dist_tbl.to_csv(FIG / "fig2_distributions.csv", sep=";", encoding="utf-8-sig", index=False)

# ---------------------------------------------------------------------------
# Рисунок 3: динамика доли оттока и размера активной базы по T (train+val+test, помечено)
# ---------------------------------------------------------------------------
dyn = e.groupby("snapshot_at").agg(n_observations=("customer_id", "size"), n_churn=("churn_target", "sum")).reset_index()
dyn["churn_rate"] = dyn.n_churn / dyn.n_observations
dyn["split"] = np.where(dyn.snapshot_at.isin(TRAIN), "обучение", np.where(dyn.snapshot_at == pd.Timestamp("2026-07-01"), "валидация", "тест"))
dyn.to_csv(FIG / "fig3_dynamics.csv", sep=";", encoding="utf-8-sig", index=False)

fig, ax1 = plt.subplots(figsize=(9, 4.5))
colors = {"обучение": "#4C72B0", "валидация": "#DD8452", "тест": "#C44E52"}
ax1.bar(dyn.snapshot_at, dyn.n_observations, width=20, color=[colors[s] for s in dyn.split], alpha=0.35, label="размер активной базы (n)")
ax1.set_ylabel("Размер активной базы, n абонентов")
ax2 = ax1.twinx()
ax2.plot(dyn.snapshot_at, dyn.churn_rate * 100, marker="o", color="black", label="доля оттока, %")
for _, r in dyn.iterrows():
    ax2.annotate(f"{r.churn_rate*100:.1f}%", (r.snapshot_at, r.churn_rate*100), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
ax2.set_ylabel("Доля оттока за 30 дней, %")
ax1.set_title("Рис. 3. Доля оттока (30 дней) и размер активной базы по T\n(синий=обучение, оранжевый=валидация 07.2026, красный=тест 08.2026)")
fig.autofmt_xdate()
fig.savefig(FIG / "fig3_dynamics.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Рисунок 4: доля оттока по долгу и обращениям (две панели), train
# ---------------------------------------------------------------------------
def debt_group(x):
    if pd.isna(x):
        return "неизвестно"
    return "долг > 0" if x > 0 else "долг = 0"
train["debt_group"] = train["closing_debt_rub"].apply(debt_group)

def support_group(x):
    if pd.isna(x):
        return "неизвестно"
    if x == 0:
        return "0 обращений"
    if x == 1:
        return "1 обращение"
    return "2+ обращений"
train["support_group"] = train["support_count"].apply(support_group)

def agg_group(df, col, order):
    g = df.groupby(col, observed=True).agg(n_observations=("customer_id", "size"),
                                              n_customers=("customer_id", "nunique"),
                                              n_churn=("churn_target", "sum")).reindex(order)
    for c in ["n_observations", "n_customers", "n_churn"]:
        g[c] = g[c].fillna(0).astype(int)
    g["churn_rate"] = np.where(g.n_observations > 0, g.n_churn / g.n_observations, np.nan)
    return g.reset_index()

debt_tbl = agg_group(train, "debt_group", ["долг = 0", "долг > 0", "неизвестно"])
supp_tbl = agg_group(train, "support_group", ["0 обращений", "1 обращение", "2+ обращений", "неизвестно"])
debt_tbl.to_csv(FIG / "fig4a_debt.csv", sep=";", encoding="utf-8-sig", index=False)
supp_tbl.to_csv(FIG / "fig4b_support.csv", sep=";", encoding="utf-8-sig", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, tbl, title in [(axes[0], debt_tbl, "Доля оттока по долгу"), (axes[1], supp_tbl, "Доля оттока по числу обращений")]:
    bars = ax.bar(tbl.iloc[:, 0], tbl.churn_rate * 100, color="#4C72B0")
    for b, (_, r) in zip(bars, tbl.iterrows()):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1, f"{r.churn_rate*100:.1f}%\nn={int(r.n_observations)}", ha="center", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Доля оттока, %")
    ax.tick_params(axis='x', rotation=20)
fig.suptitle("Рис. 4. Доля оттока по долгу и обращениям (обучение)")
fig.savefig(FIG / "fig4_debt_support.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Рисунок 5: сравнение сегмента — регион (train)
# ---------------------------------------------------------------------------
reg_tbl = train.groupby("region").agg(n_observations=("customer_id", "size"),
                                        n_customers=("customer_id", "nunique"),
                                        n_churn=("churn_target", "sum")).reset_index()
reg_tbl["churn_rate"] = reg_tbl.n_churn / reg_tbl.n_observations
reg_tbl = reg_tbl.sort_values("churn_rate", ascending=False)
reg_tbl.to_csv(FIG / "fig5_region.csv", sep=";", encoding="utf-8-sig", index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(reg_tbl.region, reg_tbl.churn_rate * 100, color="#55A868")
for b, (_, r) in zip(bars, reg_tbl.iterrows()):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f"{r.churn_rate*100:.1f}%\nn={int(r.n_observations)}", ha="center", fontsize=8)
ax.set_ylabel("Доля оттока, %")
ax.set_title("Рис. 5. Доля оттока по региону (обучение)")
ax.tick_params(axis='x', rotation=20)
fig.savefig(FIG / "fig5_region.png", dpi=150)
plt.close(fig)

# альтернативное объяснение для H1 (долг): пересчёт доли оттока по долгу отдельно по тарифам
alt_tbl = train.groupby(["tariff", "debt_group"], observed=True).agg(
    n_observations=("customer_id", "size"), n_churn=("churn_target", "sum")).reset_index()
alt_tbl["churn_rate"] = alt_tbl.n_churn / alt_tbl.n_observations
alt_tbl.to_csv(FIG / "fig4_debt_by_tariff_alt_explanation.csv", sep=";", encoding="utf-8-sig", index=False)

# ---------------------------------------------------------------------------
# Рисунок 6: региональная динамика неуспешных проб
# ---------------------------------------------------------------------------
net_raw = pd.read_csv("/home/claude/runforest_fork/3_eda/data/network_quality_monthly.csv", sep=";", encoding="utf-8-sig", parse_dates=["measurement_month"])
relevant_months = pd.date_range("2025-09-01", "2026-07-01", freq="MS")  # месяцы измерений, соответствующие prev_month для T в сетке EDA
net_relevant = net_raw[net_raw.measurement_month.isin(relevant_months)].copy()
net_relevant["fail_rate"] = net_relevant.failed_probe_count / net_relevant.probe_count

avail_flag = net_avail.drop_duplicates(["region", "measurement_month"])[["region", "measurement_month", "available"]]
net_relevant = net_relevant.merge(avail_flag, on=["region", "measurement_month"], how="left")
net_relevant.to_csv(FIG / "fig6_network.csv", sep=";", encoding="utf-8-sig", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
for r, sub in net_relevant.groupby("region"):
    sub = sub.sort_values("measurement_month")
    ax.plot(sub.measurement_month, sub.fail_rate * 100, marker="o", label=r)
    known = sub[sub.available == False]
    ax.scatter(known.measurement_month, known.fail_rate * 100, facecolors="none", edgecolors="red", s=80, zorder=5)
ax.set_ylabel("Доля неуспешных проб, %")
ax.set_title("Рис. 6. Региональная динамика неуспешных технических проб\n(красный кружок = отчёт ещё НЕ был известен на соответствующую дату T из сетки EDA)")
ax.legend(fontsize=8, ncol=2)
fig.autofmt_xdate()
fig.savefig(FIG / "fig6_network.png", dpi=150)
plt.close(fig)

overall_fail = net_relevant.groupby("measurement_month").apply(lambda d: d.failed_probe_count.sum() / d.probe_count.sum())
print("\nОбщая доля неуспешных проб по месяцам (sum/sum, не среднее долей):")
print((overall_fail * 100).round(2))

print("\nГотово. Таблицы и рисунки сохранены в", FIG)
