import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------------
# SET INPUT / OUTPUT HERE
# -------------------------
demand_path = r"C:\Users\user\Downloads\data json.json"

SUPPLY_SCENARIOS = [
    {"name": "IP4", "path": r"C:\Users\user\Downloads\ip4.csv", "year": 2026},
    {"name": "IP6", "path": r"C:\Users\user\Downloads\IP6.csv", "year": 2026},
    {"name": "IP8", "path": r"C:\Users\user\Downloads\IP8.csv", "year": 2026},
]

output_avg_png = r"C:\Users\user\Downloads\avg_24h_profile_all_scenarios.png"
output_daytype_png = r"C:\Users\user\Downloads\daytype_profile_all_scenarios.png"
output_season_png = r"C:\Users\user\Downloads\seasonal_profile_all_scenarios.png"
output_xlsx = r"C:\Users\user\Downloads\plot_data_all_scenarios.xlsx"

# Same day-type definition as Day-Type Solar Match.py: Mon-Sat = Weekday, Sun = Weekend.
WEEKDAY_DAYS_OF_WEEK = {0, 1, 2, 3, 4, 5}


def day_type_of(dow: int) -> str:
    return "Weekday" if dow in WEEKDAY_DAYS_OF_WEEK else "Weekend"


def season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (6, 7, 8):
        return "JJA"
    return "SHOULDER"


HOUR_COLS = [f"{h:02d}:00" for h in range(24)]

# -------------------------
# STYLE
# -------------------------
COLOR_WEEKDAY = "#e22a87"
COLOR_WEEKEND = "#1e75bb"
COLOR_BLENDED = "#2f3b4f"
COLOR_USED = "#ffe784"
COLOR_SUPPLY = "#6b7280"
COLOR_DJF = "#1e75bb"
COLOR_JJA = "#ffcc00"
COLOR_SHOULDER = "#6b7280"


# -------------------------
# DEMAND: build hourly-resolution Weekday/Weekend profiles
# -------------------------
def read_demand_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    times = raw["times"]
    hours = [t[:2] + ":00" for t in times]
    rows = []
    for entry in raw["dates"]:
        date = pd.to_datetime(entry["date"])
        rows.append(pd.DataFrame({
            "DayType": day_type_of(date.dayofweek),
            "Hour": hours,
            "Demand_kW": entry["demand_kw"],
        }))
    return pd.concat(rows, ignore_index=True)


def build_hourly_demand_curves(demand_path: str):
    """
    Averages the sample month into hourly-resolution Weekday/Weekend curves.
    Demand is clipped to >= 0 (regenerative/export excluded), matching the
    convention already used for "Total Demand" in Day-Type Solar Match.py --
    so these plots stay consistent with the headline numbers already reported.
    """
    demand_long = read_demand_json(demand_path)
    demand_long["Demand_kW"] = demand_long["Demand_kW"].clip(lower=0)
    hourly = demand_long.groupby(["DayType", "Hour"], as_index=False)["Demand_kW"].mean()
    weekday = hourly[hourly["DayType"] == "Weekday"].set_index("Hour")["Demand_kW"].reindex(HOUR_COLS)
    weekend = hourly[hourly["DayType"] == "Weekend"].set_index("Hour")["Demand_kW"].reindex(HOUR_COLS)
    return weekday.to_numpy(), weekend.to_numpy()


def blend_by_daycount(weekday_curve, weekend_curve, year: int):
    """Weights Weekday/Weekend curves by how many of each actually occur in `year`."""
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    n_weekday = sum(day_type_of(d.dayofweek) == "Weekday" for d in dates)
    n_weekend = len(dates) - n_weekday
    return (weekday_curve * n_weekday + weekend_curve * n_weekend) / len(dates)


# -------------------------
# SUPPLY: hourly-collapsed long table per scenario (kW half-hourly -> kWh hourly)
# -------------------------
def read_supply_halfhourly_ddmm(path: str, year: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    valid = df["Start Date"].astype(str).str.match(r"^\d{1,2}-\d{1,2}$")
    df = df[valid].copy()
    df["Date_dt"] = pd.to_datetime(df["Start Date"] + f"-{year}", format="%d-%m-%Y", errors="coerce")

    half_hour_cols = [c for c in df.columns if c not in ("Start Date", "Date_dt")]
    for c in half_hour_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    hourly = pd.DataFrame({"Date_dt": df["Date_dt"]})
    for h in range(24):
        t00, t30 = f"{h:02d}:00:00", f"{h:02d}:30:00"
        v00 = df[t00] if t00 in df.columns else 0
        v30 = df[t30] if t30 in df.columns else 0
        hourly[f"{h:02d}:00"] = (v00 + v30) / 2  # kW readings -> kWh for a 1h bucket

    long = hourly.melt(id_vars=["Date_dt"], value_vars=HOUR_COLS, var_name="Hour", value_name="Supply_kWh")
    long["Season"] = long["Date_dt"].dt.month.apply(season_from_month)
    return long


# -------------------------
# BUILD CURVES FOR EVERY SCENARIO
# -------------------------
weekday_demand, weekend_demand = build_hourly_demand_curves(demand_path)

scenario_data = {}
for scenario in SUPPLY_SCENARIOS:
    supply_long = read_supply_halfhourly_ddmm(scenario["path"], scenario["year"])
    blended_demand = blend_by_daycount(weekday_demand, weekend_demand, scenario["year"])

    avg_supply = supply_long.groupby("Hour")["Supply_kWh"].mean().reindex(HOUR_COLS).to_numpy()
    avg_used = np.minimum(blended_demand, avg_supply)

    seasonal_supply = {
        season: supply_long[supply_long["Season"] == season].groupby("Hour")["Supply_kWh"]
        .mean().reindex(HOUR_COLS).to_numpy()
        for season in ["DJF", "JJA", "SHOULDER"]
    }

    scenario_data[scenario["name"]] = {
        "blended_demand": blended_demand,
        "avg_supply": avg_supply,
        "avg_used": avg_used,
        "seasonal_supply": seasonal_supply,
    }

x = list(range(24))


def style_axes(ax):
    ax.set_xlabel("Hour")
    ax.set_ylabel("kWh / hour")
    ax.set_xticks(x)
    ax.set_xticklabels(HOUR_COLS, rotation=45, ha="right", fontsize=7)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# -------------------------
# PLOT 1: Average 24h profile, one subplot per scenario
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)
for ax, (name, d) in zip(axes, scenario_data.items()):
    ax.plot(x, d["blended_demand"], label="Demand (annual avg)", color=COLOR_BLENDED, linewidth=2.2)
    ax.plot(x, d["avg_supply"], label="Supply", color=COLOR_SUPPLY, linewidth=2.2)
    ax.fill_between(x, d["avg_used"], color=COLOR_USED, alpha=0.7, label="Used Solar")
    ax.set_title(f"{name}: Average 24h Profile")
    style_axes(ax)
axes[0].legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(output_avg_png, dpi=200)
plt.close()


# -------------------------
# PLOT 2: Weekday vs Weekend demand, with each scenario's supply overlaid
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)
for ax, (name, d) in zip(axes, scenario_data.items()):
    ax.fill_between(x, d["avg_used"], color=COLOR_USED, alpha=0.7, label="Avg Used Solar")
    ax.plot(x, weekday_demand, color=COLOR_WEEKDAY, linewidth=2.2, label="Weekday Demand (Mon-Sat)")
    ax.plot(x, weekend_demand, color=COLOR_WEEKEND, linewidth=2.2, label="Weekend Demand (Sun)")
    ax.plot(x, d["blended_demand"], color=COLOR_BLENDED, linestyle="--", linewidth=1.8, label="Blended Avg Demand")
    ax.plot(x, d["avg_supply"], color=COLOR_SUPPLY, linestyle="--", linewidth=1.8, label="Avg Supply")
    ax.set_title(f"{name}: Weekday vs Weekend Demand")
    style_axes(ax)
axes[0].legend(loc="upper left", fontsize=7)
plt.tight_layout()
plt.savefig(output_daytype_png, dpi=200)
plt.close()


# -------------------------
# PLOT 3: Seasonal supply vs (season-invariant) demand
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)
for ax, (name, d) in zip(axes, scenario_data.items()):
    ax.fill_between(x, d["avg_used"], color=COLOR_USED, alpha=0.7, label="Avg Used Solar")
    ax.plot(x, d["blended_demand"], color=COLOR_BLENDED, linewidth=2.2,
            label="Demand (same all year -- single January sample month)")
    ax.plot(x, d["seasonal_supply"]["DJF"], color=COLOR_DJF, linewidth=2.0, label="Winter Supply (DJF)")
    ax.plot(x, d["seasonal_supply"]["JJA"], color=COLOR_JJA, linewidth=2.0, label="Summer Supply (JJA)")
    ax.plot(x, d["seasonal_supply"]["SHOULDER"], color=COLOR_SHOULDER, linewidth=2.0, label="Shoulder Supply")
    ax.set_title(f"{name}: Seasonal Supply vs Demand")
    style_axes(ax)
axes[0].legend(loc="upper left", fontsize=7)
plt.tight_layout()
plt.savefig(output_season_png, dpi=200)
plt.close()


# -------------------------
# SAVE UNDERLYING DATA
# -------------------------
with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
    pd.DataFrame({"Hour": HOUR_COLS, "Weekday Demand": weekday_demand, "Weekend Demand": weekend_demand}) \
        .to_excel(writer, sheet_name="Demand Profiles", index=False)
    for name, d in scenario_data.items():
        pd.DataFrame({
            "Hour": HOUR_COLS,
            "Blended Demand": d["blended_demand"],
            "Avg Supply": d["avg_supply"],
            "Avg Used Solar": d["avg_used"],
            "DJF Supply": d["seasonal_supply"]["DJF"],
            "JJA Supply": d["seasonal_supply"]["JJA"],
            "Shoulder Supply": d["seasonal_supply"]["SHOULDER"],
        }).to_excel(writer, sheet_name=name, index=False)

print("Saved:", output_avg_png)
print("Saved:", output_daytype_png)
print("Saved:", output_season_png)
print("Saved:", output_xlsx)
