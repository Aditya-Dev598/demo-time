import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------------
# SET INPUT / OUTPUT HERE
# -------------------------
demand_path = r"C:\Users\user\Downloads\Hambleton Jn Demand 2.json"

# Single scenario -- no comparison across capacities.
SCENARIO_NAME = "2700kWp"
SUPPLY_PATH = r"C:\Users\user\Downloads\Hambleton Supply 2.csv"
SUPPLY_YEAR = 2005  # only used to weight the Weekday/Weekend demand blend by real day-of-week counts

output_avg_png = r"C:\Users\user\Downloads\hambleton_avg_24h_profile.png"
output_daytype_png = r"C:\Users\user\Downloads\hambleton_daytype_profile.png"
output_season_png = r"C:\Users\user\Downloads\hambleton_seasonal_profile.png"
output_xlsx = r"C:\Users\user\Downloads\hambleton_plot_data.xlsx"

# Same day-type definition as Day-Type Solar Match.py: Mon-Sat = Weekday, Sun = Weekend.
WEEKDAY_DAYS_OF_WEEK = {0, 1, 2, 3, 4, 5}


def day_type_of(dow: int) -> str:
    return "Weekday" if dow in WEEKDAY_DAYS_OF_WEEK else "Weekend"


def get_demand_site_name(path: str) -> str:
    """Reads the 'site' field from the demand file's metadata, so chart
    titles reflect whichever demand file is actually loaded."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("metadata", {}).get("site") or Path(path).stem


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
# SUPPLY: raw PVGIS hourly export -> long table
# -------------------------
def read_supply_pvgis_raw(path: str) -> pd.DataFrame:
    """
    Same parsing approach as Day-Type Solar Match.py's read_supply_pvgis_raw:
    finds the 'time,' header row, converts P (W) to kW, drops the footer.
    PVGIS already gives one reading per hour, so no collapsing is needed here.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    header_idx = next(i for i, line in enumerate(lines) if line.strip().lower().startswith("time,"))

    df = pd.read_csv(path, skiprows=header_idx)
    df.columns = [str(c).strip() for c in df.columns]

    valid = df["time"].astype(str).str.match(r"^\d{8}:\d{4}$")
    df = df[valid].copy()

    dt = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", errors="coerce")
    long = pd.DataFrame({
        "Hour": dt.dt.floor("h").dt.strftime("%H:00"),
        "Supply_kWh": pd.to_numeric(df["P"], errors="coerce").fillna(0) / 1000.0,
        "Season": dt.dt.month.apply(season_from_month),
    })
    return long


# -------------------------
# BUILD CURVES
# -------------------------
site_name = get_demand_site_name(demand_path)
weekday_demand, weekend_demand = build_hourly_demand_curves(demand_path)
blended_demand = blend_by_daycount(weekday_demand, weekend_demand, SUPPLY_YEAR)

supply_long = read_supply_pvgis_raw(SUPPLY_PATH)
avg_supply = supply_long.groupby("Hour")["Supply_kWh"].mean().reindex(HOUR_COLS).to_numpy()
avg_used = np.minimum(blended_demand, avg_supply)

seasonal_supply = {
    season: supply_long[supply_long["Season"] == season].groupby("Hour")["Supply_kWh"]
    .mean().reindex(HOUR_COLS).to_numpy()
    for season in ["DJF", "JJA", "SHOULDER"]
}

x = list(range(24))


def style_axes(ax):
    ax.set_xlabel("Hour")
    ax.set_ylabel("kWh / hour")
    ax.set_xticks(x)
    ax.set_xticklabels(HOUR_COLS, rotation=45, ha="right", fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# -------------------------
# PLOT 1: Average 24h profile
# -------------------------
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x, blended_demand, label="Demand (annual avg)", color=COLOR_BLENDED, linewidth=2.2)
ax.plot(x, avg_supply, label="Supply", color=COLOR_SUPPLY, linewidth=2.2)
ax.fill_between(x, avg_used, color=COLOR_USED, alpha=0.7, label="Used Solar")
ax.set_title(f"{site_name} {SCENARIO_NAME}: Average 24h Profile")
style_axes(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)
plt.tight_layout()
plt.savefig(output_avg_png, dpi=200, bbox_inches="tight")
plt.close()


# -------------------------
# PLOT 2: Weekday vs Weekend demand, with supply overlaid
# -------------------------
fig, ax = plt.subplots(figsize=(9, 6))
ax.fill_between(x, avg_used, color=COLOR_USED, alpha=0.7, label="Avg Used Solar")
ax.plot(x, weekday_demand, color=COLOR_WEEKDAY, linewidth=2.2, label="Weekday Demand (Mon-Sat)")
ax.plot(x, weekend_demand, color=COLOR_WEEKEND, linewidth=2.2, label="Weekend Demand (Sun)")
ax.plot(x, blended_demand, color=COLOR_BLENDED, linestyle="--", linewidth=1.8, label="Blended Avg Demand")
ax.plot(x, avg_supply, color=COLOR_SUPPLY, linestyle="--", linewidth=1.8, label="Avg Supply")
ax.set_title(f"{site_name} {SCENARIO_NAME}: Weekday vs Weekend Demand")
style_axes(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)
plt.tight_layout()
plt.savefig(output_daytype_png, dpi=200, bbox_inches="tight")
plt.close()


# -------------------------
# PLOT 3: Seasonal supply vs (season-invariant) demand
# -------------------------
fig, ax = plt.subplots(figsize=(9, 6))
ax.fill_between(x, avg_used, color=COLOR_USED, alpha=0.7, label="Avg Used Solar")
ax.plot(x, blended_demand, color=COLOR_BLENDED, linewidth=2.2,
        label="Demand (same all year -- single January sample month)")
ax.plot(x, seasonal_supply["DJF"], color=COLOR_DJF, linewidth=2.0, label="Winter Supply (DJF)")
ax.plot(x, seasonal_supply["JJA"], color=COLOR_JJA, linewidth=2.0, label="Summer Supply (JJA)")
ax.plot(x, seasonal_supply["SHOULDER"], color=COLOR_SHOULDER, linewidth=2.0, label="Shoulder Supply")
ax.set_title(f"{site_name} {SCENARIO_NAME}: Seasonal Supply vs Demand")
style_axes(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig(output_season_png, dpi=200, bbox_inches="tight")
plt.close()


# -------------------------
# SAVE UNDERLYING DATA
# -------------------------
with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
    pd.DataFrame({
        "Hour": HOUR_COLS,
        "Weekday Demand": weekday_demand,
        "Weekend Demand": weekend_demand,
        "Blended Demand": blended_demand,
        "Avg Supply": avg_supply,
        "Avg Used Solar": avg_used,
        "DJF Supply": seasonal_supply["DJF"],
        "JJA Supply": seasonal_supply["JJA"],
        "Shoulder Supply": seasonal_supply["SHOULDER"],
    }).to_excel(writer, sheet_name=SCENARIO_NAME, index=False)

print("Saved:", output_avg_png)
print("Saved:", output_daytype_png)
print("Saved:", output_season_png)
print("Saved:", output_xlsx)
