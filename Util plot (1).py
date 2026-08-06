import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

# -------------------------
# SET INPUT / OUTPUT HERE
# -------------------------
demand_path = r"C:\Users\user\Downloads\H1.xlsx"
supply_path = r"C:\Users\user\Downloads\PVGIS_supply_hourly_WIDE.xlsx"

output_avg_png = r"C:\Users\user\Downloads\avg_demand_supply_usedsolar_24h.png"
output_daytype_png = r"C:\Users\user\Downloads\traction_demand_daytype_weekly_avg.png"
output_season_png = r"C:\Users\user\Downloads\seasonal_supply_vs_demand.png"
output_xlsx = r"C:\Users\user\Downloads\plot_data_daytype_seasonal.xlsx"


# -------------------------
# STYLE
# -------------------------
COLOR_DEMAND = "#e22a87"
COLOR_SATURDAY = "#1e75bb"
COLOR_SUNDAY = "#ffcc00"
COLOR_USED = "#ffe784"
COLOR_AVG_WEEKLY = "#2f3b4f"
COLOR_SUPPLY = "#6b7280"
COLOR_DJF = "#1e75bb"
COLOR_JJA = "#ffcc00"
COLOR_SHOULDER = "#6b7280"


# -------------------------
# HELPERS
# -------------------------
TIME_RE = re.compile(r"^\s*(\d{1,2})[:.](\d{2})(?:[:.](\d{2}))?\s*$")


def read_table(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError("Input must be .csv or .xlsx/.xls")


def parse_date_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")

    if dt.isna().mean() > 0.2:
        dt2 = pd.to_datetime(s, errors="coerce")
        dt = dt.fillna(dt2)

    return dt


def parse_hour_label(col) -> str | None:
    if hasattr(col, "hour") and hasattr(col, "minute"):
        hh, mm = int(col.hour), int(col.minute)
        return f"{hh:02d}:{mm:02d}" if mm == 0 else None

    if isinstance(col, pd.Timedelta):
        total_seconds = int(col.total_seconds())
        hh = (total_seconds // 3600) % 24
        mm = (total_seconds % 3600) // 60
        return f"{hh:02d}:{mm:02d}" if mm == 0 else None

    if isinstance(col, str):
        s = col.strip()
        m = TIME_RE.match(s)

        if not m:
            return None

        hh = int(m.group(1))
        mm = int(m.group(2))

        return f"{hh:02d}:{mm:02d}" if mm == 0 else None

    return None


def normalize_hour_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}

    for c in df.columns:
        lab = parse_hour_label(c)
        if lab is not None:
            rename[c] = lab

    out = df.copy()
    out = out.rename(columns=rename)
    out.columns = [str(c).strip() for c in out.columns]

    return out


def ensure_24_hours(df: pd.DataFrame, who: str):
    hour_cols = [f"{h:02d}:00" for h in range(24)]
    missing = [c for c in hour_cols if c not in df.columns]

    if missing:
        raise ValueError(f"{who}: missing hour columns after normalization: {missing}")

    return hour_cols


def season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (6, 7, 8):
        return "JJA"
    return "SHOULDER"


def build_supply_md_latest(supply_df: pd.DataFrame, hour_cols):
    supply = supply_df[["Date"] + hour_cols].copy()
    supply["Date_dt"] = parse_date_series(supply["Date"])

    if supply["Date_dt"].isna().mean() > 0.2:
        raise ValueError("Could not parse supply Date reliably.")

    supply["md"] = supply["Date_dt"].dt.strftime("%m-%d")

    for c in hour_cols:
        supply[c] = pd.to_numeric(supply[c], errors="coerce").fillna(0)

    supply["_year"] = supply["Date_dt"].dt.year

    supply = (
        supply.sort_values(["md", "_year"], ascending=[True, False])
        .drop_duplicates("md", keep="first")
        .drop(columns=["_year"])
    )

    return supply


def match_supply_to_demand_by_md_latest(demand_df, supply_df, hour_cols):
    demand = demand_df[["Date"] + hour_cols].copy()
    demand["Date_dt"] = parse_date_series(demand["Date"])

    if demand["Date_dt"].isna().mean() > 0.2:
        raise ValueError("Could not parse demand Date reliably.")

    demand["md"] = demand["Date_dt"].dt.strftime("%m-%d")

    for c in hour_cols:
        demand[c] = pd.to_numeric(demand[c], errors="coerce").fillna(0)

    supply_md = build_supply_md_latest(supply_df, hour_cols)

    merged = demand.merge(
        supply_md[["md"] + hour_cols],
        on="md",
        how="left",
        suffixes=("_demand", "_supply")
    )

    supply_cols = [f"{c}_supply" for c in hour_cols]

    if merged[supply_cols].isna().all(axis=1).any():
        missing = merged.loc[merged[supply_cols].isna().all(axis=1), "md"].unique().tolist()
        raise ValueError(f"Supply missing matches for these MM-DD keys: {missing[:25]}")

    merged[supply_cols] = merged[supply_cols].fillna(0)

    merged["Day Type"] = merged["Date_dt"].dt.dayofweek.map(
        lambda x: "Saturday" if x == 5 else "Sunday" if x == 6 else "Weekday"
    )

    merged["Season"] = merged["Date_dt"].dt.month.apply(season_from_month)

    return merged


def avg_profile(df, cols):
    return [df[c].mean() for c in cols]


# -------------------------
# RUN
# -------------------------
demand_df = normalize_hour_columns(read_table(demand_path))
supply_df = normalize_hour_columns(read_table(supply_path))

if "Date" not in demand_df.columns or "Date" not in supply_df.columns:
    raise ValueError("Both files must have a 'Date' column.")

hour_cols = ensure_24_hours(demand_df, "DEMAND")
_ = ensure_24_hours(supply_df, "SUPPLY")

merged = match_supply_to_demand_by_md_latest(demand_df, supply_df, hour_cols)

demand_cols = [f"{c}_demand" for c in hour_cols]
supply_cols = [f"{c}_supply" for c in hour_cols]

x = list(range(24))


# -------------------------
# PLOT 0: BASIC AVERAGE 24H PROFILE
# -------------------------
avg_demand_basic = avg_profile(merged, demand_cols)
avg_supply_basic = avg_profile(merged, supply_cols)
avg_used_basic = [min(d, s) for d, s in zip(avg_demand_basic, avg_supply_basic)]

basic_profile = pd.DataFrame({
    "Hour": hour_cols,
    "Average Demand": avg_demand_basic,
    "Average Supply": avg_supply_basic,
    "Used Solar": avg_used_basic,
})

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(x, avg_demand_basic, label="Average Demand", color=COLOR_DEMAND, linewidth=2.5)
ax.plot(x, avg_supply_basic, label="Average Supply", color=COLOR_SUPPLY, linewidth=2.5)

ax.fill_between(
    x,
    avg_used_basic,
    color=COLOR_USED,
    alpha=0.6,
    label="Used Solar"
)

ax.set_xlabel("Hour", fontsize=12)
ax.set_ylabel("Energy (MWh)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(hour_cols, rotation=45, ha="right")

ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig(output_avg_png, dpi=300)
plt.close()


# -------------------------
# PLOT 1: DAY TYPE DEMAND
# -------------------------
weekday = merged[merged["Day Type"] == "Weekday"]
saturday = merged[merged["Day Type"] == "Saturday"]
sunday = merged[merged["Day Type"] == "Sunday"]

weekday_demand = avg_profile(weekday, demand_cols)
saturday_demand = avg_profile(saturday, demand_cols)
sunday_demand = avg_profile(sunday, demand_cols)

avg_weekly_demand = avg_profile(merged, demand_cols)
avg_weekly_supply = avg_profile(merged, supply_cols)
avg_solar_used = [min(d, s) for d, s in zip(avg_weekly_demand, avg_weekly_supply)]

daytype_profile = pd.DataFrame({
    "Hour": hour_cols,
    "Weekday Demand": weekday_demand,
    "Saturday Demand": saturday_demand,
    "Sunday Demand": sunday_demand,
    "Avg Weekly Demand": avg_weekly_demand,
    "Avg Weekly Supply": avg_weekly_supply,
    "Avg Solar Used": avg_solar_used,
})

fig, ax = plt.subplots(figsize=(16, 7))

ax.fill_between(
    x,
    avg_solar_used,
    color=COLOR_USED,
    alpha=0.6,
    label="Avg Solar Used"
)

ax.plot(x, weekday_demand, color=COLOR_DEMAND, linewidth=2.5, label="Weekday Demand")
ax.plot(x, saturday_demand, color=COLOR_SATURDAY, linewidth=2.5, label="Saturday Demand")
ax.plot(x, sunday_demand, color=COLOR_SUNDAY, linewidth=2.5, label="Sunday Demand")
ax.plot(x, avg_weekly_demand, color=COLOR_AVG_WEEKLY, linestyle="--", linewidth=2.2, label="Avg Weekly Demand")
ax.plot(x, avg_weekly_supply, color=COLOR_SUPPLY, linestyle="--", linewidth=2.2, label="Avg Weekly Supply")

ax.set_title("Traction Demand by Day Type with Weekly Averages", fontsize=15)
ax.set_xlabel("Hour", fontsize=12)
ax.set_ylabel("Energy (MWh)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(hour_cols, rotation=45, ha="right")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="center", ncol=2)

plt.tight_layout()
plt.savefig(output_daytype_png, dpi=300)
plt.close()


# -------------------------
# PLOT 2: SEASONAL SUPPLY VS DEMAND
# -------------------------
avg_demand = avg_profile(merged, demand_cols)
avg_supply_annual = avg_profile(merged, supply_cols)

djf = merged[merged["Season"] == "DJF"]
jja = merged[merged["Season"] == "JJA"]
shoulder = merged[merged["Season"] == "SHOULDER"]

djf_supply = avg_profile(djf, supply_cols)
jja_supply = avg_profile(jja, supply_cols)
shoulder_supply = avg_profile(shoulder, supply_cols)

avg_solar_used_season_plot = [
    min(d, s) for d, s in zip(avg_demand, avg_supply_annual)
]

seasonal_profile = pd.DataFrame({
    "Hour": hour_cols,
    "Average Demand": avg_demand,
    "Average Supply Annual": avg_supply_annual,
    "Winter Supply DJF": djf_supply,
    "Summer Supply JJA": jja_supply,
    "Shoulder Supply": shoulder_supply,
    "Avg Solar Used": avg_solar_used_season_plot,
})

fig, ax = plt.subplots(figsize=(16, 7))

ax.fill_between(
    x,
    avg_solar_used_season_plot,
    color=COLOR_USED,
    alpha=0.6,
    label="Avg Solar Used"
)

ax.plot(x, avg_demand, color=COLOR_DEMAND, linewidth=2.5, label="Average Demand")
ax.plot(x, avg_supply_annual, color=COLOR_AVG_WEEKLY, linestyle="--", linewidth=2.2, label="Average Supply (Annual)")
ax.plot(x, djf_supply, color=COLOR_DJF, linewidth=2.2, label="Winter Supply (DJF)")
ax.plot(x, jja_supply, color=COLOR_JJA, linewidth=2.2, label="Summer Supply (JJA)")
ax.plot(x, shoulder_supply, color=COLOR_SHOULDER, linewidth=2.2, label="Shoulder Supply")

ax.set_title("Average Hourly Supply by Season vs Demand", fontsize=15)
ax.set_xlabel("Hour", fontsize=12)
ax.set_ylabel("Energy (MWh)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(hour_cols, rotation=45, ha="right")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig(output_season_png, dpi=300)
plt.close()


# -------------------------
# SAVE PLOT DATA
# -------------------------
with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
    basic_profile.to_excel(writer, sheet_name="Average 24h Plot Data", index=False)
    daytype_profile.to_excel(writer, sheet_name="Day Type Plot Data", index=False)
    seasonal_profile.to_excel(writer, sheet_name="Seasonal Plot Data", index=False)

print("✅ Saved average plot  :", output_avg_png)
print("✅ Saved day-type plot :", output_daytype_png)
print("✅ Saved seasonal plot :", output_season_png)
print("✅ Saved plot data     :", output_xlsx)