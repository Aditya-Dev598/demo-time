import json
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------
# SET INPUT / OUTPUT HERE
# -------------------------
demand_path = r"C:\Users\user\Downloads\data json.json"                # .json, .csv, or .xlsx/.xls -- see read_demand()
supply_path = r"C:\Users\user\Downloads\supply data.xlsx"              # HOURLY supply, WIDE (Date | Total Units | 00:00..23:00)
output_path = r"C:\Users\user\Downloads\day_type_solar_summary.xlsx"

INTERVAL_SECONDS = 2  # sampling interval of the demand data

# Day-type definition used for BOTH the demand sample month and the supply
# year(s) -- overrides any Day_Type label baked into the demand file, since
# that label may use a different convention (e.g. Mon-Fri/Sat-Sun).
WEEKDAY_DAYS_OF_WEEK = {0, 1, 2, 3, 4, 5}  # Monday=0 ... Sunday=6; here Mon-Sat
WEEKEND_DAYS_OF_WEEK = {6}                 # Sunday only


def day_type_of(dow: int) -> str:
    return "Weekday" if dow in WEEKDAY_DAYS_OF_WEEK else "Weekend"


# -------------------------
# DEMAND LOADERS
# -------------------------
def read_demand_json(path: str) -> pd.DataFrame:
    """
    Expects the 'shared times array; one demand array per date' layout:
      {
        "metadata": {...},
        "times": ["00:00:00", "00:00:02", ...],
        "dates": [{"date": "2026-01-01", "day_type": "Weekday", "demand_kw": [...]}, ...]
      }
    The file's own "day_type" label is ignored for matching -- day-of-week is
    recomputed from "date" using WEEKDAY_DAYS_OF_WEEK/WEEKEND_DAYS_OF_WEEK
    above, since the file's label may follow a different convention.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    times = raw["times"]
    hours = [t[:2] + ":00" for t in times]

    rows = []
    for entry in raw["dates"]:
        values = entry["demand_kw"]
        if len(values) != len(times):
            raise ValueError(
                f"Date {entry.get('date')} has {len(values)} demand samples "
                f"but the shared 'times' array has {len(times)}."
            )
        date = pd.to_datetime(entry["date"])
        rows.append(pd.DataFrame({
            "Date": date,
            "DayType": day_type_of(date.dayofweek),
            "TimeOfDay": times,
            "Hour": hours,
            "Demand_kW": values,
        }))

    return pd.concat(rows, ignore_index=True)


def read_demand_wide_tabular(path: str) -> pd.DataFrame:
    """
    Expects a WIDE table: Date | Day_Type | <time-of-day columns ...>
    (one row per day, one column per sample -- as produced by CSV/Excel
    exports of the same 'shared times' layout). The file's own Day_Type
    column is ignored for matching, same reasoning as read_demand_json.
    """
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError("Tabular demand file must be .csv or .xlsx/.xls")

    df.columns = [str(c).strip() for c in df.columns]
    time_cols = [c for c in df.columns if c not in ("Date", "Day_Type")]
    if not time_cols:
        raise ValueError("No time-of-day columns found (expected columns besides Date/Day_Type).")

    date_dt = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.assign(DayType=date_dt.dt.dayofweek.map(day_type_of), Date=date_dt)

    long = df.melt(
        id_vars=["Date", "DayType"], value_vars=time_cols,
        var_name="TimeOfDay", value_name="Demand_raw"
    )
    long["Hour"] = long["TimeOfDay"].astype(str).str.slice(0, 2) + ":00"
    long["Demand_kW"] = pd.to_numeric(long["Demand_raw"], errors="coerce").fillna(0)

    return long[["Date", "DayType", "TimeOfDay", "Hour", "Demand_kW"]]


def read_demand(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".json":
        return read_demand_json(path)
    elif ext in (".csv", ".xlsx", ".xls"):
        return read_demand_wide_tabular(path)
    else:
        raise ValueError("Demand file must be .json, .csv, or .xlsx/.xls")


def build_representative_profiles(demand_long: pd.DataFrame) -> pd.DataFrame:
    """
    Averages the sample month's dates into one representative curve per
    DayType (Weekday/Weekend), at native time-of-day resolution -- so
    sub-hour demand/supply crossings are preserved when this profile is
    later matched against a full year of supply.
    """
    counts = demand_long.groupby("DayType")["Date"].nunique()
    print(f"Representative profiles built from: "
          + ", ".join(f"{n} {t} date(s)" for t, n in counts.items()))

    profile = (
        demand_long.groupby(["DayType", "TimeOfDay", "Hour"], as_index=False)["Demand_kW"]
        .mean()
    )
    return profile


# -------------------------
# SUPPLY LOADER
# -------------------------
def read_supply_long(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError("Supply file must be .csv or .xlsx/.xls")

    df.columns = [str(c).strip() for c in df.columns]
    hour_cols = [f"{h:02d}:00" for h in range(24)]
    missing = [c for c in hour_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Supply file missing hour columns: {missing}")

    long = df.melt(id_vars=["Date"], value_vars=hour_cols, var_name="Hour", value_name="Supply_kWh")
    long["Date_dt"] = pd.to_datetime(long["Date"], dayfirst=True, errors="coerce")
    if long["Date_dt"].isna().mean() > 0.2:
        raise ValueError("Could not parse supply Date reliably.")

    long["DayType"] = long["Date_dt"].dt.dayofweek.map(day_type_of)
    long["Supply_kWh"] = pd.to_numeric(long["Supply_kWh"], errors="coerce").fillna(0)
    long["Season"] = long["Date_dt"].dt.month.apply(season_from_month)
    long["Year"] = long["Date_dt"].dt.year

    return long


# -------------------------
# HELPERS
# -------------------------
def season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (6, 7, 8):
        return "JJA"
    return "SHOULDER"


def kwh_to_gwh(v: float) -> float:
    return round(v / 1_000_000, 3)


# -------------------------
# MAIN
# -------------------------
def compute_day_type_metrics(demand_path: str, supply_path: str, interval_seconds: int):
    demand_long = read_demand(demand_path)
    supply_long = read_supply_long(supply_path)

    hours_span = sorted(demand_long["Hour"].unique())
    print(f"Demand: hours {hours_span[0]}-{hours_span[-1]} ({len(hours_span)} of 24 hourly buckets present)")
    if len(hours_span) < 24:
        print(f"WARNING: demand does not cover a full 24-hour day -- hours "
              f"{set(f'{h:02d}:00' for h in range(24)) - set(hours_span)} have no demand samples "
              f"and will be excluded from matching, not treated as zero demand.")

    profile = build_representative_profiles(demand_long)

    n_negative = (profile["Demand_kW"] < 0).sum()
    if n_negative:
        pct = n_negative / len(profile) * 100
        print(f"Clipping {n_negative} negative points in the averaged profile ({pct:.2f}%) to 0 for the "
              f"min(demand, supply) calc -- tracked separately below as Regenerative Export.")
    # Magnitude of negative (regenerative/export) demand in the representative
    # curve, captured before clipping so it isn't just discarded.
    profile["Regen_kW"] = (-profile["Demand_kW"]).clip(lower=0)
    profile["Demand_kW"] = profile["Demand_kW"].clip(lower=0)

    print(f"Supply: {supply_long['Year'].nunique()} year(s) -- {sorted(supply_long['Year'].unique())}")

    # Day-type match: every real (Date, Hour) in supply picks up the
    # representative Weekday or Weekend profile according to that date's
    # ACTUAL day-of-week (Mon-Sat -> Weekday, Sun -> Weekend). Inner join so
    # hours absent from the demand sample are dropped, not treated as zero.
    merged = supply_long.merge(
        profile[["DayType", "Hour", "Demand_kW", "Regen_kW"]],
        on=["DayType", "Hour"],
        how="inner",
    )

    interval_h = interval_seconds / 3600.0
    merged["Demand_kWh"] = merged["Demand_kW"] * interval_h
    merged["Used_kWh"] = np.minimum(merged["Demand_kW"], merged["Supply_kWh"]) * interval_h
    merged["Regen_kWh"] = merged["Regen_kW"] * interval_h

    hourly = merged.groupby(["Date_dt", "Hour"], as_index=False).agg(
        Season=("Season", "first"),
        DayType=("DayType", "first"),
        Year=("Date_dt", lambda s: s.iloc[0].year),
        Demand_kWh=("Demand_kWh", "sum"),
        Used_kWh=("Used_kWh", "sum"),
        Supply_kWh=("Supply_kWh", "first"),
        Regen_kWh=("Regen_kWh", "sum"),
        Samples=("Demand_kWh", "size"),
    )

    return build_summary(hourly), hourly


def build_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for period in ["DJF", "JJA", "SHOULDER", "Annual"]:
        sub = hourly if period == "Annual" else hourly[hourly["Season"] == period]
        demand_total = sub["Demand_kWh"].sum()
        supply_total = sub["Supply_kWh"].sum()
        used_total = sub["Used_kWh"].sum()
        regen_total = sub["Regen_kWh"].sum()
        spill = supply_total - used_total
        # Unused solar and regenerative braking both push power upstream through the
        # same grid connection, so they're additive for an export/curtailment view.
        total_grid_export = spill + regen_total

        summary_rows.append({
            "Period": period,
            "Hours Covered": len(sub),
            "Total Demand (GWh)": kwh_to_gwh(demand_total),
            "Total PV Supply (GWh)": kwh_to_gwh(supply_total),
            "Used Solar (GWh)": kwh_to_gwh(used_total),
            "Solar Spillage (GWh)": kwh_to_gwh(spill),
            "Regenerative Export (GWh)": kwh_to_gwh(regen_total),
            "Total Grid Export (GWh)": kwh_to_gwh(total_grid_export),
            "Solar Share (%)": round(used_total / demand_total * 100, 3) if demand_total > 0 else 0.0,
            "Utilisation (%)": round(used_total / supply_total * 100, 3) if supply_total > 0 else 0.0,
        })

    return pd.DataFrame(summary_rows)


# -------------------------
# RUN
# -------------------------
summary, hourly_detail = compute_day_type_metrics(demand_path, supply_path, INTERVAL_SECONDS)

year_summaries = {
    year: build_summary(hourly_detail[hourly_detail["Year"] == year])
    for year in sorted(hourly_detail["Year"].unique())
}

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary (all years)", index=False)
    for year, ys in year_summaries.items():
        ys.to_excel(writer, sheet_name=f"Summary {year}", index=False)
    hourly_detail.to_excel(writer, sheet_name="Hourly Detail", index=False)

print("\nSaved:", output_path)
print("\n=== All years combined ===")
print(summary.to_string(index=False))
for year, ys in year_summaries.items():
    print(f"\n=== {year} only ===")
    print(ys.to_string(index=False))
