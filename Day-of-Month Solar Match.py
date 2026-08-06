import json
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------
# SET INPUT / OUTPUT HERE
# -------------------------
demand_path = r"C:\Users\user\Downloads\data json.json"                # .json, .csv, or .xlsx/.xls -- see read_demand()
supply_path = r"C:\Users\user\Downloads\supply data.xlsx"              # HOURLY supply, WIDE (Date | Total Units | 00:00..23:00)
output_path = r"C:\Users\user\Downloads\day_of_month_solar_summary.xlsx"

INTERVAL_SECONDS = 2  # sampling interval of the demand data


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
        day_of_month = pd.to_datetime(entry["date"]).day
        rows.append(pd.DataFrame({
            "DayOfMonth": day_of_month,
            "Day_Type": entry["day_type"],
            "Hour": hours,
            "Demand_kW": values,
        }))

    return pd.concat(rows, ignore_index=True)


def read_demand_wide_tabular(path: str) -> pd.DataFrame:
    """
    Expects a WIDE table: Date | Day_Type | <time-of-day columns ...>
    (one row per day, one column per sample -- as produced by CSV/Excel
    exports of the same 'shared times' layout).
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

    df["DayOfMonth"] = pd.to_datetime(df["Date"], dayfirst=True).dt.day

    long = df.melt(
        id_vars=["DayOfMonth", "Day_Type"], value_vars=time_cols,
        var_name="TimeOfDay", value_name="Demand_raw"
    )
    long["Hour"] = long["TimeOfDay"].astype(str).str.slice(0, 2) + ":00"
    long["Demand_kW"] = pd.to_numeric(long["Demand_raw"], errors="coerce").fillna(0)

    return long[["DayOfMonth", "Day_Type", "Hour", "Demand_kW"]]


def read_demand(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".json":
        return read_demand_json(path)
    elif ext in (".csv", ".xlsx", ".xls"):
        return read_demand_wide_tabular(path)
    else:
        raise ValueError("Demand file must be .json, .csv, or .xlsx/.xls")


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

    long["DayOfMonth"] = long["Date_dt"].dt.day
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
def compute_day_of_month_metrics(demand_path: str, supply_path: str, interval_seconds: int):
    demand_long = read_demand(demand_path)
    supply_long = read_supply_long(supply_path)

    n_days = demand_long["DayOfMonth"].nunique()
    hours_span = sorted(demand_long["Hour"].unique())
    print(f"Demand: {n_days} distinct day(s)-of-month, hours {hours_span[0]}-{hours_span[-1]} "
          f"({len(hours_span)} of 24 hourly buckets present)")
    if len(hours_span) < 24:
        print(f"WARNING: demand does not cover a full 24-hour day -- hours "
              f"{set(f'{h:02d}:00' for h in range(24)) - set(hours_span)} have no demand samples "
              f"and will be excluded from matching, not treated as zero demand.")

    n_negative = (demand_long["Demand_kW"] < 0).sum()
    if n_negative:
        pct = n_negative / len(demand_long) * 100
        print(f"Clipping {n_negative} negative demand samples ({pct:.2f}%) to 0 for the "
              f"min(demand, supply) calc -- tracked separately below as Regenerative Export.")
    # Magnitude of negative (regenerative/export) demand, captured before clipping so it
    # isn't just discarded -- it competes with unused solar for the same grid connection.
    demand_long["Regen_kW"] = (-demand_long["Demand_kW"]).clip(lower=0)
    demand_long["Demand_kW"] = demand_long["Demand_kW"].clip(lower=0)

    print(f"Supply: {supply_long['Year'].nunique()} year(s) -- {sorted(supply_long['Year'].unique())}")

    # Day-of-month match: every real (Date, Hour) in supply picks up whichever
    # demand day-of-month shares its calendar day-of-month, regardless of
    # month/year. Inner join so hours absent from demand are dropped rather
    # than raising or being treated as zero demand.
    merged = supply_long.merge(
        demand_long[["DayOfMonth", "Hour", "Day_Type", "Demand_kW", "Regen_kW"]],
        on=["DayOfMonth", "Hour"],
        how="inner",
    )

    interval_h = interval_seconds / 3600.0
    merged["Demand_kWh"] = merged["Demand_kW"] * interval_h
    merged["Used_kWh"] = np.minimum(merged["Demand_kW"], merged["Supply_kWh"]) * interval_h
    merged["Regen_kWh"] = merged["Regen_kW"] * interval_h

    hourly = merged.groupby(["Date_dt", "Hour"], as_index=False).agg(
        Season=("Season", "first"),
        Demand_kWh=("Demand_kWh", "sum"),
        Used_kWh=("Used_kWh", "sum"),
        Supply_kWh=("Supply_kWh", "first"),
        Regen_kWh=("Regen_kWh", "sum"),
        Samples=("Demand_kWh", "size"),
    )

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

    summary = pd.DataFrame(summary_rows)
    return summary, hourly


# -------------------------
# RUN
# -------------------------
summary, hourly_detail = compute_day_of_month_metrics(demand_path, supply_path, INTERVAL_SECONDS)

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    hourly_detail.to_excel(writer, sheet_name="Hourly Detail", index=False)

print("\nSaved:", output_path)
print(summary.to_string(index=False))
