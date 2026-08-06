import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------
# SET INPUT / OUTPUT HERE
# -------------------------
demand_path = r"C:\Users\user\Downloads\demand_2s.csv"                    # 2-SECOND DEMAND (long format: timestamp + power)
supply_path = r"C:\Users\user\Downloads\PVGIS_supply_hourly_WIDE.xlsx"    # HOURLY SUPPLY (wide format, from Supply cleaner.py)
output_path = r"C:\Users\user\Downloads\fine_resolution_solar_summary.xlsx"

INTERVAL_SECONDS = 2  # nominal sampling interval of the demand data


# -------------------------
# HELPERS
# -------------------------
def read_table(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError("Input must be .csv or .xlsx/.xls")


def parse_datetime_robust(s: pd.Series) -> pd.Series:
    """
    Tries ISO/format-inferred parsing first (safe for 'YYYY-MM-DD ...' logger
    timestamps), and only falls back to dayfirst parsing if that does
    noticeably worse -- forcing dayfirst=True unconditionally corrupts
    unambiguous ISO timestamps (e.g. mis-parses '2021-06-15').
    """
    default = pd.to_datetime(s, errors="coerce")
    if default.notna().mean() >= 0.99:
        return default

    dayfirst = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return dayfirst if dayfirst.notna().mean() > default.notna().mean() else default


def detect_timestamp_column(df: pd.DataFrame) -> str:
    named = [c for c in df.columns if isinstance(c, str) and
             any(k in c.lower() for k in ("time", "date", "stamp"))]
    ordered = named + [c for c in df.columns if c not in named]

    for c in ordered:
        if pd.api.types.is_numeric_dtype(df[c]):
            # A plain numeric column is almost never a real timestamp, and
            # pd.to_datetime will otherwise "succeed" by reinterpreting the
            # numbers as epoch nanoseconds -- a false positive.
            continue
        parsed = parse_datetime_robust(df[c])
        if parsed.notna().mean() > 0.95:
            return c

    raise ValueError("Could not detect a timestamp column in the demand file.")


def detect_power_column(df: pd.DataFrame, exclude: str) -> str:
    named = [c for c in df.columns if c != exclude and isinstance(c, str) and
             any(k in c.lower() for k in ("power", "kw", "demand", "load"))]
    if named:
        return named[0]

    numeric = [c for c in df.columns if c != exclude and pd.api.types.is_numeric_dtype(df[c])]
    if numeric:
        return numeric[0]

    raise ValueError("Could not detect a power/demand column in the demand file.")


def season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (6, 7, 8):
        return "JJA"
    return "SHOULDER"


def kwh_to_gwh(value):
    return round(value / 1_000_000, 3)


def build_supply_lookup(supply_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshapes an hourly WIDE supply table (Date | Total Units | 00:00..23:00)
    into a long (md, Hour, Supply_kWh) lookup, matched to the LATEST year
    on file per month-day/hour -- same convention as Util output.py.
    """
    supply_df = supply_df.copy()
    supply_df.columns = [str(c).strip() for c in supply_df.columns]

    hour_cols = [f"{h:02d}:00" for h in range(24)]
    missing = [c for c in hour_cols if c not in supply_df.columns]
    if missing:
        raise ValueError(f"Supply file missing hour columns: {missing}")

    long = supply_df.melt(
        id_vars=["Date"], value_vars=hour_cols,
        var_name="Hour", value_name="Supply_kWh"
    )

    long["Date_dt"] = pd.to_datetime(long["Date"], dayfirst=True, errors="coerce")
    if long["Date_dt"].isna().mean() > 0.2:
        raise ValueError("Could not parse supply Date reliably.")

    long["md"] = long["Date_dt"].dt.strftime("%m-%d")
    long["Supply_kWh"] = pd.to_numeric(long["Supply_kWh"], errors="coerce").fillna(0)
    long["_year"] = long["Date_dt"].dt.year

    long = (
        long.sort_values(["md", "Hour", "_year"], ascending=[True, True, False])
        .drop_duplicates(["md", "Hour"], keep="first")
        [["md", "Hour", "Supply_kWh"]]
    )

    return long


# -------------------------
# MAIN
# -------------------------
def compute_fine_resolution_metrics(demand_path: str, supply_path: str, interval_seconds: int):
    demand_df = read_table(demand_path)
    ts_col = detect_timestamp_column(demand_df)
    power_col = detect_power_column(demand_df, exclude=ts_col)

    print(f"Detected timestamp column: '{ts_col}'")
    print(f"Detected power column    : '{power_col}'")

    demand = demand_df[[ts_col, power_col]].rename(
        columns={ts_col: "Timestamp", power_col: "Demand_kW"}
    ).copy()

    demand["Timestamp"] = parse_datetime_robust(demand["Timestamp"])
    if demand["Timestamp"].isna().mean() > 0.01:
        raise ValueError("Too many unparseable timestamps in demand data.")
    demand = demand.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    demand["Demand_kW"] = pd.to_numeric(demand["Demand_kW"], errors="coerce").fillna(0)

    # Sanity-check the assumed sampling interval against the actual data.
    actual_median_s = demand["Timestamp"].diff().dt.total_seconds().median()
    if actual_median_s and abs(actual_median_s - interval_seconds) > 0.5:
        print(
            f"WARNING: configured INTERVAL_SECONDS={interval_seconds} but the data's "
            f"median sample spacing is {actual_median_s:.2f}s. Using {interval_seconds}s as "
            f"configured -- update INTERVAL_SECONDS if this is wrong."
        )

    # 'Date' identifies a real calendar day (distinguishes repeats of the same
    # month-day across different years); 'md' is only used to look up the
    # single-year PV reference profile.
    demand["Date"] = demand["Timestamp"].dt.normalize()
    demand["md"] = demand["Timestamp"].dt.strftime("%m-%d")
    demand["Hour"] = demand["Timestamp"].dt.floor("h").dt.strftime("%H:00")
    demand["Season"] = demand["Timestamp"].dt.month.apply(season_from_month)

    supply_long = build_supply_lookup(read_table(supply_path))

    merged = demand.merge(supply_long, on=["md", "Hour"], how="left")

    unmatched = merged["Supply_kWh"].isna()
    if unmatched.any():
        missing_keys = merged.loc[unmatched, ["md", "Hour"]].drop_duplicates()
        raise ValueError(
            "No supply match found for these month-day/hour keys: "
            + ", ".join(f"{r.md} {r.Hour}" for r in missing_keys.head(25).itertuples())
            + (" ..." if len(missing_keys) > 25 else "")
        )

    interval_h = interval_seconds / 3600.0
    merged["Demand_kWh"] = merged["Demand_kW"] * interval_h
    merged["Used_kWh"] = np.minimum(merged["Demand_kW"], merged["Supply_kWh"]) * interval_h

    # The WIDE supply table stores one energy value PER REAL HOUR, not per
    # power sample. Summing it per 2-second row would multiply it by however
    # many samples fall in that hour, so it must be counted once per distinct
    # (real calendar day, hour) that actually appears in the demand data --
    # not once per (month-day, hour), which would collapse repeat years.
    hours_present = merged[["Date", "Hour", "Season", "Supply_kWh"]].drop_duplicates(["Date", "Hour"])

    summary_rows = []
    for period in ["DJF", "JJA", "SHOULDER", "Annual"]:
        d_sub = merged if period == "Annual" else merged[merged["Season"] == period]
        s_sub = hours_present if period == "Annual" else hours_present[hours_present["Season"] == period]

        demand_total = d_sub["Demand_kWh"].sum()
        supply_total = s_sub["Supply_kWh"].sum()
        used_total = d_sub["Used_kWh"].sum()
        spillage_total = supply_total - used_total

        summary_rows.append({
            "Period": period,
            "Hours Covered": len(s_sub),
            "Total Demand (GWh)": kwh_to_gwh(demand_total),
            "Total PV Supply (GWh)": kwh_to_gwh(supply_total),
            "Used Solar (GWh)": kwh_to_gwh(used_total),
            "Spillage (GWh)": kwh_to_gwh(spillage_total),
            "Solar Share (%)": round((used_total / demand_total) * 100, 3) if demand_total > 0 else 0.0,
            "Utilisation (%)": round((used_total / supply_total) * 100, 3) if supply_total > 0 else 0.0,
        })

    summary = pd.DataFrame(summary_rows)

    # The raw 2-second merged table is far too large for a spreadsheet (a
    # year at 2s intervals is ~15.8M rows, above Excel's ~1.05M row limit),
    # so the detail sheet is an hourly rollup instead of the raw samples.
    hourly_detail = (
        merged.groupby(["Date", "Hour"], as_index=False)
        .agg(Demand_kWh=("Demand_kWh", "sum"),
             Used_kWh=("Used_kWh", "sum"),
             Supply_kWh=("Supply_kWh", "first"),
             Season=("Season", "first"))
        .sort_values(["Date", "Hour"])
    )

    return summary, hourly_detail


# -------------------------
# RUN
# -------------------------
summary, hourly_detail = compute_fine_resolution_metrics(demand_path, supply_path, INTERVAL_SECONDS)

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    hourly_detail.to_excel(writer, sheet_name="Hourly Rollup", index=False)

print("Saved:", output_path)
print(summary)
