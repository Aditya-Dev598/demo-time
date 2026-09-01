import json
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------
# SET INPUT / OUTPUT HERE
# -------------------------
demand_path = r"C:\Users\user\Downloads\data json.json"                # .json, .csv, or .xlsx/.xls -- see read_demand()
output_path = r"C:\Users\user\Downloads\day_type_solar_summary.xlsx"

INTERVAL_SECONDS = 2  # sampling interval of the demand data

# Each entry is matched against the SAME demand profile independently, for
# side-by-side comparison. "year" is only needed for supply files with no
# year in their dates (e.g. 'Start Date' half-hourly tables like DD-MM);
# omit/None for files that already carry real dates (e.g. supply data.xlsx).
SUPPLY_SCENARIOS = [
    {"name": "IP4", "path": r"C:\Users\user\Downloads\ip4.csv", "year": 2026},
    {"name": "IP6", "path": r"C:\Users\user\Downloads\IP6.csv", "year": 2026},
    {"name": "IP8", "path": r"C:\Users\user\Downloads\IP8.csv", "year": 2026},
]

# A separate site: its own demand sample matched against its own dedicated
# supply file (raw PVGIS export), not part of the SUPPLY_SCENARIOS comparison
# above -- that loop compares PV sizes against ONE demand profile, whereas
# this is a different demand profile entirely.
HAMBLETON_DEMAND_PATH = r"C:\Users\user\Downloads\Hambleton_Jn_N_T1_January_2026_2sec.json"
HAMBLETON_SUPPLY_PATH = r"C:\Users\user\Downloads\Hambleton.csv"
HAMBLETON_OUTPUT_PATH = r"C:\Users\user\Downloads\hambleton_solar_summary.xlsx"

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


def read_supply_halfhourly_ddmm(path: str, year: int) -> pd.DataFrame:
    """
    Reads a WIDE half-hourly supply table with no year in its dates:
      Start Date | 00:00:00 | 00:30:00 | ... | 23:30:00
    Dates are 'DD-MM' (day-first -- confirmed by '31-12' as the last day of
    the year) and get anchored to `year`. Non-date footer rows (e.g. a
    trailing 'Total Result' summary row) and blank trailing columns are
    dropped. Half-hour columns are summed in pairs into hourly buckets to
    match the rest of the pipeline's hourly supply format.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    date_col = "Start Date"
    if date_col not in df.columns:
        raise ValueError(f"Expected a '{date_col}' column in {path}, found: {df.columns.tolist()}")

    valid = df[date_col].astype(str).str.match(r"^\d{1,2}-\d{1,2}$")
    if (~valid).any():
        print(f"Dropping {(~valid).sum()} non-date row(s) from {path}: "
              f"{df.loc[~valid, date_col].tolist()}")
    df = df[valid].copy()

    df["Date_dt"] = pd.to_datetime(df[date_col] + f"-{year}", format="%d-%m-%Y", errors="coerce")
    if df["Date_dt"].isna().any():
        raise ValueError(f"Could not parse some dates in {path} with year {year}.")

    half_hour_cols = [c for c in df.columns if c not in (date_col, "Date_dt")]
    for c in half_hour_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Values are POWER (kW), not energy -- a half-hour sample's energy is
    # kW * 0.5h, so the hourly bucket is the AVERAGE of the two half-hour
    # readings (numerically equal to that hour's energy in kWh, since the
    # bucket duration is 1h), not their sum.
    hourly = pd.DataFrame({"Date_dt": df["Date_dt"]})
    for h in range(24):
        t00, t30 = f"{h:02d}:00:00", f"{h:02d}:30:00"
        v00 = df[t00] if t00 in df.columns else 0
        v30 = df[t30] if t30 in df.columns else 0
        hourly[f"{h:02d}:00"] = (v00 + v30) / 2

    hour_cols = [f"{h:02d}:00" for h in range(24)]
    long = hourly.melt(id_vars=["Date_dt"], value_vars=hour_cols, var_name="Hour", value_name="Supply_kWh")
    long["DayType"] = long["Date_dt"].dt.dayofweek.map(day_type_of)
    long["Season"] = long["Date_dt"].dt.month.apply(season_from_month)
    long["Year"] = long["Date_dt"].dt.year

    return long


def load_supply_halfhourly_raw(path: str, year: int) -> pd.DataFrame:
    """
    Like read_supply_halfhourly_ddmm, but keeps the raw 48 half-hour (kW)
    columns per date instead of collapsing them into hourly buckets --
    needed to interpolate onto a 2-second grid rather than holding supply
    flat within an hour.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    date_col = "Start Date"
    if date_col not in df.columns:
        raise ValueError(f"Expected a '{date_col}' column in {path}, found: {df.columns.tolist()}")

    valid = df[date_col].astype(str).str.match(r"^\d{1,2}-\d{1,2}$")
    if (~valid).any():
        print(f"Dropping {(~valid).sum()} non-date row(s) from {path}: "
              f"{df.loc[~valid, date_col].tolist()}")
    df = df[valid].copy()

    df["Date_dt"] = pd.to_datetime(df[date_col] + f"-{year}", format="%d-%m-%Y", errors="coerce")
    if df["Date_dt"].isna().any():
        raise ValueError(f"Could not parse some dates in {path} with year {year}.")

    half_hour_cols = [f"{h:02d}:{m:02d}:00" for h in range(24) for m in (0, 30)]
    missing = [c for c in half_hour_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing half-hour columns: {missing}")
    for c in half_hour_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df[["Date_dt"] + half_hour_cols].sort_values("Date_dt").reset_index(drop=True)


def compute_metrics_2s(profile: pd.DataFrame, halfhourly_df: pd.DataFrame, interval_seconds: int):
    """
    True 2-second-vs-2-second matching: each real date's 30-minute (kW)
    supply readings are linearly interpolated onto the SAME 2-second
    time-of-day grid the demand profile already uses, instead of holding
    supply flat within an hourly or half-hourly bucket. The last half-hour
    of each day is closed using that day's own 00:00 reading as the next
    midnight's anchor -- negligible error since solar output near midnight
    is ~0, and avoids needing the following calendar day's data.
    """
    times_grid = sorted(profile["TimeOfDay"].unique())
    n = len(times_grid)
    time_index = {t: i for i, t in enumerate(times_grid)}
    x_new = np.array([int(t[0:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:8]) for t in times_grid])

    half_hour_cols = [f"{h:02d}:{m:02d}:00" for h in range(24) for m in (0, 30)]
    x_known = np.array([h * 3600 + m * 60 for h in range(24) for m in (0, 30)] + [86400])

    daytype_demand, daytype_regen = {}, {}
    for dt_label, grp in profile.groupby("DayType"):
        arr_d, arr_r = np.zeros(n), np.zeros(n)
        idx = grp["TimeOfDay"].map(time_index).to_numpy()
        arr_d[idx] = grp["Demand_kW"].to_numpy()
        arr_r[idx] = grp["Regen_kW"].to_numpy()
        daytype_demand[dt_label] = arr_d
        daytype_regen[dt_label] = arr_r

    interval_h = interval_seconds / 3600.0
    rows = []
    for _, day_row in halfhourly_df.iterrows():
        date = day_row["Date_dt"]
        y_known = day_row[half_hour_cols].to_numpy(dtype=float)
        y_known = np.append(y_known, y_known[0])
        supply_2s = np.interp(x_new, x_known, y_known)

        dtype = day_type_of(date.dayofweek)
        demand_2s = daytype_demand[dtype]
        regen_2s = daytype_regen[dtype]

        rows.append({
            "Date_dt": date,
            "DayType": dtype,
            "Season": season_from_month(date.month),
            "Year": date.year,
            "Demand_kWh": demand_2s.sum() * interval_h,
            "Used_kWh": np.minimum(demand_2s, supply_2s).sum() * interval_h,
            "Supply_kWh": supply_2s.sum() * interval_h,
            "Regen_kWh": regen_2s.sum() * interval_h,
        })

    daily = pd.DataFrame(rows)
    return build_summary(daily, row_label="Days Covered"), daily


def load_supply_long(path: str, year: int | None = None) -> pd.DataFrame:
    """Dispatches to the right supply loader based on the file's own column layout."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        header = pd.read_csv(path, nrows=0).columns
        header = [str(c).strip() for c in header]
        if "Start Date" in header:
            if year is None:
                raise ValueError(f"{path} has no year in its dates -- pass a reference year.")
            return read_supply_halfhourly_ddmm(path, year)
    return read_supply_long(path)


def read_supply_pvgis_raw(path: str) -> pd.DataFrame:
    """
    Reads a raw PVGIS hourly export: a handful of metadata header lines
    (latitude, slope, nominal kWp, etc.), then a table starting with
    'time,P,G(i),H_sun,T2m,WS10m,Int'. Timestamps are 'YYYYMMDD:HHMM' (PVGIS
    offsets these to HH:30, one reading per real hour) and 'P' is power in
    Watts. A trailing blank line + copyright footer follows the data table.

    Converts P (W) to kW -- numerically equal to that hour's energy in kWh,
    since PVGIS already gives one reading per hour (same convention already
    used for supply data.xlsx, which was itself originally produced this way).
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = next(
        (i for i, line in enumerate(lines) if line.strip().lower().startswith("time,")), None
    )
    if header_idx is None:
        raise ValueError(f"Could not find PVGIS table header (a line starting with 'time,') in {path}.")

    df = pd.read_csv(path, skiprows=header_idx)
    df.columns = [str(c).strip() for c in df.columns]

    if "time" not in df.columns or "P" not in df.columns:
        raise ValueError(f"Expected 'time' and 'P' columns in {path}. Found: {df.columns.tolist()}")

    # Drop the footer (blank line + copyright notice) by keeping only rows
    # whose 'time' actually matches PVGIS's YYYYMMDD:HHMM format.
    valid = df["time"].astype(str).str.match(r"^\d{8}:\d{4}$")
    if (~valid).any():
        print(f"Dropping {(~valid).sum()} non-data row(s) from {path} (PVGIS footer/blank lines).")
    df = df[valid].copy()

    dt = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", errors="coerce")
    if dt.isna().any():
        raise ValueError(f"Failed to parse some PVGIS timestamps in {path} with format %Y%m%d:%H%M.")

    long = pd.DataFrame({
        "Date_dt": dt.dt.normalize(),
        "Hour": dt.dt.floor("h").dt.strftime("%H:00"),
        "Supply_kWh": pd.to_numeric(df["P"], errors="coerce").fillna(0) / 1000.0,
    })
    long["DayType"] = long["Date_dt"].dt.dayofweek.map(day_type_of)
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
def build_demand_profile(demand_path: str) -> pd.DataFrame:
    demand_long = read_demand(demand_path)

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

    return profile


def compute_metrics_from_profile(profile: pd.DataFrame, supply_long: pd.DataFrame, interval_seconds: int):
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


def build_summary(hourly: pd.DataFrame, row_label: str = "Hours Covered") -> pd.DataFrame:
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
            row_label: len(sub),
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
profile = build_demand_profile(demand_path)

results = {}
for scenario in SUPPLY_SCENARIOS:
    print(f"\n--- Scenario: {scenario['name']} ({scenario['path']}) ---")
    halfhourly_df = load_supply_halfhourly_raw(scenario["path"], scenario["year"])
    summary, daily_detail = compute_metrics_2s(profile, halfhourly_df, INTERVAL_SECONDS)
    results[scenario["name"]] = {"summary": summary, "hourly": daily_detail}

# Side-by-side Annual comparison across scenarios, for the quick read.
comparison = pd.concat(
    [res["summary"][res["summary"]["Period"] == "Annual"].assign(Scenario=name)
     for name, res in results.items()],
    ignore_index=True,
)
comparison = comparison[["Scenario"] + [c for c in comparison.columns if c != "Scenario"]]

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    comparison.to_excel(writer, sheet_name="Comparison (Annual)", index=False)
    for name, res in results.items():
        res["summary"].to_excel(writer, sheet_name=f"Summary {name}", index=False)
        res["hourly"].to_excel(writer, sheet_name=f"Daily {name}", index=False)

print("\nSaved:", output_path)
print("\n=== Annual comparison across scenarios ===")
print(comparison.to_string(index=False))
for name, res in results.items():
    print(f"\n=== {name}: full seasonal breakdown ===")
    print(res["summary"].to_string(index=False))


# -------------------------
# HAMBLETON JN N (T1) -- a separate site: its own demand matched to its own
# dedicated PVGIS supply file, not part of the SUPPLY_SCENARIOS comparison.
# -------------------------
print("\n--- Site: Hambleton Jn N (T1) ---")
hambleton_profile = build_demand_profile(HAMBLETON_DEMAND_PATH)
hambleton_supply_long = read_supply_pvgis_raw(HAMBLETON_SUPPLY_PATH)
hambleton_summary, hambleton_hourly = compute_metrics_from_profile(
    hambleton_profile, hambleton_supply_long, INTERVAL_SECONDS
)

with pd.ExcelWriter(HAMBLETON_OUTPUT_PATH, engine="openpyxl") as writer:
    hambleton_summary.to_excel(writer, sheet_name="Summary", index=False)
    hambleton_hourly.to_excel(writer, sheet_name="Hourly Detail", index=False)

print("\nSaved:", HAMBLETON_OUTPUT_PATH)
print("\n=== Hambleton Jn N (T1): full seasonal breakdown ===")
print(hambleton_summary.to_string(index=False))
