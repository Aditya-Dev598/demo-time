import pandas as pd
from pathlib import Path

# -------------------------
# SET INPUT / OUTPUT HERE
# -------------------------
demand_path = r"C:\Users\user\Downloads\H1.xlsx"
supply_path = r"C:\Users\user\Downloads\PVGIS_supply_hourly_WIDE.xlsx"
output_path = r"C:\Users\user\Downloads\solar_metrics_summary.xlsx"


# -------------------------
# HELPERS
# -------------------------
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


def season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (6, 7, 8):
        return "JJA"
    return "SHOULDER"


def kwh_to_gwh(value):
    return round(value / 1_000_000, 3)


def compute_metrics_match_by_monthday_latest(demand_df: pd.DataFrame, supply_df: pd.DataFrame):
    # Standardise column names
    demand_df.columns = [str(c).strip() for c in demand_df.columns]
    supply_df.columns = [str(c).strip() for c in supply_df.columns]

    hour_cols = [f"{h:02d}:00" for h in range(24)]

    if "Date" not in demand_df.columns or "Date" not in supply_df.columns:
        raise ValueError("Both files must have a 'Date' column.")

    missing_d = [c for c in hour_cols if c not in demand_df.columns]
    missing_s = [c for c in hour_cols if c not in supply_df.columns]

    if missing_d:
        raise ValueError(f"Demand missing hour columns: {missing_d}")
    if missing_s:
        raise ValueError(f"Supply missing hour columns: {missing_s}")

    # Keep only required columns
    demand = demand_df[["Date"] + hour_cols].copy()
    supply = supply_df[["Date"] + hour_cols].copy()

    # Parse dates
    demand["Date_dt"] = parse_date_series(demand["Date"])
    supply["Date_dt"] = parse_date_series(supply["Date"])

    if demand["Date_dt"].isna().mean() > 0.2:
        raise ValueError("Could not parse demand Date reliably.")
    if supply["Date_dt"].isna().mean() > 0.2:
        raise ValueError("Could not parse supply Date reliably.")

    # Match by month-day only, ignoring year
    demand["md"] = demand["Date_dt"].dt.strftime("%m-%d")
    supply["md"] = supply["Date_dt"].dt.strftime("%m-%d")

    # Ensure hourly values are numeric
    for c in hour_cols:
        demand[c] = pd.to_numeric(demand[c], errors="coerce").fillna(0)
        supply[c] = pd.to_numeric(supply[c], errors="coerce").fillna(0)

    # Pick the latest available supply year for each month-day
    supply["_year"] = supply["Date_dt"].dt.year
    supply = (
        supply.sort_values(["md", "_year"], ascending=[True, False])
        .drop_duplicates("md", keep="first")
        .drop(columns=["_year"])
    )

    # Rename demand columns before merging
    demand = demand.rename(columns={c: f"{c}_demand" for c in hour_cols})

    # Merge demand and supply using month-day key
    merged = demand.merge(
        supply[["md"] + hour_cols],
        on="md",
        how="left"
    )

    # Check for missing supply matches
    supply_nan_mask = merged[hour_cols].isna().all(axis=1)

    if supply_nan_mask.any():
        missing_md = merged.loc[supply_nan_mask, "md"].unique().tolist()
        raise ValueError(
            "No supply match found for these month-days: "
            + ", ".join(missing_md[:25])
            + (" ..." if len(missing_md) > 25 else "")
        )

    # Rename supply columns after merge
    merged = merged.rename(columns={c: f"{c}_supply" for c in hour_cols})

    # Calculate used solar per hour
    used_cols = []

    for c in hour_cols:
        used_col = f"{c}_used"
        merged[used_col] = merged[[f"{c}_demand", f"{c}_supply"]].min(axis=1)
        used_cols.append(used_col)

    # Add season column
    merged["Season"] = merged["Date_dt"].dt.month.apply(season_from_month)

    # Create combined summary: DJF, JJA, SHOULDER, Annual
    summary_rows = []

    for period in ["DJF", "JJA", "SHOULDER", "Annual"]:
        if period == "Annual":
            sub = merged
        else:
            sub = merged[merged["Season"] == period]

        demand_total = sub[[f"{c}_demand" for c in hour_cols]].to_numpy().sum()
        supply_total = sub[[f"{c}_supply" for c in hour_cols]].to_numpy().sum()
        used_total = sub[used_cols].to_numpy().sum()
        spillage_total = supply_total - used_total

        summary_rows.append({
            "Period": period,
            "Days": len(sub),
            "Total Demand (GWh)": kwh_to_gwh(demand_total),
            "Total PV Supply (GWh)": kwh_to_gwh(supply_total),
            "Used Solar (GWh)": kwh_to_gwh(used_total),
            "Spillage (GWh)": kwh_to_gwh(spillage_total),
            "Solar Share (%)": round((used_total / demand_total) * 100, 3) if demand_total > 0 else 0.0,
            "Utilisation (%)": round((used_total / supply_total) * 100, 3) if supply_total > 0 else 0.0,
        })

    summary = pd.DataFrame(summary_rows)

    return summary, merged


# -------------------------
# RUN
# -------------------------
demand_df = read_table(demand_path)
supply_df = read_table(supply_path)

summary, matched_detail = compute_metrics_match_by_monthday_latest(
    demand_df,
    supply_df
)

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    matched_detail.to_excel(writer, sheet_name="Matched Detail", index=False)

print("✅ Saved:", output_path)
print(summary)