import pandas as pd
import re
from pathlib import Path

# -------------------------
# SET INPUT / OUTPUT HERE
# -------------------------
input_path  = r"C:\Users\user\Downloads\Twyford_ATS_full_year_2026.csv"          # HH INPUT (48 cols/day)
output_path = r"C:\Users\user\Downloads\H1.xlsx"      # HOURLY OUTPUT (24 cols/day)


# -------------------------
# HELPERS
# -------------------------
# Matches: 0:30, 00:30, 00:30:00, 00.30
TIME_RE = re.compile(r"^\s*(\d{1,2})[:.](\d{2})(?:[:.](\d{2}))?\s*$")

def parse_time_to_hhmm(col) -> str | None:
    """
    Convert a column header into 'HH:MM' if it looks like a time.
    Returns None if not a time column.
    """
    # Excel may load time headers as datetime.time objects
    if hasattr(col, "hour") and hasattr(col, "minute"):
        return f"{col.hour:02d}:{col.minute:02d}"

    # Sometimes headers come through as Timedelta
    if isinstance(col, pd.Timedelta):
        total_seconds = int(col.total_seconds())
        h = (total_seconds // 3600) % 24
        m = (total_seconds % 3600) // 60
        return f"{h:02d}:{m:02d}"

    if isinstance(col, str):
        s = col.strip()
        m = TIME_RE.match(s)
        if not m:
            return None
        hh = int(m.group(1))
        mm = int(m.group(2))
        return f"{hh:02d}:{mm:02d}"

    return None

def write_out(df: pd.DataFrame, path: str):
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in [".xlsx", ".xls"]:
        df.to_excel(path, index=False)
    else:
        raise ValueError("Output must be .csv or .xlsx/.xls")


# -------------------------
# MAIN CONVERTER
# -------------------------
def hh_wide_to_hourly_wide_no_day(input_path: str) -> pd.DataFrame:
    """
    Takes a WIDE HH file:
      Date + 48 half-hour columns (00:00, 00:30, ... 23:30) + optional extra columns

    Outputs WIDE HOURLY file in SAME style (no Day column):
      Date | Total Units | 00:00 | 01:00 | ... | 23:00
    """

    # Read input
    ext = Path(input_path).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Input must be .csv or .xlsx/.xls")

    # Find date column (prefer column containing 'date', else first column)
    date_col = None
    for c in df.columns:
        if isinstance(c, str) and "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    # Detect time columns and normalize their labels to HH:MM
    time_map = {}
    for c in df.columns:
        if c == date_col:
            continue
        hhmm = parse_time_to_hhmm(c)
        if hhmm is not None:
            time_map[c] = hhmm

    # We expect ~48 HH columns; allow some tolerance
    if len(time_map) < 40:
        raise ValueError(
            f"Could not detect HH time columns properly (found {len(time_map)}). "
            f"Quick debug: print(df.columns.tolist()) and check your headers."
        )

    # Keep only date + detected time columns
    hh = df[[date_col] + list(time_map.keys())].rename(columns=time_map).copy()

    # Parse date values
    hh[date_col] = pd.to_datetime(hh[date_col], dayfirst=True, errors="coerce")
    if hh[date_col].isna().mean() > 0.2:
        raise ValueError(f"Date parsing failed for column '{date_col}'. Check date format.")

    # Ensure numeric demand values
    for c in hh.columns:
        if c != date_col:
            hh[c] = pd.to_numeric(hh[c], errors="coerce").fillna(0)

    # Build hourly output (NO Day column)
    out = pd.DataFrame()
    out["Date"] = hh[date_col].dt.strftime("%d/%m/%Y")

    hour_cols = []
    for h in range(24):
        t00 = f"{h:02d}:00"
        t30 = f"{h:02d}:30"

        v00 = hh[t00] if t00 in hh.columns else 0
        v30 = hh[t30] if t30 in hh.columns else 0

        out[t00] = v00 + v30
        hour_cols.append(t00)

    out["Total Units"] = out[hour_cols].sum(axis=1)

    # Order like: Date | Total Units | 00:00..23:00
    out = out[["Date", "Total Units"] + hour_cols]

    return out


# -------------------------
# RUN
# -------------------------
hourly_wide_df = hh_wide_to_hourly_wide_no_day(input_path)
write_out(hourly_wide_df, output_path)

print("✅ Done")
print("Input :", input_path)
print("Output:", output_path)
print("Rows :", len(hourly_wide_df), "| Cols:", len(hourly_wide_df.columns))
print("First/Last Date:", hourly_wide_df["Date"].iloc[0], "→", hourly_wide_df["Date"].iloc[-1])