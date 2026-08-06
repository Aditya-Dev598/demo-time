import pandas as pd

# -------------------------
# SET INPUT / OUTPUT HERE
# -------------------------
input_path  = r"C:\Users\user\Downloads\P.csv"
output_path = r"C:\Users\user\Downloads\PVGIS_supply_hourly_WIDE.xlsx"  # can also be .csv


def clean_pvgis_to_wide_hourly(input_path: str, output_path: str):
    # 1) Find where PVGIS real table begins (line that starts with "time,")
    header_row = None
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.strip().lower().startswith("time,"):
                header_row = i
                break
    if header_row is None:
        raise ValueError("Could not find PVGIS table header (line starting with 'time,').")

    # 2) Read the PVGIS table
    df = pd.read_csv(input_path, skiprows=header_row)

    if "time" not in df.columns or "P" not in df.columns:
        raise ValueError(f"Expected columns 'time' and 'P'. Found: {df.columns.tolist()}")

    # 3) Parse PVGIS time: YYYYMMDD:HHMM (e.g. 20160101:0030)
    dt = pd.to_datetime(df["time"].astype(str).str.strip(), format="%Y%m%d:%H%M", errors="coerce")
    if dt.isna().mean() > 0.1:
        raise ValueError("Failed to parse PVGIS 'time' column with format %Y%m%d:%H%M.")

    # 4) Convert P from W → kW
    kw = pd.to_numeric(df["P"], errors="coerce").fillna(0) / 1000.0

    out = pd.DataFrame({"datetime": dt, "kW": kw})

    # 5) Bin to hourly (PVGIS is usually at :30) to match your demand bins
    out["datetime"] = out["datetime"].dt.floor("h")

    # If duplicates after flooring, average them (safe)cl
    out = out.groupby("datetime", as_index=False)["kW"].mean()

    # 6) Convert to WIDE daily table: Date | Total Units | 00:00..23:00
    out["DateKey"] = out["datetime"].dt.normalize()
    out["Hour"] = out["datetime"].dt.strftime("%H:%M")

    wide = out.pivot_table(index="DateKey", columns="Hour", values="kW", aggfunc="mean")

    hour_cols = [f"{h:02d}:00" for h in range(24)]
    wide = wide.reindex(columns=hour_cols).fillna(0)

    final = pd.DataFrame()
    final["Date"] = wide.index.strftime("%d/%m/%Y")
    final["Total Units"] = wide[hour_cols].sum(axis=1)

    for c in hour_cols:
        final[c] = wide[c].values

    final = final[["Date", "Total Units"] + hour_cols]

    # 7) Save output
    if output_path.lower().endswith(".csv"):
        final.to_csv(output_path, index=False)
    else:
        final.to_excel(output_path, index=False)

    print("✅ Saved:", output_path)
    print("Rows:", len(final), "| Cols:", len(final.columns))
    print("First/Last Date:", final["Date"].iloc[0], "→", final["Date"].iloc[-1])


# RUN
clean_pvgis_to_wide_hourly(input_path, output_path)