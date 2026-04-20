
from pathlib import Path
import numpy as np
import pandas as pd

PANEL_EFFICIENCY = 0.20 
PANEL_AREA_M2 = 6 #m^2

INPUT_CSV = Path(__file__).resolve().parent / "telemetry.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "telemetry_cleaned.csv"


def audit(df: pd.DataFrame) -> None:
    """(a) Initial audit: shape, dtypes, null counts."""
    print("=== (a) Initial audit ===")
    print("shape:", df.shape)
    print("\ndtypes:\n", df.dtypes)
    print("\nnull counts per column:\n", df.isna().sum())
    print()


def flag_invalid(df: pd.DataFrame) -> pd.DataFrame:
 
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["timestamp"] = ts
    out = out.sort_values("timestamp").reset_index(drop=True)

    v = out["velocity_kmh"]
    out.loc[v < 0, "velocity_kmh"] = np.nan

    bv = out["battery_voltage"]
    out.loc[(bv <= 0) | (bv > 160), "battery_voltage"] = np.nan

    ir = out["solar_irradiance_wm2"]
    out.loc[(ir < 0) | (ir > 1200), "solar_irradiance_wm2"] = np.nan

    mc = out["motor_current_A"]
    out.loc[mc.abs() > 500, "motor_current_A"] = np.nan

    return out


def interpolate_numeric_time_series(df: pd.DataFrame) -> pd.DataFrame:
    out = df.set_index("timestamp").sort_index()
    numeric_cols = [
        c
        for c in out.columns
        if c != "timestamp" and pd.api.types.is_numeric_dtype(out[c])
    ]
    out[numeric_cols] = out[numeric_cols].interpolate(method="time")
    out = out.reset_index()
    
    return out


def add_power_input(df: pd.DataFrame) -> pd.DataFrame:
    """(c) power_input [W] = irradiance [W/m²] * efficiency * area [m²]."""
    out = df.copy()
    out["power_input_W"] = (
        out["solar_irradiance_wm2"] * PANEL_EFFICIENCY * PANEL_AREA_M2
    )
    return out


def main():
    raw = pd.read_csv(INPUT_CSV)
    audit(raw)

    print("=== (b) Cleaning ===")
    masked = flag_invalid(raw)
    cleaned = interpolate_numeric_time_series(masked)
    
    print(
        "Constants: PANEL_EFFICIENCY =",
        PANEL_EFFICIENCY,
        "PANEL_AREA_M2 =",
        PANEL_AREA_M2,
    )

    print("\n=== (c) power_input column ===")
    final_df = add_power_input(cleaned)
    print(final_df.head(10).to_string())
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
