"""Section 4 numbers: evening-peak washing and heater energy per home, and whether the
final model, from the aggregate alone, ranks test homes the same way as the sub-meter."""
import numpy as np
import pandas as pd

from common import detect_cycles
from config import PROC_DIR, TAB_DIR, TEST_HOUSES

PEAK = range(16, 20)


def main():
    cyc = pd.read_parquet(PROC_DIR / "cycles.parquet")
    av = pd.read_csv(TAB_DIR / "s2_availability.csv").set_index("house")
    weeks = av["valid_pct"] / 100 * av["span_days"] / 7
    g = cyc.groupby("house")
    per_home = pd.DataFrame({
        "wm_kwh_per_year": g["energy_wh"].sum() / 1000 / weeks * 52,
        "heater_kwh_per_year": g["heater_wh"].sum() / 1000 / weeks * 52,
        "peak_cycles_per_week": g["hour"].apply(lambda x: x.isin(PEAK).sum()) / weeks,
        "peak_cycle_share": g["hour"].apply(lambda x: x.isin(PEAK).mean()),
    }).round(3)
    per_home.to_csv(TAB_DIR / "s4_per_home_opportunity.csv")
    print(per_home.to_string())
    print(per_home.median().round(2).to_string())

    pred = pd.read_parquet(PROC_DIR / "test_predictions.parquet")
    rows = []
    for h in TEST_HOUSES:
        d = pred[pred["house"] == h].set_index("utc")
        for col in ("wm", "final"):
            s = d[col].where(d["valid"], 0).fillna(0)
            c = detect_cycles(s)
            hours = d["local"].iloc[c["i0"]].dt.hour
            rows.append({"house": h, "source": "sub-meter (truth)" if col == "wm" else "model (aggregate only)",
                         "cycles": len(c), "peak_cycle_share": round(hours.isin(PEAK).mean(), 3),
                         "wm_kwh": round(s.sum() / 60000, 1)})
    check = pd.DataFrame(rows).pivot(index="house", columns="source")
    check.to_csv(TAB_DIR / "s4_model_peak_share_check.csv")
    print(check.to_string())


if __name__ == "__main__":
    main()
