"""Evidence for the clock correction in clock.py, for every washing-machine house.

The raw files are UK wall clock (proof: raw House 1 Part1 repeats 01:00-02:00 on
27 Oct 2013, see s1_raw_checks.csv). For each house we find the lag (minutes) that best
aligns the official cleaned aggregate with the raw aggregate in two summer (BST) months:
  * a month inside raw Part1 (before ~Oct 2014): expect -60 if the clean file is UTC
  * May 2015 (raw Part2):                        expect -60 if UTC, 0 if wall clock
then repeats it after applying clock.to_utc(), where both should read -60 (true UTC).
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from clock import raw_file, switch_point, to_utc
from config import PROC_DIR, TAB_DIR, WM_CHANNEL

LAGS = range(-120, 121, 1)


def raw_agg(house: int) -> pd.Series:
    parts = []
    for p in (1, 2):
        t = pacsv.read_csv(raw_file(house, p), convert_options=pacsv.ConvertOptions(
            include_columns=["Time", "Aggregate"], column_types={"Time": pa.timestamp("s")})).to_pandas()
        parts.append(t.groupby("Time")["Aggregate"].mean().resample("1min").mean())
    return parts[0].combine_first(parts[1])


def best_lag(x: pd.Series, y: pd.Series) -> float:
    """Lag L (minutes) maximising corr(x(t), y(t+L)). y(t+L) == x(t) means y's clock reads L min more."""
    x = x - x.mean()
    best, best_r = np.nan, -1
    for L in LAGS:
        j = pd.concat([x, y.shift(-L)], axis=1).dropna()
        if len(j) < 5000:
            continue
        r = j.corr().iloc[0, 1]
        if r > best_r:
            best, best_r = L, r
    return best


def main():
    rows = []
    for h in WM_CHANNEL:
        raw = raw_agg(h)
        clean_file = pd.read_parquet(PROC_DIR / f"clean_1min_h{h}.parquet")["Aggregate"]
        clean_utc = to_utc(pd.read_parquet(PROC_DIR / f"clean_1min_h{h}.parquet"), h)["Aggregate"]
        sw = switch_point(h)
        # a BST month inside Part1: the last full summer month before the switch
        m1 = ("2014-07-01", "2014-07-31") if sw > pd.Timestamp("2014-08-01") else ("2014-05-01", "2014-05-31")
        m2 = ("2015-05-01", "2015-05-31")
        row = {"house": h, "clean_switches_to_wallclock_after_utc": sw}
        for tag, (a, b) in (("part1_bst", m1), ("part2_bst_2015", m2)):
            r = raw[a:b]
            row[f"{tag}_month"] = a[:7]
            row[f"{tag}_lag_clean_as_shipped"] = best_lag(r, clean_file[a:b])
            row[f"{tag}_lag_after_fix"] = best_lag(r, clean_utc[a:b])
        rows.append(row)
        print(row, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(TAB_DIR / "clock_check.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
