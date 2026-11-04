"""Clock handling for the official cleaned REFIT files.

Finding (verified in clock_check.py): the cleaned files are in UTC only up to the end
of each house's raw *_Part1 file (~1 Oct 2014). After that the rows come from raw
Part2, which is UK wall-clock time, and the cleaning did not convert it. The readme
says the data were "corrected for UK daylight savings", so this is undocumented.

`file_time` = the timestamp as written in the CSV (Unix column, read naively).
`utc`       = true UTC, after the fix below.
`local`     = Europe/London wall clock, used for hour-of-day / day-of-week analysis,
              because people schedule washing by the clock on the wall.
"""
import os

import pandas as pd

from config import RAW_DIR


def raw_file(house: int, part: int):
    name = f"RAW_House{house}_Part{part}.csv" if house <= 12 else f"RAW_House_{house}_Part{part}.csv"
    return RAW_DIR / name


def part1_end_wallclock(house: int) -> pd.Timestamp:
    """Last timestamp of raw Part1 (wall clock), read from the end of the file without loading it."""
    path = raw_file(house, 1)
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - 4096))
        last = f.read().decode().strip().splitlines()[-1]
    return pd.Timestamp(last.split(",")[0])


def switch_point(house: int) -> pd.Timestamp:
    """file_time at and before which the clean file is UTC (Part1 end converted to UTC)."""
    end_wall = part1_end_wallclock(house)
    return end_wall.tz_localize("Europe/London").tz_convert("UTC").tz_localize(None)


def to_utc(df: pd.DataFrame, house: int) -> pd.DataFrame:
    """Re-index a file_time-indexed frame onto true UTC (naive) and re-grid to 1 minute."""
    sw = switch_point(house)
    early = df[df.index <= sw]
    late = df[df.index > sw].copy()
    loc = late.index.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT")
    late = late[~loc.isna()]                                  # the 1 h that repeats each Oct is unrecoverable
    late.index = loc[~loc.isna()].tz_convert("UTC").tz_localize(None)
    out = pd.concat([early, late])
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out = out.reindex(pd.date_range(out.index[0], out.index[-1], freq="1min"))
    out.index.name = "utc"
    return out


def add_local(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["local"] = df.index.tz_localize("UTC").tz_convert("Europe/London").tz_localize(None)
    return df
