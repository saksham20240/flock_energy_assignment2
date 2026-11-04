"""Section 1: inspect and clean raw REFIT House 1, resample to 1 minute.

Pipeline (each step logged to outputs/tables/s1_cleaning_log.csv):
  1. parse   Time column -> wall clock; check it against the Unix column
  2. clock   raw times are UK wall clock (the 01:00-02:00 hour repeats on 27 Oct 2013);
             convert to UTC, resolving the repeated hour by row order
  3. sort    stable sort on UTC
  4. dedup   drop exact duplicate rows; for the same timestamp with conflicting values keep the
             LAST row (the logger writes on change, so the later row is the newer state)
  5. limits  Aggregate > 20 kW or == 0 -> NaN; IAM > 4 kW or < 0 -> NaN (sensor/bit errors,
             e.g. 4096, 8192, 32767, 65535, 98301 W)
  6. 1-min   mean of readings inside each minute; empty minute -> NaN
  7. parts   Part1 and Part2 overlap (27 Jun - 1 Oct 2014). Use Part1's minute when it has one,
             else Part2's, so the overlap is never double-counted
  8. gaps    NaN runs <= 5 min are forward-filled (the logger only writes on change, so a short
             silence most likely means "unchanged"); longer runs are left NaN and flagged
  9. flags   `agg_outage` (long aggregate gap), `sub_gt_agg` (sum of IAMs > aggregate + 50 W)
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import matplotlib.pyplot as plt

from clock import raw_file, to_utc
from config import (FIG_DIR, MAX_AGG_W, MAX_IAM_W, PROC_DIR, RAW_HOUSE, RAW_PLOT_CHANNELS,
                    SHORT_GAP_MIN, TAB_DIR)
from plotstyle import style

CH = ["Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
log = []


def read_part(part: int) -> pd.DataFrame:
    tbl = pacsv.read_csv(raw_file(RAW_HOUSE, part),
                         convert_options=pacsv.ConvertOptions(column_types={"Time": pa.timestamp("s")}))
    df = tbl.to_pandas()
    df[CH] = df[CH].astype("float64")
    return df


def inspect(df: pd.DataFrame, part: int) -> dict:
    t = df["Time"]
    step = t.diff().dt.total_seconds().iloc[1:]
    unix_vs_time = ((t - pd.Timestamp(0)).dt.total_seconds() - df["Unix"]).abs()
    num = df[CH]
    same_ts = df[df["Time"].duplicated(keep=False)]
    conflicting = same_ts.groupby("Time")[CH].nunique().gt(1).any(axis=1).sum()
    row = {
        "part": part, "rows": len(df), "first": t.min(), "last": t.max(),
        "unparsed_time": int(t.isna().sum()),
        "unix_minus_time_le_1s_pct": round(100 * (unix_vs_time <= 1).mean(), 3),
        "backward_steps": int((step < 0).sum()),
        "largest_backward_step_s": float(step.min()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_timestamps": int(t.duplicated().sum()),
        "timestamps_with_conflicting_values": int(conflicting),
        "median_interval_s": float(step[step > 0].median()),
        "p99_interval_s": float(step[step > 0].quantile(0.99)),
        "gaps_gt_1min": int((step > 60).sum()), "gaps_gt_1h": int((step > 3600).sum()),
        "gaps_gt_1day": int((step > 86400).sum()),
        "agg_gt_20kW": int((num["Aggregate"] > MAX_AGG_W).sum()),
        "agg_eq_0": int((num["Aggregate"] == 0).sum()),
        "iam_gt_4kW": int((num.iloc[:, 1:] > MAX_IAM_W).sum().sum()),
        "negative": int((num < 0).sum().sum()),
        "nan_cells": int(num.isna().sum().sum()),
        "rows_sum_iam_gt_agg_pct": round(100 * (num.iloc[:, 1:].sum(axis=1) > num["Aggregate"] + 50).mean(), 2),
    }
    row["backward_steps_at"] = "; ".join(f"{df['Time'].iloc[i]} -> {df['Time'].iloc[i + 1]}"
                                         for i in np.flatnonzero(step.values < 0))
    return row


def channel_quality(df: pd.DataFrame, part: int) -> pd.DataFrame:
    num = df[CH]
    lim = pd.Series(MAX_IAM_W, index=CH)
    lim["Aggregate"] = MAX_AGG_W
    big = num.gt(lim)
    top = {c: ", ".join(f"{int(v)}" for v in num.loc[big[c], c].value_counts().index[:5]) for c in CH}
    return pd.DataFrame({"part": part, "channel": CH, "min": num.min().values, "max": num.max().values,
                         "mean": num.mean().round(1).values, "nan": num.isna().sum().values,
                         "over_limit": big.sum().values, "most_common_over_limit_values": [top[c] for c in CH]})


def wall_to_utc(df: pd.DataFrame) -> pd.Series:
    """Wall clock -> UTC. In the repeated autumn hour, rows before the backward jump are BST."""
    t = df["Time"]
    back = t.diff().dt.total_seconds() < 0
    ambiguous = np.zeros(len(t), dtype=bool)                  # True = DST (BST) in tz_localize
    for i in np.flatnonzero(back.values):
        hour_start = t.iloc[i].floor("h")
        before = (t.index < t.index[i]) & (t.dt.floor("h") == hour_start).values
        ambiguous |= before
    loc = t.dt.tz_localize("Europe/London", ambiguous=ambiguous, nonexistent="NaT")
    return loc.dt.tz_convert("UTC").dt.tz_localize(None)


def clean_part(df: pd.DataFrame, part: int) -> pd.DataFrame:
    n0 = len(df)
    df = df.assign(utc=wall_to_utc(df))
    nat = df["utc"].isna().sum()
    df = df.dropna(subset=["utc"])
    log.append((part, "2 clock: wall clock -> UTC; drop non-existent spring-forward times", n0, len(df), int(nat)))

    n = len(df)
    df = df.sort_values("utc", kind="stable")
    log.append((part, "3 sort by UTC (rows moved out of order)", n, n,
                int((df.index.to_series().diff().fillna(1) < 0).sum())))

    n = len(df)
    df = df.drop_duplicates(subset=["utc"] + CH)
    log.append((part, "4a drop exact duplicate rows", n, len(df), n - len(df)))
    n = len(df)
    df = df.drop_duplicates(subset=["utc"], keep="last")
    log.append((part, "4b same timestamp, conflicting values: keep last", n, len(df), n - len(df)))

    num = df[CH].copy()
    bad_agg = (num["Aggregate"] > MAX_AGG_W) | (num["Aggregate"] <= 0)
    bad_iam = (num.iloc[:, 1:] > MAX_IAM_W) | (num.iloc[:, 1:] < 0)
    num.loc[bad_agg, "Aggregate"] = np.nan
    num.iloc[:, 1:] = num.iloc[:, 1:].mask(bad_iam)
    log.append((part, "5 out-of-range values -> NaN (cells)", len(df), len(df), int(bad_agg.sum() + bad_iam.sum().sum())))

    num.index = df["utc"].values
    out = num.resample("1min").mean()
    log.append((part, "6 resample to 1-min mean (rows -> minutes)", len(df), len(out), 0))
    return out


def fill_short_gaps(s: pd.Series, max_len: int) -> tuple[pd.Series, pd.Series]:
    """Forward-fill NaN runs of length <= max_len only; return filled series and long-gap mask."""
    na = s.isna()
    run_id = (na != na.shift()).cumsum()
    run_len = na.groupby(run_id).transform("sum")
    short = na & (run_len <= max_len)
    filled = s.copy()
    filled[short] = s.ffill()[short]
    return filled, na & ~short


def gap_runs(mask: pd.Series) -> pd.DataFrame:
    run_id = (mask != mask.shift()).cumsum()
    g = mask[mask].groupby(run_id[mask])
    return pd.DataFrame({"start_utc": g.apply(lambda x: x.index[0]),
                         "end_utc": g.apply(lambda x: x.index[-1]),
                         "minutes": g.size()}).reset_index(drop=True)


def naive_1min(df: pd.DataFrame) -> pd.DataFrame:
    """'Before' view: what you get by resampling the raw file as-is (no sorting fixes, no limits)."""
    x = df[["Unix"] + CH].copy()
    x.index = pd.to_datetime(x.pop("Unix"), unit="s")
    return x.groupby(level=0).mean().resample("1min").mean()


def main():
    style()
    parts = {p: read_part(p) for p in (1, 2)}
    checks = pd.DataFrame([inspect(parts[p], p) for p in parts])
    checks.T.to_csv(TAB_DIR / "s1_raw_checks.csv", header=[f"Part{p}" for p in parts])
    pd.concat([channel_quality(parts[p], p) for p in parts]).to_csv(TAB_DIR / "s1_channel_quality.csv", index=False)
    print(checks.T.to_string())

    before = pd.concat([naive_1min(parts[1]), naive_1min(parts[2])])
    before = before.groupby(level=0).mean()                  # naive merge: averages the overlapping parts

    m1, m2 = clean_part(parts[1], 1), clean_part(parts[2], 2)
    # Overlap: how do the two parts agree, and which has better coverage?
    ov = m1.index.intersection(m2.index)
    a, b = m1.loc[ov, "Aggregate"], m2.loc[ov, "Aggregate"]
    both = a.notna() & b.notna()
    overlap = pd.Series({"overlap_minutes": len(ov), "part1_minutes_with_data": int(a.notna().sum()),
                         "part2_minutes_with_data": int(b.notna().sum()),
                         "corr_1min_aggregate": round(a[both].corr(b[both]), 3),
                         "mae_1min_aggregate_W": round((a[both] - b[both]).abs().mean(), 1),
                         "corr_15min_aggregate": round(a.resample("15min").mean().corr(b.resample("15min").mean()), 3)})
    overlap.to_csv(TAB_DIR / "s1_part_overlap.csv", header=["value"])
    print(overlap.to_string())

    m = m1.combine_first(m2)                                 # Part1 wins where both have data
    log.append((0, "7 merge parts: Part1 minute if present else Part2", len(m1) + len(m2), len(m), 0))
    m = m.reindex(pd.date_range(m.index[0], m.index[-1], freq="1min"))

    cleaned, gap_rows = pd.DataFrame(index=m.index), []
    for c in CH:
        filled, long_gap = fill_short_gaps(m[c], SHORT_GAP_MIN)
        cleaned[c] = filled
        gap_rows.append({"channel": c, "missing_minutes_before": int(m[c].isna().sum()),
                         "short_gap_minutes_filled": int(m[c].isna().sum() - long_gap.sum()),
                         "long_gap_minutes_left_nan": int(long_gap.sum()),
                         "pct_minutes_usable": round(100 * cleaned[c].notna().mean(), 2)})
        if c == "Aggregate":
            cleaned["agg_outage"] = long_gap
    log.append((0, f"8 forward-fill gaps <= {SHORT_GAP_MIN} min; flag longer", len(m), len(cleaned),
                int(sum(r["short_gap_minutes_filled"] for r in gap_rows))))
    cleaned["sub_gt_agg"] = cleaned[CH[1:]].sum(axis=1, min_count=1) > cleaned["Aggregate"] + 50
    pd.DataFrame(gap_rows).to_csv(TAB_DIR / "s1_gap_handling.csv", index=False)
    pd.DataFrame(log, columns=["part", "step", "rows_in", "rows_out", "affected"]).to_csv(
        TAB_DIR / "s1_cleaning_log.csv", index=False)

    outages = gap_runs(cleaned["agg_outage"])
    outages.to_csv(TAB_DIR / "s1_aggregate_outages.csv", index=False)
    bins = [SHORT_GAP_MIN, 60, 6 * 60, 24 * 60, 10**7]
    dist = pd.cut(outages["minutes"], bins, labels=["5 min-1 h", "1-6 h", "6-24 h", "> 1 day"]).value_counts().sort_index()
    print("long aggregate outages:", len(outages), dist.to_dict(), "total h", round(outages.minutes.sum() / 60, 1))
    cleaned.astype({c: "float32" for c in CH}).to_parquet(PROC_DIR / "raw_house1_cleaned_1min.parquet")

    # Validate against the official cleaned House 1 (put both on true UTC first)
    official = to_utc(pd.read_parquet(PROC_DIR / "clean_1min_h1.parquet"), 1)["Aggregate"]
    j = pd.concat([cleaned["Aggregate"], official], axis=1, keys=["mine", "official"]).dropna()
    val = pd.Series({"minutes_compared": len(j), "corr": round(j.corr().iloc[0, 1], 4),
                     "mae_W": round((j.mine - j.official).abs().mean(), 1),
                     "median_abs_err_W": round((j.mine - j.official).abs().median(), 1),
                     "mean_W_mine": round(j.mine.mean(), 1), "mean_W_official": round(j.official.mean(), 1)})
    val.to_csv(TAB_DIR / "s1_vs_official.csv", header=["value"])
    print(val.to_string())

    summary = pd.Series({
        "period_utc": f"{cleaned.index[0]} -> {cleaned.index[-1]}",
        "minutes_total": len(cleaned),
        "aggregate_usable_pct": round(100 * cleaned["Aggregate"].notna().mean(), 2),
        "long_outages": len(outages), "long_outage_hours": round(outages.minutes.sum() / 60, 1),
        "longest_outage_days": round(outages.minutes.max() / 1440, 1),
        "minutes_sub_gt_agg_pct": round(100 * cleaned["sub_gt_agg"].mean(), 2),
    })
    summary.to_csv(TAB_DIR / "s1_summary.csv", header=["value"])
    print(summary.to_string())

    plot_full(before, cleaned)
    plot_zoom(before, cleaned, pd.concat([parts[1], parts[2]], ignore_index=True))


def plot_full(before, after):
    """Hourly maximum over the whole period: spikes and gaps are visible at this scale."""
    fig, axes = plt.subplots(len(RAW_PLOT_CHANNELS), 2, figsize=(14, 10), sharex=True)
    for r, (c, name) in enumerate(RAW_PLOT_CHANNELS.items()):
        for k, (lab, d) in enumerate((("Before cleaning (raw, naive 1-min)", before),
                                      ("After cleaning", after))):
            ax = axes[r, k]
            h = d[c].resample("1h").max()
            ax.plot(h.index, h.values, lw=0.4, color="#c0392b" if k == 0 else "#1f6f8b")
            if k == 1:
                miss = d[c].isna().resample("1h").mean() > 0.5
                ax.fill_between(miss.index, 0, 1, where=miss.values, transform=ax.get_xaxis_transform(),
                                color="0.6", alpha=0.35, lw=0, label="flagged outage")
            ax.set_yscale("symlog", linthresh=100)
            ax.set_ylabel(f"{name}\n(W, hourly max)")
            if r == 0:
                ax.set_title(lab)
    axes[0, 1].legend(loc="upper right")
    fig.suptitle("House 1 raw data, whole period: before vs after cleaning (log scale)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s1_01_full_period_before_after.png")
    plt.close(fig)


def plot_zoom(before, after, raw):
    """Zoom on the largest washing-machine sensor spike and on the October 2013 clock change."""
    wm = raw.loc[raw["Appliance5"] > MAX_IAM_W]
    t_spike = wm.loc[wm["Appliance5"].idxmax(), "Time"]
    windows = [(t_spike.floor("D"), f"day of largest washing-machine spike ({t_spike:%d %b %Y})"),
               (pd.Timestamp("2013-10-26 18:00"), "27 Oct 2013 clock change (raw repeats 01:00-02:00)")]
    fig, axes = plt.subplots(len(RAW_PLOT_CHANNELS), 2, figsize=(14, 10))
    for k, (start, title) in enumerate(windows):
        end = start + pd.Timedelta(hours=24)
        for r, (c, name) in enumerate(RAW_PLOT_CHANNELS.items()):
            ax = axes[r, k]
            b = before[c][start:end]
            a = after[c][start:end]
            ax.plot(b.index, b.values, lw=0.8, color="#c0392b", alpha=0.8, label="before")
            ax.plot(a.index, a.values, lw=0.8, color="#1f6f8b", label="after")
            ax.set_ylabel(f"{name} (W)")
            if c != "Aggregate" and b.max() > MAX_IAM_W:
                ax.set_ylim(-50, max(a.max() * 1.3, 300))
                ax.annotate(f"raw spike {b.max():,.0f} W (off-scale)", xy=(0.01, 0.88),
                            xycoords="axes fraction", color="#c0392b", fontsize=8)
            if r == 0:
                ax.set_title(f"{title}\n(x-axis: UTC for 'after', raw file clock for 'before')")
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s1_02_zoom_before_after.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
