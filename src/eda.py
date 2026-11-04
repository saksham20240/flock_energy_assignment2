"""Section 2: washing-machine EDA on the official cleaned REFIT data (1-minute, true UTC).

All 18 houses with a single washing-machine channel are used (House 4 has two machines,
House 12 none). Hour-of-day and day-of-week use Europe/London local time.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import cycle_mask, detect_cycles, load_house
from config import FIG_DIR, HEATER_W, MAX_PAUSE_MIN, ON_W, PROC_DIR, TAB_DIR, WM_CHANNEL
from plotstyle import style

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
PEAK_HOURS = range(16, 20)  # GB distribution-network evening peak, 16:00-19:59 local


def availability(h, df, cyc):
    grid_missing = df["agg"].isna() | df["wm"].isna()
    st = df.loc[df["valid"], "wm"]
    weeks_valid = df["valid"].sum() / 10080
    agg_wh = df.loc[df["valid"], "agg"].sum() / 60
    wm_wh = st.sum() / 60
    return {
        "house": h, "wm_channel": WM_CHANNEL[h],
        "first_utc": df.index[0].strftime("%Y-%m-%d"), "last_utc": df.index[-1].strftime("%Y-%m-%d"),
        "span_days": round(len(df) / 1440, 1),
        "missing_pct": round(100 * grid_missing.mean(), 1),
        "flatlined_pct": round(100 * df["flat"].mean(), 1),
        "issues_flag_pct": round(100 * (df["issues"] == 1).mean(), 1),
        "valid_pct": round(100 * df["valid"].mean(), 1),
        "wm_exactly_0_pct": round(100 * (st == 0).mean(), 1),
        "wm_standby_0_to_20W_pct": round(100 * ((st > 0) & (st <= ON_W)).mean(), 1),
        "wm_active_gt_20W_pct": round(100 * (st > ON_W).mean(), 2),
        "cycles": len(cyc), "cycles_per_week": round(len(cyc) / weeks_valid, 2),
        "wm_share_of_house_energy_pct": round(100 * wm_wh / agg_wh, 1),
    }


def sensitivity(wm_by_house):
    rows = []
    for on_w in (10, 20, 50):
        for pause in (10, 20, 30, 60):
            n = sum(len(detect_cycles(wm, on_w=on_w, max_pause=pause)) for wm in wm_by_house.values())
            rows.append({"on_threshold_w": on_w, "max_pause_min": pause, "cycles": n})
    out = pd.DataFrame(rows)
    base = out.query("on_threshold_w == @ON_W and max_pause_min == @MAX_PAUSE_MIN")["cycles"].iloc[0]
    out["vs_chosen_pct"] = (100 * (out["cycles"] / base - 1)).round(1)
    return out


def main():
    style()
    avail, cycles, wm_by_house, frames = [], [], {}, {}
    for h in WM_CHANNEL:
        df = load_house(h)
        frames[h] = df
        wm = df["wm"].where(df["valid"])                     # cycles only from trustworthy minutes
        wm_by_house[h] = wm
        cyc = detect_cycles(wm)
        cyc["house"] = h
        cyc["start_local"] = df["local"].iloc[cyc["i0"]].values if len(cyc) else []
        cyc["agg_wh_during"] = [df["agg"].iloc[s:e + 1].sum() / 60 for s, e in zip(cyc["i0"], cyc["i1"])]
        cycles.append(cyc)
        avail.append(availability(h, df, cyc))
        print(f"house {h}: {len(cyc)} cycles", flush=True)
    avail = pd.DataFrame(avail)
    cycles = pd.concat(cycles, ignore_index=True)
    cycles["hour"] = cycles["start_local"].dt.hour
    cycles["dow"] = cycles["start_local"].dt.day_of_week
    cycles["heater_share"] = cycles["heater_wh"] / cycles["energy_wh"]
    cycles["wm_share_of_agg_during"] = cycles["energy_wh"] / cycles["agg_wh_during"]
    avail.to_csv(TAB_DIR / "s2_availability.csv", index=False)
    cycles.drop(columns=["i0", "i1"]).to_csv(TAB_DIR / "s2_cycles.csv", index=False)
    cycles.to_parquet(PROC_DIR / "cycles.parquet")
    print(avail.to_string())

    stats = cycles.groupby("house").agg(
        cycles=("energy_wh", "size"), median_minutes=("minutes", "median"),
        median_energy_wh=("energy_wh", "median"), iqr_energy_wh=("energy_wh", lambda x: x.quantile(.75) - x.quantile(.25)),
        median_peak_w=("peak_w", "median"), median_heater_share=("heater_share", "median"),
        share_no_heater=("heater_min", lambda x: (x == 0).mean()),
        median_wm_share_of_agg_during=("wm_share_of_agg_during", "median"),
        weekend_share=("dow", lambda x: (x >= 5).mean()),
        peak_16_20_share=("hour", lambda x: x.isin(PEAK_HOURS).mean())).round(3)
    stats.to_csv(TAB_DIR / "s2_cycle_stats_by_house.csv")
    print(stats.to_string())

    sens = sensitivity(wm_by_house)
    sens.to_csv(TAB_DIR / "s2_rule_sensitivity.csv", index=False)
    print(sens.to_string())

    # Usage pattern tables: share of each home's cycles by local hour / weekday
    hour_share = pd.crosstab(cycles["house"], cycles["hour"], normalize="index").reindex(columns=range(24), fill_value=0)
    dow_share = pd.crosstab(cycles["house"], cycles["dow"], normalize="index").reindex(columns=range(7), fill_value=0)
    hour_share.round(4).to_csv(TAB_DIR / "s2_hour_share.csv")
    dow_share.round(4).to_csv(TAB_DIR / "s2_dow_share.csv")

    opp = pd.Series({
        "cycles_total": len(cycles),
        "median_energy_per_cycle_wh": round(cycles["energy_wh"].median()),
        "median_heater_share_of_cycle_energy": round(cycles["heater_share"].median(), 3),
        "pooled_heater_share_of_wm_energy": round(cycles["heater_wh"].sum() / cycles["energy_wh"].sum(), 3),
        "cycles_starting_16_20_local_pct": round(100 * cycles["hour"].isin(PEAK_HOURS).mean(), 1),
        "wm_energy_in_16_20_window_pct": round(100 * wm_energy_in_window(frames, wm_by_house), 1),
        "weekend_cycles_pct": round(100 * (cycles["dow"] >= 5).mean(), 1),
        "median_cycles_per_week": avail["cycles_per_week"].median(),
        "median_wm_share_of_house_energy_pct": avail["wm_share_of_house_energy_pct"].median(),
    })
    opp.to_csv(TAB_DIR / "s2_opportunity.csv", header=["value"])
    print(opp.to_string())

    plot_availability(frames)
    plot_representative_day(frames, cycles)
    plot_cycles(frames, cycles)
    plot_cross_house(cycles, avail)
    plot_usage(hour_share, dow_share, cycles)


def wm_energy_in_window(frames, wm_by_house):
    tot = inwin = 0.0
    for h, df in frames.items():
        wm = wm_by_house[h].where(wm_by_house[h] > ON_W)
        tot += wm.sum()
        inwin += wm[df["local"].dt.hour.isin(PEAK_HOURS).values].sum()
    return inwin / tot


def plot_availability(frames):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for k, (h, df) in enumerate(frames.items()):
        d = df["valid"].resample("1D").mean()
        cmap = plt.get_cmap("RdYlGn")
        ax.scatter(d.index, np.full(len(d), k), c=cmap(d.fillna(0).values), marker="|", s=60, lw=1.6)
    ax.set_yticks(range(len(frames)), [f"House {h}" for h in frames])
    ax.invert_yaxis()
    ax.set_title("Daily share of usable minutes (green = all valid, red = none): gaps, flat-lined outages and Issues rows removed")
    sm = plt.cm.ScalarMappable(cmap="RdYlGn")
    fig.colorbar(sm, ax=ax, fraction=0.02, label="usable share")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s2_01_availability.png")
    plt.close(fig)


def plot_representative_day(frames, cycles):
    """A weekday in a typical home whose cycle energy is close to that home's median."""
    h = 2
    c = cycles[cycles["house"] == h]
    med = c["energy_wh"].median()
    df = frames[h]
    cand = c.assign(d=(c["energy_wh"] - med).abs()).sort_values("d")
    for _, row in cand.iterrows():
        day = row["start_local"].normalize()
        sel = df[(df["local"] >= day) & (df["local"] < day + pd.Timedelta("1D"))]
        if sel["valid"].mean() > 0.98 and row["start_local"].day_of_week < 5:
            break
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(sel["local"], sel["agg"], color="0.55", lw=0.8, label="household aggregate")
    ax.fill_between(sel["local"], 0, sel["wm"], color="#1f6f8b", alpha=0.75, label="washing machine (IAM)")
    day_cyc = c[(c["start_local"] >= day) & (c["start_local"] < day + pd.Timedelta("1D"))]
    for _, r in day_cyc.iterrows():
        ax.axvspan(r["start_local"], r["start_local"] + pd.Timedelta(minutes=r["minutes"]), color="#f4a259", alpha=0.18)
    ax.set_ylabel("W")
    ax.set_title(f"House {h}, {day:%a %d %b %Y} (local time): one {row['minutes']:.0f}-min wash, "
                 f"{row['energy_wh']:.0f} Wh (shaded = detected cycle)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s2_02_representative_24h.png")
    plt.close(fig)


def plot_cycles(frames, cycles):
    """Six detected cycles from six homes: median-energy cycle of each."""
    homes = [1, 3, 7, 10, 16, 19]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), sharey=True)
    for ax, h in zip(axes.flat, homes):
        c = cycles[cycles["house"] == h]
        row = c.iloc[(c["energy_wh"] - c["energy_wh"].median()).abs().argmin()]
        df = frames[h]
        pad = 20
        sl = df.iloc[max(0, row["i0"] - pad): row["i1"] + pad + 1]
        t = (sl.index - row["start"]).total_seconds() / 60
        ax.plot(t, sl["agg"], color="0.6", lw=0.8, label="aggregate")
        ax.plot(t, sl["wm"], color="#1f6f8b", lw=1.1, label="washing machine")
        ax.axvspan(0, row["minutes"], color="#f4a259", alpha=0.15, label="detected cycle")
        ax.axhline(HEATER_W, color="#c0392b", ls=":", lw=0.8)
        ax.set_title(f"House {h}: {row['minutes']:.0f} min, {row['energy_wh']:.0f} Wh, "
                     f"heater {100 * row['heater_share']:.0f}% of energy", fontsize=9)
        ax.set_xlabel("minutes from cycle start")
    axes[0, 0].set_ylabel("W")
    axes[1, 0].set_ylabel("W")
    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle(f"Detected washing-machine cycles (median-energy cycle per home). Red dotted line = {HEATER_W} W heater level")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s2_03_detected_cycles.png")
    plt.close(fig)


def plot_cross_house(cycles, avail):
    order = cycles.groupby("house")["energy_wh"].median().sort_values().index
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    data = [cycles.loc[cycles["house"] == h] for h in order]
    for ax, col, lab in ((axes[0], "energy_wh", "energy per cycle (Wh)"),
                         (axes[1], "minutes", "cycle duration (min)")):
        ax.boxplot([d[col] for d in data], orientation="horizontal", showfliers=False, widths=0.6,
                   medianprops=dict(color="#c0392b"))
        ax.set_xlabel(lab)
    axes[0].set_yticks(range(1, len(order) + 1), [f"House {h}" for h in order])
    cpw = avail.set_index("house").loc[order, "cycles_per_week"]
    axes[2].barh(range(1, len(order) + 1), cpw.values, color="#1f6f8b")
    axes[2].set_xlabel("cycles per week (valid data)")
    fig.suptitle("Cross-house comparison (homes sorted by median energy per cycle; boxes = IQR, whiskers = 1.5 IQR)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s2_04_cross_house.png")
    plt.close(fig)


def plot_usage(hour_share, dow_share, cycles):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [2.2, 1, 1.2]})
    im = axes[0].imshow(hour_share.values, aspect="auto", cmap="Blues")
    axes[0].set_yticks(range(len(hour_share)), [f"H{h}" for h in hour_share.index])
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].set_xlabel("local hour of cycle start")
    axes[0].set_title("Share of each home's cycles by start hour")
    fig.colorbar(im, ax=axes[0], fraction=0.03)
    im2 = axes[1].imshow(dow_share.values, aspect="auto", cmap="Blues")
    axes[1].set_yticks(range(len(dow_share)), [f"H{h}" for h in dow_share.index])
    axes[1].set_xticks(range(7), DOW)
    axes[1].set_title("... by weekday")
    fig.colorbar(im2, ax=axes[1], fraction=0.05)
    wk = cycles[cycles["dow"] < 5]["hour"].value_counts(normalize=True).reindex(range(24), fill_value=0)
    we = cycles[cycles["dow"] >= 5]["hour"].value_counts(normalize=True).reindex(range(24), fill_value=0)
    axes[2].plot(wk.index, 100 * wk.values, marker="o", ms=3, label="weekday", color="#1f6f8b")
    axes[2].plot(we.index, 100 * we.values, marker="o", ms=3, label="weekend", color="#f4a259")
    axes[2].axvspan(16, 20, color="#c0392b", alpha=0.08, label="16:00-20:00 peak")
    axes[2].set_xlabel("local hour of cycle start")
    axes[2].set_ylabel("% of cycles (all homes)")
    axes[2].set_title("Pooled start-time profile")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s2_05_usage_patterns.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
