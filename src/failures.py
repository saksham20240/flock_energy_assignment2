"""Section 3 failure analysis for the final model (chosen on validation) on the test homes.

1. False alarms: predicted cycles that overlap no true cycle. The other IAM channels of the
   test homes (never given to the model) tell us which appliance was running instead.
2. Missed cycles: recall split by whether the wash heated water and by how much of the
   household load the washing machine made up while it ran.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.csv as pacsv

from clock import to_utc
from common import detect_cycles
from config import CLEAN_DIR, FIG_DIR, OTHER_APPLIANCES, PROC_DIR, TAB_DIR, TEST_HOUSES
from metrics import overlaps
from plotstyle import style


def other_iams(h):
    cols = [f"Appliance{i}" for i in OTHER_APPLIANCES[h]]
    t = pacsv.read_csv(CLEAN_DIR / f"CLEAN_House{h}.csv",
                       convert_options=pacsv.ConvertOptions(include_columns=["Unix"] + cols)).to_pandas()
    t.index = pd.to_datetime(t.pop("Unix"), unit="s")
    t = t[~t.index.duplicated()].sort_index().resample("1min").mean()
    t = to_utc(t, h)
    return t.rename(columns={f"Appliance{i}": n for i, n in OTHER_APPLIANCES[h].items()})


def main():
    style()
    pred = pd.read_parquet(PROC_DIR / "test_predictions.parquet")
    fa_rows, miss_rows, examples = [], [], {}
    for h in TEST_HOUSES:
        d = pred[pred["house"] == h].set_index("utc")
        y = d["wm"].where(d["valid"], 0).fillna(0)
        p = d["final"].where(d["valid"], 0)
        ct, cp = detect_cycles(y), detect_cycles(p)
        cp["correct"] = overlaps(cp, ct)
        ct["found"] = overlaps(ct, cp)
        others = other_iams(h).reindex(d.index)
        for _, r in cp[~cp["correct"]].iterrows():
            seg = others.iloc[r["i0"]:r["i1"] + 1]
            # the metered appliance with the highest peak; only a >= 500 W load can imitate a heater
            peak = seg.max()
            top = peak.idxmax() if peak.max() >= 500 else "no metered load >= 500 W (unmetered)"
            fa_rows.append({"house": h, "start": r["start"], "minutes": r["minutes"],
                            "pred_wh": r["energy_wh"], "likely_source": top,
                            "source_peak_w": peak.max()})
        for _, r in ct.iterrows():
            seg = d.iloc[r["i0"]:r["i1"] + 1]
            miss_rows.append({"house": h, "found": r["found"], "heater": r["heater_min"] > 0,
                              "wm_share_of_agg": r["energy_wh"] / max(seg["agg"].sum() / 60, 1),
                              "energy_wh": r["energy_wh"]})
        examples[h] = (d, ct, cp, others)
    fa = pd.DataFrame(fa_rows)
    fa.to_csv(TAB_DIR / "s3_false_alarms.csv", index=False)
    src = fa.groupby("likely_source").agg(false_alarms=("pred_wh", "size"), predicted_wh=("pred_wh", "sum"))
    src["share_of_false_alarms_pct"] = (100 * src["false_alarms"] / src["false_alarms"].sum()).round(1)
    src.sort_values("false_alarms", ascending=False).round(0).to_csv(TAB_DIR / "s3_false_alarm_sources.csv")
    print(src.sort_values("false_alarms", ascending=False).to_string())

    miss = pd.DataFrame(miss_rows)
    miss["wm_share_band"] = pd.cut(miss["wm_share_of_agg"], [0, 0.15, 0.3, 0.5, 1.01],
                                   labels=["<15%", "15-30%", "30-50%", ">50%"])
    rb = pd.concat([
        miss.groupby("heater")["found"].agg(["mean", "size"]).rename(index={True: "heated wash", False: "cold wash (no heater)"}),
        miss.groupby("wm_share_band", observed=True)["found"].agg(["mean", "size"]).rename(index=lambda x: f"WM share of load {x}"),
    ]).round(3)
    rb.columns = ["recall", "cycles"]
    rb.to_csv(TAB_DIR / "s3_recall_breakdown.csv")
    print(rb.to_string())

    plot_examples(examples, fa)


def plot_examples(examples, fa):
    """Left: biggest false alarm and the appliance that caused it. Right: a missed cold wash."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4))
    big = fa.sort_values("pred_wh", ascending=False).iloc[0]
    d, ct, cp, others = examples[big["house"]]
    w = slice(big["start"] - pd.Timedelta("40min"), big["start"] + pd.Timedelta(minutes=big["minutes"] + 40))
    ax = axes[0]
    ax.plot(d.loc[w, "local"], d.loc[w, "agg"], color="0.7", lw=0.8, label="aggregate")
    if big["likely_source"] in others:
        ax.fill_between(d.loc[w, "local"], 0, others.loc[w, big["likely_source"]].fillna(0), color="#8e44ad",
                        alpha=0.4, label=f"{big['likely_source']} (IAM, not seen by model)")
    ax.plot(d.loc[w, "local"], d.loc[w, "wm"], color="#1f6f8b", lw=1.2, label="actual washing machine")
    ax.plot(d.loc[w, "local"], d.loc[w, "final"], color="#c0392b", lw=1.0, label="predicted")
    ax.set_title(f"Largest false alarm, house {big['house']}: {big['minutes']:.0f} min of 2-6 kW heating loads "
                 f"(highest metered peak: {big['likely_source']})", fontsize=9)
    ax.set_ylabel("W")
    ax.legend(fontsize=7)

    ax = axes[1]
    for h, (d, ct, cp, others) in examples.items():
        missed = ct[(~ct["found"]) & (ct["heater_min"] == 0)]
        if len(missed):
            r = missed.sort_values("energy_wh", ascending=False).iloc[0]
            break
    seg = d.iloc[max(0, r["i0"] - 40): r["i1"] + 40]
    ax.plot(seg["local"], seg["agg"], color="0.7", lw=0.8, label="aggregate")
    ax.plot(seg["local"], seg["wm"], color="#1f6f8b", lw=1.2, label="actual washing machine")
    ax.plot(seg["local"], seg["final"], color="#c0392b", lw=1.0, label="predicted")
    ax.set_title(f"Missed cold wash, house {h}: {r['minutes']:.0f} min, {r['energy_wh']:.0f} Wh, no heater")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s3_04_failure_cases.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
