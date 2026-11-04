"""Shared loading and washing-machine cycle detection for Sections 2 and 3."""
import numpy as np
import pandas as pd

from clock import add_local, to_utc
from config import (HEATER_W, MAX_PAUSE_MIN, MIN_CYCLE_MIN, MIN_CYCLE_WH, ON_W, PROC_DIR,
                    WM_END)

FLAT_MIN = 60  # aggregate identical for >= 60 min: official forward-fill over an outage, not real load


def flat_runs(s: pd.Series, min_len: int) -> pd.Series:
    """True where the series holds exactly the same value for >= min_len consecutive minutes."""
    same = s.diff().eq(0)
    run_id = (~same).cumsum()
    run_len = s.groupby(run_id).transform("size")
    return (run_len >= min_len) & s.notna()


def load_house(h: int) -> pd.DataFrame:
    """1-minute frame on true UTC with local time and a `valid` mask.

    valid = aggregate and WM present, Issues flag not set, aggregate not flat-lined (a
    forward-filled outage in the official data), and within the documented WM period.
    """
    df = to_utc(pd.read_parquet(PROC_DIR / f"clean_1min_h{h}.parquet"), h)
    if h in WM_END:
        df = df[df.index < pd.Timestamp(WM_END[h])]
    df = df.rename(columns={"Aggregate": "agg"})
    df["flat"] = flat_runs(df["agg"], FLAT_MIN)
    df["valid"] = (df["agg"].notna() & df["wm"].notna() & (df["issues"].fillna(0) == 0)
                   & ~df["flat"])
    return add_local(df)


def detect_cycles(wm: pd.Series, on_w=ON_W, max_pause=MAX_PAUSE_MIN, min_len=MIN_CYCLE_MIN,
                  min_wh=MIN_CYCLE_WH) -> pd.DataFrame:
    """Washing-machine cycles from a 1-minute power series (W).

    Plain language: the machine is 'running' in any minute it draws more than `on_w` watts.
    Running minutes separated by quiet spells of up to `max_pause` minutes are the same wash
    (machines pause while soaking, draining or between rinse and spin). A wash must last at
    least `min_len` minutes and use at least `min_wh` Wh; shorter blips are the display,
    door lock or a one-off sensor glitch.
    """
    p = wm.fillna(0).to_numpy()
    on = np.flatnonzero(p > on_w)
    if len(on) == 0:
        return pd.DataFrame(columns=["start", "end", "minutes", "energy_wh", "peak_w", "heater_min", "heater_wh"])
    breaks = np.flatnonzero(np.diff(on) > max_pause + 1)
    starts = np.r_[on[0], on[breaks + 1]]
    ends = np.r_[on[breaks], on[-1]]
    idx = wm.index
    rows = []
    for s, e in zip(starts, ends):
        seg = p[s:e + 1]
        energy = seg.sum() / 60
        if e - s + 1 < min_len or energy < min_wh:
            continue
        heat = seg > HEATER_W
        rows.append({"start": idx[s], "end": idx[e], "i0": s, "i1": e, "minutes": e - s + 1,
                     "energy_wh": energy, "peak_w": seg.max(), "heater_min": int(heat.sum()),
                     "heater_wh": seg[heat].sum() / 60})
    return pd.DataFrame(rows)


def cycle_mask(index: pd.Index, cycles: pd.DataFrame) -> np.ndarray:
    m = np.zeros(len(index), dtype=bool)
    for s, e in zip(cycles["i0"], cycles["i1"]):
        m[s:e + 1] = True
    return m
