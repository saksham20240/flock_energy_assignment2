"""Model inputs: one long array per split (houses concatenated, separated by padding),
so any minute's input window is a slice of the array."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import load_house
from config import AGG_CLIP_W, ON_W, WINDOW, WM_CLIP_W

HALF = WINDOW // 2


@dataclass
class Split:
    agg: np.ndarray        # W, NaN-free (gaps filled), clipped
    wm: np.ndarray         # W target, NaN where unknown
    valid: np.ndarray      # minute can be scored / used as a training target
    house: np.ndarray      # house id per minute (0 in padding)
    utc: np.ndarray
    local: np.ndarray

    def index_of(self, h: int) -> np.ndarray:
        return np.flatnonzero((self.house == h) & self.valid)


def build_split(houses) -> Split:
    parts = []
    for h in houses:
        df = load_house(h)
        agg = df["agg"].ffill(limit=10).fillna(0).clip(0, AGG_CLIP_W)
        pad = pd.DataFrame({"agg": 0.0, "wm": np.nan, "valid": False, "house": 0,
                            "utc": pd.NaT, "local": pd.NaT}, index=range(HALF + 1))
        parts.append(pd.DataFrame({"agg": agg.values, "wm": df["wm"].clip(0, WM_CLIP_W).values,
                                   "valid": df["valid"].values, "house": h,
                                   "utc": df.index.values, "local": df["local"].values}))
        parts.append(pad)
    d = pd.concat([parts[-1]] + parts, ignore_index=True)
    return Split(agg=d["agg"].to_numpy("float32"), wm=d["wm"].to_numpy("float32"),
                 valid=d["valid"].to_numpy(bool), house=d["house"].to_numpy("int16"),
                 utc=d["utc"].to_numpy(), local=pd.DatetimeIndex(d["local"]).to_numpy())


def sample_training_indices(s: Split, rng: np.random.Generator, neg_per_pos: float) -> np.ndarray:
    """All 'on' minutes plus a fresh random set of 'off' minutes, half of them hard negatives
    (aggregate > 1 kW: kettles, dishwashers, dryers, showers look most like a wash)."""
    ok = s.valid
    pos = np.flatnonzero(ok & (s.wm > ON_W))
    off = ok & (s.wm <= ON_W)
    hard = np.flatnonzero(off & (s.agg > 1000))
    easy = np.flatnonzero(off)
    n = int(len(pos) * neg_per_pos / 2)
    neg = np.r_[rng.choice(hard, min(n, len(hard)), replace=False), rng.choice(easy, n, replace=False)]
    idx = np.r_[pos, neg]
    rng.shuffle(idx)
    return idx


def windows(s: Split, idx: np.ndarray) -> np.ndarray:
    offs = np.arange(-HALF, HALF + 1)
    return s.agg[idx[:, None] + offs[None, :]]


# ---------- hand-built features for the gradient-boosting model ----------
SPANS = (5, 15, 31, 61, 99)


def features(s: Split) -> pd.DataFrame:
    """Centred rolling statistics of the aggregate, computed per house (padding stops leakage
    across houses because each house is followed by a zero block wider than the largest span)."""
    a = pd.Series(s.agg)
    f = {"agg": a, "d1_prev": a.diff(), "d1_next": a.shift(-1) - a}
    for w in SPANS:
        r = a.rolling(w, center=True, min_periods=1)
        f[f"mean_{w}"] = r.mean()
        f[f"std_{w}"] = r.std()
        f[f"max_{w}"] = r.max()
        f[f"min_{w}"] = r.min()
        f[f"above_min_{w}"] = a - f[f"min_{w}"]
        f[f"n_gt1800_{w}"] = (a > 1800).astype("float32").rolling(w, center=True, min_periods=1).sum()
        f[f"n_steps_{w}"] = (a.diff().abs() > 150).astype("float32").rolling(w, center=True, min_periods=1).sum()
    loc = pd.DatetimeIndex(s.local)
    f["hour"] = loc.hour + loc.minute / 60
    f["weekend"] = (loc.day_of_week >= 5).astype("float32")
    return pd.DataFrame(f).astype("float32")
