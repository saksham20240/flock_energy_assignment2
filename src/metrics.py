"""Evaluation metrics. All are computed on valid minutes only.

minute level  MAE (W); NEP = sum|err| / sum(actual) (0 = perfect, 1 = always-off);
              on/off precision/recall/F1 with the same 20 W threshold as the EDA
energy        total energy error  = (pred - actual) / actual over the whole test period;
              hourly / daily NEP  = NEP after summing energy to local hours / days;
              daily abs error     = median over days of |pred - actual| daily Wh
cycle level   cycles are detected in actual and predicted power with the Section 2 rule;
              a true cycle is 'found' if any predicted cycle overlaps it (recall), a predicted
              cycle is 'correct' if it overlaps any true cycle (precision)
"""
import numpy as np
import pandas as pd

from common import detect_cycles
from config import ON_W


def minute_metrics(y, p):
    on_t, on_p = y > ON_W, p > ON_W
    tp = (on_t & on_p).sum()
    prec = tp / max(on_p.sum(), 1)
    rec = tp / max(on_t.sum(), 1)
    return {"mae_w": np.abs(p - y).mean(), "nep": np.abs(p - y).sum() / y.sum(),
            "total_energy_err_pct": 100 * (p.sum() - y.sum()) / y.sum(),
            "on_precision": prec, "on_recall": rec, "on_f1": 2 * prec * rec / max(prec + rec, 1e-9)}


def overlaps(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """For each interval in a, does it overlap any interval in b? (i0/i1 inclusive positions)"""
    if len(a) == 0:
        return np.zeros(0, bool)
    if len(b) == 0:
        return np.zeros(len(a), bool)
    b = b.sort_values("i0")
    j = np.searchsorted(b["i0"].values, a["i1"].values, side="right") - 1   # last b starting <= a.end
    ok = j >= 0
    res = np.zeros(len(a), bool)
    # b intervals don't overlap each other, so the candidate is b[j] (or earlier ones ending later is impossible)
    res[ok] = b["i1"].values[j[ok]] >= a["i0"].values[ok]
    return res


def cycle_metrics(y: pd.Series, p: pd.Series):
    ct, cp = detect_cycles(y), detect_cycles(p)
    found = overlaps(ct, cp)
    correct = overlaps(cp, ct)
    rec = found.mean() if len(ct) else np.nan
    prec = correct.mean() if len(cp) else np.nan
    return ({"true_cycles": len(ct), "pred_cycles": len(cp), "cycle_recall": rec, "cycle_precision": prec,
             "cycle_f1": 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0},
            ct.assign(found=found), cp.assign(correct=correct))


def daily_metrics(y: pd.Series, p: pd.Series, local: pd.Series):
    """Energy errors after summing to local hours and days: what a bill or an app would show.
    A wash predicted a few minutes early is a large per-minute error but no hourly error."""
    m = pd.DataFrame({"y": y.values, "p": p.values}, index=pd.DatetimeIndex(local.values))
    hr = m.resample("1h").sum() / 60
    d = m.resample("1D").sum() / 60
    out = {"actual_wh": d["y"].sum(),
           "hourly_abs_err_wh": (hr["p"] - hr["y"]).abs().sum(),
           "daily_abs_err_wh": (d["p"] - d["y"]).abs().sum()}
    out["hourly_nep"] = out["hourly_abs_err_wh"] / out["actual_wh"]
    out["daily_nep"] = out["daily_abs_err_wh"] / out["actual_wh"]
    d = d[d["y"] + d["p"] > 0]
    out.update({"daily_median_abs_err_wh": (d["p"] - d["y"]).abs().median(),
                "daily_energy_corr": d["y"].corr(d["p"])})
    return out, d


def evaluate(y: np.ndarray, p: np.ndarray, valid: np.ndarray, utc, local):
    """y, p: per-minute arrays over one house's contiguous minutes; invalid minutes are
    excluded from minute metrics and treated as 'off' for cycle detection."""
    m = minute_metrics(y[valid], p[valid])
    ys = pd.Series(np.where(valid, y, 0.0), index=pd.DatetimeIndex(utc))
    ps = pd.Series(np.where(valid, p, 0.0), index=pd.DatetimeIndex(utc))
    c, ct, cp = cycle_metrics(ys, ps)
    dm, daily = daily_metrics(ys, ps, pd.Series(local))
    return {**m, **c, **dm}, ct, cp, daily
