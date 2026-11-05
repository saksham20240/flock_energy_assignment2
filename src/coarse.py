
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import load_house
from config import SEED, TAB_DIR, TEST_HOUSES, TRAIN_HOUSES, VAL_HOUSES

RES = (1, 15, 30)


def house_frame(h, res):
    df = load_house(h)
    agg = df["agg"].where(df["valid"])
    wm = df["wm"].where(df["valid"])
    g = pd.DataFrame({"agg": agg, "wm": wm, "ok": df["valid"].astype(float)}).resample(f"{res}min")
    d = g.mean()
    d = d[d["ok"] >= 0.9].drop(columns="ok")
    d = d.reindex(pd.date_range(d.index[0], d.index[-1], freq=f"{res}min"))
    a = d["agg"].ffill(limit=2).fillna(0)
    f = {"agg": a, "lag1": a.shift(1), "lead1": a.shift(-1), "lag2": a.shift(2), "lead2": a.shift(-2)}
    span_min = (15, 45, 120, 240)
    for m in span_min:
        k = max(3, int(round(m / res)) | 1)
        r = a.rolling(k, center=True, min_periods=1)
        f[f"mean_{m}m"], f[f"max_{m}m"], f[f"std_{m}m"], f[f"min_{m}m"] = r.mean(), r.max(), r.std(), r.min()
    loc = d.index.tz_localize("UTC").tz_convert("Europe/London")
    f["hour"] = loc.hour + loc.minute / 60
    f["weekend"] = (loc.day_of_week >= 5).astype(float)
    X = pd.DataFrame(f, index=d.index)
    keep = d["wm"].notna()
    return X[keep], d.loc[keep, "wm"], loc[keep.values]


def stack(houses, res):
    parts = [house_frame(h, res) + (h,) for h in houses]
    X = pd.concat([p[0] for p in parts])
    y = pd.concat([p[1] for p in parts])
    local = np.concatenate([p[2].tz_localize(None) for p in parts])
    house = np.concatenate([np.full(len(p[1]), p[3]) for p in parts])
    return X, y.to_numpy(), local, house


def main():
    rows = []
    for res in RES:
        Xtr, ytr, _, _ = stack(TRAIN_HOUSES, res)
        Xva, yva, _, _ = stack(VAL_HOUSES, res)
        Xte, yte, lte, hte = stack(TEST_HOUSES, res)
        if res == 1:                                         # 1-min: subsample train rows to keep the run short
            take = np.random.default_rng(SEED).random(len(ytr)) < 0.25
            Xtr, ytr = Xtr[take], ytr[take]
        reg = lgb.LGBMRegressor(objective="huber", alpha=300, n_estimators=2000, learning_rate=0.05,
                                num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, random_state=SEED, verbose=-1)
        reg.fit(Xtr, ytr, eval_X=(Xva,), eval_y=(yva,), callbacks=[lgb.early_stopping(100, verbose=False)])
        p = np.clip(reg.predict(Xte), 0, None)
        for h in TEST_HOUSES + ["all"]:
            m = np.ones(len(yte), bool) if h == "all" else hte == h
            y, q = yte[m], p[m]
            # daily energy per home (for "all", per-home days are pooled, never summed across homes)
            d = pd.concat([pd.DataFrame({"y": yte[hte == k], "p": p[hte == k]}, index=pd.DatetimeIndex(lte[hte == k]))
                           .resample("1D").sum() * res / 60 for k in (TEST_HOUSES if h == "all" else [h])])
            d = d[d["y"] + d["p"] > 0]
            # interval-level "was the machine running": mean WM power over the interval > 20 W
            on_t, on_p = y > 20, q > 20
            tp = (on_t & on_p).sum()
            pr, rc = tp / max(on_p.sum(), 1), tp / max(on_t.sum(), 1)
            rows.append({"resolution_min": res, "house": h, "intervals": int(m.sum()),
                         "nep": np.abs(q - y).sum() / y.sum(),
                         "interval_on_f1": 2 * pr * rc / max(pr + rc, 1e-9),
                         "daily_median_abs_err_wh": (d["p"] - d["y"]).abs().median(),
                         "daily_energy_corr": d["y"].corr(d["p"]),
                         "total_energy_err_pct": 100 * (q.sum() - y.sum()) / y.sum(),
                         "wm_share_of_interval_energy_when_on": np.median(y[on_t] / np.maximum(Xte["agg"].to_numpy()[m][on_t], 1))})
        print(pd.DataFrame(rows).query("resolution_min == @res").round(3).to_string(), flush=True)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(TAB_DIR / "s4_resolution_experiment.csv", index=False)


if __name__ == "__main__":
    main()
