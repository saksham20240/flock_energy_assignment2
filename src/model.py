"""Section 3: estimate washing-machine power from the household aggregate (1-minute).

Split by household (config.py): TRAIN homes fit the models, VAL homes choose the epoch and
the on/off gate threshold, TEST homes are touched once, at the end. No home is on two sides,
so the model can't memorise a test home's appliances, habits or base load.

Models
  baseline_off      always predicts 0 W
  baseline_profile  mean washing-machine power by local hour x weekday/weekend, from TRAIN homes
  lightgbm          gated model on hand-built rolling-window features
  seq2point_cnn     Seq2Point CNN (99-min window -> centre minute) with an on/off gate head
Both learned models output p(on) and power-if-on; prediction = power if p(on) > threshold else 0.
"""
import json
import sys
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import (EPOCHS, FIG_DIR, NEG_PER_POS, ON_W, PROC_DIR, SEED, TAB_DIR, TEST_HOUSES,
                    TRAIN_HOUSES, VAL_HOUSES, WINDOW)
from metrics import evaluate, minute_metrics
from nilm_data import Split, build_split, features, sample_training_indices, windows
from plotstyle import style

torch.manual_seed(SEED)
torch.set_num_threads(max(1, torch.get_num_threads()))
THRESHOLDS = np.round(np.arange(0.2, 0.96, 0.05), 2)


# ------------------------------------------------------------------ CNN
class Seq2Point(nn.Module):
    """Zhang et al. (2018) Seq2Point, slimmed for CPU, with a classification gate as in
    Shin et al. (2019) subtask-gated networks. Input channels: aggregate (kW) and aggregate
    minus the window minimum (removes each home's base load)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, 9, padding=4), nn.ReLU(),
            nn.Conv1d(32, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 48, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(48, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(12),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 12, 128), nn.ReLU(), nn.Dropout(0.2))
        self.power = nn.Linear(128, 1)
        self.gate = nn.Linear(128, 1)

    def forward(self, x):                                    # x: (B, WINDOW) in W
        x = x / 1000.0
        x = torch.stack([x, x - x.min(dim=1, keepdim=True).values], dim=1)
        z = self.fc(self.conv(x))
        return self.power(z).squeeze(1), self.gate(z).squeeze(1)


def cnn_predict(net, s: Split, idx, bs=8192):
    net.eval()
    pw, pr = np.empty(len(idx), "float32"), np.empty(len(idx), "float32")
    with torch.no_grad():
        for k in range(0, len(idx), bs):
            x = torch.from_numpy(windows(s, idx[k:k + bs]))
            a, b = net(x)
            pw[k:k + bs] = a.clamp(min=0).numpy() * 1000
            pr[k:k + bs] = torch.sigmoid(b).numpy()
    return pw, pr


def train_cnn(tr: Split, va: Split, rng):
    net = Seq2Point()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    bce, huber = nn.BCEWithLogitsLoss(), nn.HuberLoss(delta=0.5)
    va_idx = np.flatnonzero(va.valid)[::3]                   # every 3rd minute keeps per-epoch checks fast
    history, best, best_state = [], np.inf, None
    for ep in range(EPOCHS):
        t0 = time.time()
        net.train()
        idx = sample_training_indices(tr, rng, NEG_PER_POS)
        tot = 0.0
        for k in range(0, len(idx), 512):
            b = idx[k:k + 512]
            x = torch.from_numpy(windows(tr, b))
            y = torch.from_numpy(tr.wm[b]) / 1000
            on = (y * 1000 > ON_W).float()
            pw, lg = net(x)
            loss = bce(lg, on) + (huber(pw[on > 0], y[on > 0]) if on.sum() > 0 else 0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        pw, pr = cnn_predict(net, va, va_idx)
        nep = min(np.abs(np.where(pr > t, pw, 0) - va.wm[va_idx]).sum() for t in THRESHOLDS) / va.wm[va_idx].sum()
        history.append({"epoch": ep + 1, "train_loss": tot / len(idx), "val_best_nep": nep, "sec": time.time() - t0})
        print(f"  epoch {ep + 1}: loss {tot / len(idx):.4f}  val NEP {nep:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if nep < best:
            best, best_state = nep, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, pd.DataFrame(history)


# ------------------------------------------------------------------ LightGBM
def train_lgbm(tr: Split, va: Split, ftr, fva, rng):
    idx = sample_training_indices(tr, rng, NEG_PER_POS)
    on = (tr.wm[idx] > ON_W).astype(int)
    vidx = np.flatnonzero(va.valid)[::3]
    params = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    clf = lgb.LGBMClassifier(**params)
    clf.fit(ftr.iloc[idx], on, eval_X=(fva.iloc[vidx],), eval_y=((va.wm[vidx] > ON_W).astype(int),),
            callbacks=[lgb.early_stopping(100, verbose=False)])
    pos = idx[on == 1]
    vpos = vidx[va.wm[vidx] > ON_W]
    reg = lgb.LGBMRegressor(objective="huber", alpha=300, **params)
    reg.fit(ftr.iloc[pos], tr.wm[pos], eval_X=(fva.iloc[vpos],), eval_y=(va.wm[vpos],),
            callbacks=[lgb.early_stopping(100, verbose=False)])
    return clf, reg


def lgbm_predict(clf, reg, f, idx):
    X = f.iloc[idx]
    return np.clip(reg.predict(X), 0, None).astype("float32"), clf.predict_proba(X)[:, 1].astype("float32")


# ------------------------------------------------------------------ baselines
def profile_table(tr: Split):
    loc = pd.DatetimeIndex(tr.local[tr.valid])
    df = pd.DataFrame({"h": loc.hour, "we": loc.day_of_week >= 5, "wm": tr.wm[tr.valid]})
    return df.groupby(["h", "we"])["wm"].mean()


def profile_predict(tab, s: Split, idx):
    loc = pd.DatetimeIndex(s.local[idx])
    keys = pd.MultiIndex.from_arrays([loc.hour, loc.day_of_week >= 5])
    return tab.reindex(keys).to_numpy("float32")


# ------------------------------------------------------------------ evaluation helpers
def choose_threshold(pw, pr, y):
    rows = [{"threshold": t, "nep": np.abs(np.where(pr > t, pw, 0) - y).sum() / y.sum()} for t in THRESHOLDS]
    rows = pd.DataFrame(rows)
    return float(rows.loc[rows["nep"].idxmin(), "threshold"]), rows


def full_house_eval(s: Split, h: int, pred_full: np.ndarray):
    sel = np.flatnonzero(s.house == h)
    return evaluate(s.wm[sel], pred_full[sel], s.valid[sel], s.utc[sel], s.local[sel])


def main(retrain=False):
    style()
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    tr, va, te = build_split(TRAIN_HOUSES), build_split(VAL_HOUSES), build_split(TEST_HOUSES)
    print(f"data built in {time.time() - t0:.0f}s: train {tr.valid.sum():,} / val {va.valid.sum():,} / "
          f"test {te.valid.sum():,} valid minutes", flush=True)
    split_info = pd.DataFrame([
        {"split": name, "houses": ", ".join(map(str, hs)), "valid_minutes": int(s.valid.sum()),
         "on_minutes": int((s.valid & (s.wm > ON_W)).sum()),
         "period_utc": f"{pd.Timestamp(np.nanmin(s.utc[s.house > 0])):%Y-%m-%d} -> {pd.Timestamp(np.nanmax(s.utc[s.house > 0])):%Y-%m-%d}"}
        for name, hs, s in (("train", TRAIN_HOUSES, tr), ("validation", VAL_HOUSES, va), ("test", TEST_HOUSES, te))])
    split_info.to_csv(TAB_DIR / "s3_split.csv", index=False)
    print(split_info.to_string())

    preds = {}   # model -> full-length prediction array over te (0 where not scored)
    va_idx, te_idx = np.flatnonzero(va.valid), np.flatnonzero(te.valid)

    # Baselines
    preds["baseline_off"] = np.zeros(len(te.agg), "float32")
    tab = profile_table(tr)
    p = np.zeros(len(te.agg), "float32")
    p[te_idx] = profile_predict(tab, te, te_idx)
    preds["baseline_profile"] = p

    settings = {}
    # LightGBM
    t1 = time.time()
    ftr, fva, fte = features(tr), features(va), features(te)
    clf, reg = train_lgbm(tr, va, ftr, fva, rng)
    pw, pr = lgbm_predict(clf, reg, fva, va_idx)
    thr, grid_l = choose_threshold(pw, pr, va.wm[va_idx])
    pw, pr = lgbm_predict(clf, reg, fte, te_idx)
    p = np.zeros(len(te.agg), "float32")
    p[te_idx] = np.where(pr > thr, pw, 0)
    preds["lightgbm"] = p
    settings["lightgbm"] = {"threshold": thr, "clf_trees": clf.best_iteration_, "reg_trees": reg.best_iteration_,
                            "train_sec": round(time.time() - t1)}
    imp = pd.Series(clf.booster_.feature_importance("gain"), index=ftr.columns).sort_values(ascending=False)
    (imp / imp.sum()).round(4).to_csv(TAB_DIR / "s3_lightgbm_feature_importance.csv", header=["gain_share"])
    print("lightgbm", settings["lightgbm"], flush=True)

    lgb_va = lgbm_predict(clf, reg, fva, va_idx)
    lgb_te = (pw, pr)
    del ftr, fva, fte

    # CNN (training takes ~1 h on CPU; an existing checkpoint is reused unless --retrain)
    t1 = time.time()
    ckpt = PROC_DIR / "seq2point_cnn.pt"
    if ckpt.exists() and not retrain:
        net = Seq2Point()
        net.load_state_dict(torch.load(ckpt))
        hist = pd.read_csv(TAB_DIR / "s3_cnn_training_history.csv")
        print("cnn: reusing checkpoint", ckpt, flush=True)
    else:
        net, hist = train_cnn(tr, va, rng)
        hist.to_csv(TAB_DIR / "s3_cnn_training_history.csv", index=False)
        torch.save(net.state_dict(), ckpt)
    cnn_va = cnn_predict(net, va, va_idx)
    thr_c, grid_c = choose_threshold(*cnn_va, va.wm[va_idx])
    cnn_te = cnn_predict(net, te, te_idx)
    p = np.zeros(len(te.agg), "float32")
    p[te_idx] = np.where(cnn_te[1] > thr_c, cnn_te[0], 0)
    preds["seq2point_cnn"] = p
    prob = np.zeros(len(te.agg), "float32")
    prob[te_idx] = cnn_te[1]
    settings["seq2point_cnn"] = {"threshold": thr_c, "epochs": EPOCHS, "best_epoch": int(hist["val_best_nep"].idxmin() + 1),
                                 "window_min": WINDOW, "params": sum(q.numel() for q in net.parameters())}
    print("cnn", settings["seq2point_cnn"], flush=True)

    # Ensemble: weighted mean of the two models' p(on) and power; weight and threshold chosen on VAL only
    grid_e, best = [], None
    for w in (0.25, 0.5, 0.75):
        pw_v = w * cnn_va[0] + (1 - w) * lgb_va[0]
        pr_v = w * cnn_va[1] + (1 - w) * lgb_va[1]
        t, g = choose_threshold(pw_v, pr_v, va.wm[va_idx])
        grid_e.append(g.assign(cnn_weight=w))
        nep = g["nep"].min()
        if best is None or nep < best[0]:
            best = (nep, w, t)
    _, w, t = best
    p = np.zeros(len(te.agg), "float32")
    pr_e = w * cnn_te[1] + (1 - w) * lgb_te[1]
    p[te_idx] = np.where(pr_e > t, w * cnn_te[0] + (1 - w) * lgb_te[0], 0)
    preds["ensemble"] = p
    settings["ensemble"] = {"cnn_weight": w, "threshold": t}
    val_nep = {"lightgbm": grid_l["nep"].min(), "seq2point_cnn": grid_c["nep"].min(), "ensemble": best[0]}
    settings["val_nep"] = val_nep
    settings["final_model"] = min(val_nep, key=val_nep.get)   # picked on validation, before looking at test
    print("ensemble", settings["ensemble"], "val NEP", val_nep, "final:", settings["final_model"], flush=True)
    pd.concat([grid_l.assign(model="lightgbm"), grid_c.assign(model="seq2point_cnn"),
               pd.concat(grid_e).assign(model="ensemble")]).to_csv(TAB_DIR / "s3_threshold_search_val.csv", index=False)
    (TAB_DIR / "s3_model_settings.json").write_text(json.dumps(settings, indent=2, default=float))
    final = settings["final_model"]

    # ---- Test evaluation (only now are the test homes scored)
    rows, cyc_rows, daily_rows = [], [], []
    for name, pfull in preds.items():
        for h in TEST_HOUSES:
            m, ct, cp, daily = full_house_eval(te, h, pfull)
            rows.append({"model": name, "house": h, **m})
            if name == final:
                cyc_rows.append(ct.assign(house=h))
            daily_rows.append(daily.assign(model=name, house=h))
    by_house = pd.DataFrame(rows)
    by_house.round(4).to_csv(TAB_DIR / "s3_test_by_house.csv", index=False)
    pooled = []
    for name, pfull in preds.items():
        mm = minute_metrics(te.wm[te.valid], pfull[te.valid])
        bh = by_house[by_house["model"] == name]
        pooled.append({"model": name, **mm,
                       "cycle_recall": (bh["cycle_recall"] * bh["true_cycles"]).sum() / bh["true_cycles"].sum(),
                       "cycle_precision": (bh["cycle_precision"].fillna(0) * bh["pred_cycles"]).sum() / max(bh["pred_cycles"].sum(), 1),
                       "hourly_nep": bh["hourly_abs_err_wh"].sum() / bh["actual_wh"].sum(),
                       "daily_nep": bh["daily_abs_err_wh"].sum() / bh["actual_wh"].sum(),
                       "daily_median_abs_err_wh": bh["daily_median_abs_err_wh"].median(),
                       "daily_energy_corr_mean": bh["daily_energy_corr"].mean(),
                       "houses_beating_off_on_nep": int((bh["nep"] < 1).sum())})
    pooled = pd.DataFrame(pooled)
    r, p_ = pooled["cycle_recall"], pooled["cycle_precision"]
    pooled["cycle_f1"] = 2 * r * p_ / (r + p_).replace(0, np.nan)
    pooled.round(4).to_csv(TAB_DIR / "s3_test_summary.csv", index=False)
    print(pooled.round(3).to_string())
    print(by_house.round(3).to_string())

    cyc = pd.concat(cyc_rows, ignore_index=True)
    cyc["has_heater"] = cyc["heater_min"] > 0
    cyc.drop(columns=["i0", "i1"]).to_csv(TAB_DIR / "s3_final_test_cycles.csv", index=False)
    recall_by = cyc.groupby(["house", "has_heater"])["found"].agg(["mean", "size"]).round(3)
    recall_by.to_csv(TAB_DIR / "s3_final_recall_by_heater.csv")
    print(recall_by.to_string())
    daily = pd.concat(daily_rows)
    daily.to_csv(TAB_DIR / "s3_test_daily_energy.csv")

    out = pd.DataFrame({"utc": te.utc, "local": te.local, "house": te.house, "valid": te.valid,
                        "agg": te.agg, "wm": te.wm, "cnn_prob": prob, **preds, "final": preds[final]})
    out = out[out["house"] > 0]
    out.to_parquet(PROC_DIR / "test_predictions.parquet")

    plot_actual_vs_pred(out, final)
    plot_daily(daily, final)
    plot_summary(by_house)


def plot_actual_vs_pred(out, final):
    """For each test home, the 36 h around a cycle near the home's median cycle energy."""
    cyc = pd.read_csv(TAB_DIR / "s3_final_test_cycles.csv", parse_dates=["start", "end"])
    fig, axes = plt.subplots(len(TEST_HOUSES), 1, figsize=(13, 3.0 * len(TEST_HOUSES)))
    for ax, h in zip(axes, TEST_HOUSES):
        c = cyc[cyc["house"] == h]
        row = c.iloc[(c["energy_wh"] - c["energy_wh"].median()).abs().argmin()]
        d = out[out["house"] == h].set_index("utc")
        w = d[row["start"] - pd.Timedelta("16h"): row["start"] + pd.Timedelta("20h")]
        x = w["local"]
        ax.plot(x, w["agg"], color="0.75", lw=0.7, label="aggregate (input)")
        ax.fill_between(x, 0, w["wm"].fillna(0), color="#1f6f8b", alpha=0.45, label="actual washing machine")
        ax.plot(x, w["seq2point_cnn"], color="#c0392b", lw=1.0, label="Seq2Point CNN")
        ax.plot(x, w["lightgbm"], color="#f39c12", lw=0.9, ls="--", label="LightGBM")
        if final == "ensemble":
            ax.plot(x, w["ensemble"], color="#2c3e50", lw=1.0, ls=":", label="ensemble (final)")
        ax.set_ylim(0, max(3000, w["wm"].max() * 1.1))
        ax.set_ylabel("W")
        ax.set_title(f"Test house {h} (never seen in training), local time", loc="left")
    axes[0].legend(ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s3_01_actual_vs_predicted.png")
    plt.close(fig)


def plot_daily(daily, final):
    d = daily[daily["model"] == final]
    fig, axes = plt.subplots(1, len(TEST_HOUSES), figsize=(15, 3.8), sharex=True, sharey=True)
    for ax, h in zip(axes, TEST_HOUSES):
        x = d[d["house"] == h]
        ax.scatter(x["y"] / 1000, x["p"] / 1000, s=8, alpha=0.5, color="#1f6f8b")
        lim = max(x["y"].max(), x["p"].max()) / 1000 * 1.05
        ax.plot([0, lim], [0, lim], color="0.4", lw=0.8, ls=":")
        ax.set_title(f"House {h}: r = {x['y'].corr(x['p']):.2f}")
        ax.set_xlabel("actual daily WM kWh")
    axes[0].set_ylabel("predicted daily WM kWh")
    fig.suptitle(f"Daily washing-machine energy on the test homes, final model = {final} (dotted = perfect)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s3_02_daily_energy_scatter.png")
    plt.close(fig)


def plot_summary(by_house):
    names = ["baseline_off", "baseline_profile", "lightgbm", "seq2point_cnn", "ensemble"]
    cols = ["#95a5a6", "#bdc3c7", "#f39c12", "#c0392b", "#2c3e50"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, lab in ((axes[0], "hourly_nep", "hourly-energy NEP (lower is better; always-off = 1)"),
                            (axes[1], "on_f1", "on/off F1 per minute"),
                            (axes[2], "cycle_recall", "share of true cycles found")):
        piv = by_house.pivot(index="house", columns="model", values=metric)[names]
        piv.plot.bar(ax=ax, color=cols, width=0.8, legend=False)
        ax.set_title(lab)
        ax.set_xlabel("test house")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s3_03_model_vs_baselines.png")
    plt.close(fig)


if __name__ == "__main__":
    main(retrain="--retrain" in sys.argv)
