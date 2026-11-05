
import time

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv

from config import CLEAN_DIR, PROC_DIR, WM_CHANNEL


def load_clean_1min(house: int) -> pd.DataFrame:
    ch = WM_CHANNEL[house]
    tbl = pacsv.read_csv(
        CLEAN_DIR / f"CLEAN_House{house}.csv",
        convert_options=pacsv.ConvertOptions(include_columns=["Unix", "Aggregate", ch, "Issues"]),
    )
    df = tbl.to_pandas().rename(columns={ch: "wm"})
    df.index = pd.to_datetime(df.pop("Unix"), unit="s")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    out = df[["Aggregate", "wm"]].resample("1min").mean()
    out["issues"] = df["Issues"].resample("1min").max()
    out["n_readings"] = df["Aggregate"].resample("1min").count()
    out.index.name = "file_time"
    return out.astype("float32")


def main():
    for house in WM_CHANNEL:
        t = time.time()
        out = load_clean_1min(house)
        out.to_parquet(PROC_DIR / f"clean_1min_h{house}.parquet")
        print(f"house {house:2d}: {len(out):,} minutes, {out.index[0]} -> {out.index[-1]}, "
              f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
