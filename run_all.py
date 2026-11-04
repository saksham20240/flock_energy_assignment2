"""Run the whole pipeline end to end, in order.

    python run_all.py              # everything (~1.5 h on a 14-core laptop CPU, most of it CNN training)
    python run_all.py --skip-clock # skip clock_check.py (reads all 8 GB of raw files, ~3 min)
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = [
    ("prepare_clean.py", "official cleaned CSVs -> 1-minute parquet"),
    ("clock_check.py", "verify the clock correction on all 18 homes"),
    ("clean_raw.py", "Section 1: raw House 1 inspection and cleaning"),
    ("eda.py", "Section 2: washing-machine EDA"),
    ("model.py", "Section 3: baselines, LightGBM, Seq2Point CNN, test evaluation"),
    ("failures.py", "Section 3: failure analysis"),
    ("coarse.py", "Section 4: 1 vs 15 vs 30-minute data experiment"),
    ("opportunity.py", "Section 4: evening-peak washing per home, model vs sub-meter"),
]

for script, what in STEPS:
    if script == "clock_check.py" and "--skip-clock" in sys.argv:
        continue
    print(f"\n=== {script}: {what} ===", flush=True)
    subprocess.run([sys.executable, "-u", script], check=True, cwd=SRC)
