"""Paths, house lists and every tunable threshold used in the pipeline, in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "REFIT_RAW_081116"
CLEAN_DIR = ROOT / "CLEAN_REFIT_081116"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
for _d in (PROC_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# Washing-machine channel per house, from docs/CLEAN_READ_ME_081116.txt.
# House 4 has two washing machines on separate IAMs (the aggregate contains both),
# so a single-target model would be trained on a half-labelled signal: excluded.
# House 12 has no washing-machine channel. Washer-dryers (H1 App4, H8 App3, H9 App2,
# H18 App4) are a different appliance and are not used as the target.
WM_CHANNEL = {
    1: "Appliance5", 2: "Appliance2", 3: "Appliance6", 5: "Appliance3",
    6: "Appliance2", 7: "Appliance5", 8: "Appliance4", 9: "Appliance3",
    10: "Appliance5", 11: "Appliance3", 13: "Appliance3", 15: "Appliance3",
    16: "Appliance5", 17: "Appliance4", 18: "Appliance5", 19: "Appliance2",
    20: "Appliance4", 21: "Appliance3",
}
# Readme: "House 13 ... Washing Machine (Changes 25 Mar 2015)" -> keep data before it.
WM_END = {13: "2015-03-25"}

# ---- Section 1: raw House 1 -------------------------------------------------
RAW_HOUSE = 1
RAW_PLOT_CHANNELS = {"Aggregate": "Aggregate", "Appliance1": "Fridge",
                     "Appliance5": "Washing machine", "Appliance6": "Dishwasher"}
MAX_AGG_W = 20_000      # 100 A main fuse x 230 V ~ 23 kW: anything above is a sensor error
MAX_IAM_W = 4_000       # a 13 A plug socket gives at most ~3.1 kW; the official cleaner uses 4 kW too
SHORT_GAP_MIN = 5       # gaps up to this many 1-min steps are forward-filled (load held)
                        # longer gaps are flagged and left NaN

# ---- Section 2: washing-machine cycles --------------------------------------
ON_W = 20               # a minute is "active" if WM power > 20 W (standby/off is 0-5 W)
MAX_PAUSE_MIN = 30      # active minutes separated by <=30 min belong to one cycle (soak/pause phases)
MIN_CYCLE_MIN = 10      # a cycle must last >= 10 min ...
MIN_CYCLE_WH = 20       # ... and use >= 20 Wh, otherwise it's a door/display blip, not a wash
HEATER_W = 1_000        # a minute above 1 kW is the water-heating element

# ---- Section 3: model --------------------------------------------------------
# Split by household. Test homes are unseen in training; validation homes pick
# hyper-parameters/thresholds; training homes fit the model.
TEST_HOUSES = [2, 9, 17, 21]   # include known confusers: dishwasher, washer-dryer, tumble dryers, cold washes
VAL_HOUSES = [15, 20]
TRAIN_HOUSES = [h for h in WM_CHANNEL if h not in TEST_HOUSES + VAL_HOUSES]
WINDOW = 99             # Seq2Point input window in minutes (centre minute is predicted)
AGG_CLIP_W = 8_000      # aggregate clipped before scaling (rare showers/EV-type spikes)
WM_CLIP_W = 3_000       # target clipped (a washing machine's heater is ~2.0-2.3 kW)
EPOCHS = 12
NEG_PER_POS = 2.5       # off-minutes sampled per on-minute each epoch (half of them "hard": aggregate > 1 kW)

# Other IAM channels of the test homes, used only to explain false alarms (from the readme)
OTHER_APPLIANCES = {
    2: {1: "Fridge-freezer", 3: "Dishwasher", 4: "TV", 5: "Microwave", 6: "Toaster", 7: "Hi-Fi", 8: "Kettle", 9: "Fan"},
    9: {1: "Fridge-freezer", 2: "Washer-dryer", 4: "Dishwasher", 5: "TV", 6: "Microwave", 7: "Kettle", 8: "Hi-Fi", 9: "Electric heater"},
    17: {1: "Freezer", 2: "Fridge-freezer", 3: "Tumble dryer", 5: "Computer", 6: "TV", 7: "Microwave", 8: "Kettle", 9: "TV (bedroom)"},
    21: {1: "Fridge-freezer", 2: "Tumble dryer", 4: "Dishwasher", 5: "Food mixer", 6: "TV", 7: "Kettle", 8: "Vivarium", 9: "Pond pump"},
}
