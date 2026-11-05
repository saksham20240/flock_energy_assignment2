# REFIT: data cleaning and washing-machine disaggregation

Take-home assignment for Flock Energy. A reproducible pipeline that:

1. cleans raw REFIT House 1 to 1-minute resolution;
2. analyses washing-machine use in the official cleaned data for **18 homes**;
3. trains and evaluates models that estimate washing-machine power from the whole-home signal **in homes the model has never seen**;
4. tests what is lost with 15- and 30-minute smart-meter data.

| Document | Contents |
|---|---|
| [DATA.md](DATA.md) | Raw-data coverage, quality findings, cleaning rules, the clock problem, remaining concerns |
| [RESULTS.md](RESULTS.md) | Washing-machine EDA, model results, validation design, failure analysis, interpretation |
| [Recommendation.md](Recommendation.md) | One-page utility summary and the Section 4 answers |

## Headline findings

- **The official cleaned REFIT files mix two clocks.** They are UTC until about 1 Oct 2014 and UK wall clock afterwards, which the readme does not mention. I proved this for all 18 homes by aligning them with the raw data and corrected it ([DATA.md §4](DATA.md#4-the-official-cleaned-files-mix-two-clocks-undocumented)). Without the fix, summer-2015 hour-of-day results are an hour off.
- **A simple raw-data pipeline reproduces the official cleaning.** The raw House 1 files contain 596k duplicate rows, 49k conflicting duplicate timestamps, sensor overflow values (4096 to 98301 W), a repeated hour at the October clock change and a 3-month overlap between Part1 and Part2. After cleaning, my 1-minute aggregate matches the official cleaned file at **r = 0.996 (MAE 2.7 W)**. Unlike the official file, long outages (374 of them, 1,904 h) are flagged instead of forward-filled.
- **Washing is mostly about heating water.** I detected 5,941 cycles in 18 homes. A median wash lasts 71 min and uses 394 Wh, and **76% of that energy is the heater**. Energy per wash varies 8× between homes (89–693 Wh). Some homes always wash cold, and one home washes cold only in summer.
- **20% of washing energy falls in the 16:00–20:00 network peak.** The share ranges from 1% to 43% per home, so targeting the right homes matters more than averages.
- **On 4 homes the model never saw**, the final model (Seq2Point CNN + LightGBM ensemble, chosen on validation homes) gets total washing energy right to within **+0.3%**, finds **83% of all washes and 89% of heated ones**, and cuts **daily energy error by 35%** compared with always-off. Both baselines do far worse: the usage profile overestimates energy by 52%.
- **Where it fails, honestly:**
  - Minute-level timing is no better than always-off; the model is useful per hour or day.
  - About half of predicted washes are false. 45% of those coincide with a kettle or dishwasher, and most of the rest with unmetered loads such as cooking or showers.
  - Only 49% of cold washes are found.
  - With 15- or 30-minute data, detection falls by about 40–50% ([Recommendation.md](Recommendation.md)).

## Setup and run

Tested on Windows 11, Python 3.14, CPU only (14 cores, 16 GB RAM).

1. Download and unpack the two archives into the project root, so that `REFIT_RAW_081116/` and `CLEAN_REFIT_081116/` sit next to this README (about 14 GB unpacked):
   - [raw REFIT](https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z)
   - [cleaned REFIT](https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z)

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Install the dependencies:

```bash
.venv/Scripts/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

4. Run the pipeline:

```bash
.venv/Scripts/python run_all.py
```

On Linux or macOS, use `.venv/bin/python`. Every script also runs on its own from `src/`, in the order below. All thresholds, house lists and the split are in [`src/config.py`](src/config.py). Seeds are fixed (`SEED = 42`).

| Step | Script | Produces |
|---|---|---|
| 0 | `prepare_clean.py` | `data/processed/clean_1min_h*.parquet` (1-minute means, file clock) |
| 0b | `clock_check.py` | `outputs/tables/clock_check.csv`: proof of the clock correction for 18 homes |
| 1 | `clean_raw.py` | Section 1 tables `s1_*`, figures `s1_01`, `s1_02`, `data/processed/raw_house1_cleaned_1min.parquet` |
| 2 | `eda.py` | Section 2 tables `s2_*`, figures `s2_01` to `s2_05` |
| 3 | `model.py` | Section 3 tables `s3_*`, figures `s3_01` to `s3_03`, test-set predictions |
| 3b | `failures.py` | False-alarm sources, recall breakdown, figure `s3_04` |
| 4 | `coarse.py` | `s4_resolution_experiment.csv` (1 vs 15 vs 30-minute data) |
| 4b | `opportunity.py` | `s4_per_home_opportunity.csv`, `s4_model_peak_share_check.csv` |

Shared code: `clock.py` (clock correction), `common.py` (loading, valid-minute mask, cycle detector), `nilm_data.py` (windows, sampling, features), `metrics.py`.

## Assumptions and preprocessing decisions

- **Appliance mapping** comes from the [official readme](docs/CLEAN_READ_ME_081116.txt).
  - House 4 (two washing machines) and House 12 (none) are excluded.
  - House 13 is cut at 25 Mar 2015, when its washing-machine channel changes.
  - Washer-dryers are not used as the target.
- **Clock.** The raw files are UK wall clock. The cleaned files are UTC until ~1 Oct 2014 and wall clock after that, which the readme doesn't mention. Everything here is moved to true UTC, and hour-of-day and day-of-week use local UK time ([DATA.md §4](DATA.md#4-the-official-cleaned-files-mix-two-clocks-undocumented)).
- **1-minute resampling** takes the mean of the readings in each minute. A minute with no readings is left empty.
- **Raw gaps of 5 minutes or less** are forward-filled, because the logger writes only when a value changes. Longer gaps are flagged and left NaN.
- **Out-of-range values become NaN**, not 0: aggregate above 20 kW or at most 0, and IAM readings above 4 kW. A spike is unknown, not "off".
- **Valid minutes** (used for training and scoring) have an aggregate and a washing-machine value present, have `Issues` = 0, and are not part of a flat-lined stretch. An aggregate that holds exactly the same value for 60 minutes or more is the official forward-fill over an outage.
- **Active-cycle rule.** A minute counts as active above 20 W. Active minutes less than 30 minutes apart belong to the same wash. A cycle must last at least 10 minutes and use at least 20 Wh. The cycle count changes by −7% to +4% for any threshold from 10 to 50 W and any pause from 20 to 60 minutes.

## Validation design

- **Split by household, not by time or at random.**
  - 12 train homes: 1, 3, 5, 6, 7, 8, 10, 11, 13, 16, 18, 19.
  - 2 validation homes: 15, 20.
  - 4 test homes: 2, 9, 17, 21.
- **Why this avoids leakage.** A random split would put neighbouring minutes, which share most of their 99-minute input window, on both sides, so the model would score well by memorising. Even a time split inside one home lets the model learn that home's own machine and habits. A utility wants the opposite: a model that works in a home with no sub-meter.
- **What the validation homes choose:** the training epoch (early stopping), the LightGBM tree counts and the on/off gate threshold. The test homes are scored once, at the end.
- **Test homes were chosen to be hard.** They include a dishwasher (H2, H21), a washer-dryer next to the washing machine (H9), tumble dryers (H17, H21) and seasonal cold washing (H21).

## Models

| Model | What it does |
|---|---|
| `baseline_off` | Always predicts 0 W. It scores NEP = 1 by definition. |
| `baseline_profile` | Mean washing-machine power for each local hour, weekday vs weekend, learned from the train homes |
| `lightgbm` | Gated LightGBM on 40 rolling-window features (means, maxima, standard deviations, step counts, minutes above 1.8 kW over 5 to 99-minute windows, plus hour of day). A classifier predicts p(on) and a Huber regressor predicts power when on. |
| `seq2point_cnn` | Seq2Point CNN (Zhang et al., 2018). A 99-minute window of the aggregate predicts the washing-machine power at the centre minute. It has two output heads, power and p(on), in the style of subtask-gated networks (Shin et al., 2019). |
| `ensemble` (**final**) | 0.75 × CNN + 0.25 × LightGBM. The weight, the threshold and the choice of final model were made on the validation homes, before the test homes were scored. |

Both learned models output `power if p(on) > threshold else 0`. The threshold minimises NEP on the validation homes.

**Training sampling.** Each epoch uses every "on" minute plus 2.5 times as many "off" minutes, freshly resampled. Half of the off minutes are hard negatives where the aggregate is above 1 kW (kettles, dishwashers, showers). Without this, 97% of minutes are off and the model learns to predict zero.

## Trade-offs

- **1-minute resolution and a small 99-minute window.** Seq2Point papers use about 600 samples at 8 s (~80 minutes). A 99-minute window at 1 minute covers a whole heater phase plus context, and it trains on a laptop CPU in under an hour.
- **Two learned models, not one.** LightGBM is faster and each of its features can be explained. The CNN learns the cycle shape itself. Comparing them shows how much the extra complexity buys.
- **Threshold tuned for energy (NEP), not for cycle F1.** A utility mostly wants energy per home. A different threshold would trade missed cycles against false alarms (RESULTS.md shows the curve).
- **One house split, not full cross-validation over homes.** CNN training costs about 1 hour on CPU, so the spread across the 4 test homes stands in for a variance estimate. Leave-one-home-out is the first thing to add with a GPU.
- **Reusing the CNN checkpoint.** `model.py` reuses `data/processed/seq2point_cnn.pt` if it exists. Pass `--retrain` to train it again. The epoch times in `s3_cnn_training_history.csv` include periods when the laptop was asleep. A clean epoch takes about 2.5–4.5 minutes.

## Research used

| Paper | What I took from it |
|---|---|
| Murray, Stankovic & Stankovic (2017), *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study*, Scientific Data 4:160122 | Dataset description, channel semantics, known gaps |
| Zhang, Zhong, Wang, Goddard & Sutton (2018), *Sequence-to-point learning with neural networks for non-intrusive load monitoring*, AAAI (arXiv:1612.09106) | Predicting the centre point of a window, training on some REFIT homes and testing on an unseen one, MAE and SAE metrics |
| Shin, Joo, Ko & Rhee (2019), *Subtask gated networks for non-intrusive load monitoring*, AAAI | The on/off gate head multiplied with the power head |
| Kelly & Knottenbelt (2015), *Neural NILM: Deep neural networks applied to energy disaggregation*, BuildSys (arXiv:1507.06594) | Balanced sampling of activations and non-activations; testing on unseen houses |
| D'Incecco, Squartini & Zhong (2020), *Transfer learning for non-intrusive load monitoring*, IEEE Trans. Smart Grid | Evidence that cross-house generalisation is the hard part; washing machines transfer worst |
| Makonin & Popowich (2015), *Nonintrusive load monitoring (NILM) performance evaluation*, Energy Efficiency | Reporting energy-based and on/off metrics together, never accuracy alone |

## AI tools used

This project was built using some help from claude.
 Every number in the documents comes from a CSV in `outputs/tables/`, and the key findings are backed by checks inside the pipeline:
- the clock problem: `clock_check.csv`, 18 homes;
- duplicate and sentinel values: `s1_raw_checks.csv`;
- agreement with the official cleaned data: `s1_vs_official.csv`.

## Possible improvements

1. **Leave-one-home-out cross-validation** over all 18 homes, which needs a GPU. Two validation homes gave an optimistic NEP (0.53 against 1.03 on test).
2. **Fine-tuning on a few labelled weeks from the target home**, or synthetic training mixes (known washing-machine cycles added to other homes' aggregates). Cross-home transfer is the weak point.
3. **A joint dishwasher and washing-machine model**, so each 2 kW heater block is assigned to one appliance. Dishwashers cause 20% of false alarms.
4. **Cycle-level post-processing with duration and energy priors** from the EDA, to remove short kettle- and oven-triggered "washes".
5. **Better coverage of cold and eco washes**, through a dedicated class or synthetic examples.
6. **Calibrated uncertainty**, such as "3–4 washes this week", for customer-facing use.
