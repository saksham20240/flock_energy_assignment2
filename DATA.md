# DATA.md — coverage, quality findings and cleaning decisions

## 1. Dataset and files used

**REFIT Electrical Load Measurements** (University of Strathclyde; Murray et al., 2017). 20 UK homes in the Loughborough area. Each home has one whole-house current clamp ("Aggregate") and 9 plug-level monitors (IAMs, "Appliance1-9"). Units are **active power in watts**, and the nominal sampling interval is **~8 s**. The [official readme](docs/CLEAN_READ_ME_081116.txt) maps channels to appliances.

| Use | Files | Homes | Period |
|---|---|---|---|
| Section 1, raw cleaning | `REFIT_RAW_081116/RAW_House1_Part1.csv`, `RAW_House1_Part2.csv` | House 1 | 9 Oct 2013 – 10 Jul 2015 |
| Sections 2–3, EDA and model | `CLEAN_REFIT_081116/CLEAN_House{N}.csv` | 18 homes with one washing-machine channel: 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21 | Sep 2013 – Jul 2015 (each home's range is in `outputs/tables/s2_availability.csv`) |
| Clock verification | Raw `Time`/`Aggregate` of the same 18 homes | 18 | Jul 2014, May 2015 |

**Excluded homes:**
- House 4 has two washing machines on separate channels. The aggregate contains both, so a single-machine target would be half-labelled.
- House 12 has no washing-machine channel.
- House 13 is kept only up to 25 Mar 2015, because the readme says its washing-machine channel changes then.
- Washer-dryers (H1 App4, H8 App3, H9 App2, H18 App4) are a different appliance and are never used as the target.

**Documented limitations (readme):**
- Sensors are not synchronised. Each reading may be up to 6–8 s old.
- Rows are written only when a load changes.
- In Part1, a sensor that was unavailable reads 0. Part2 uses NaN, so 0 and "missing" can't be told apart in Part1.
- Occupants may have moved IAMs between appliances.
- The cleaned files forward-fill NaNs and replace IAM spikes above 4 kW with 0. They also add an `Issues` flag where the sum of IAMs exceeds the aggregate.

## 2. Raw House 1: what the inspection found

Full tables: `outputs/tables/s1_raw_checks.csv`, `s1_channel_quality.csv`, `s1_cleaning_log.csv`, `s1_gap_handling.csv`, `s1_aggregate_outages.csv`.

| Check | Part1 (Oct 2013 – Oct 2014) | Part2 (Jun 2014 – Jul 2015) | What it means |
|---|---|---|---|
| Rows | 4,377,877 | 4,140,886 | |
| Timestamp parsing | 0 failures; `Unix` and `Time` agree within 1 s on 100% of rows | same | `Unix` is **not** true Unix time. It encodes the wall-clock `Time` as if it were UTC. |
| Ordering | 2 backward steps: **27 Oct 2013 01:59:59 → 01:00:01** (−3,598 s) and a 1 s step on 15 Aug 2014 | none | The repeated hour proves the raw clock is **UK wall clock** (BST in summer). |
| Exact duplicate rows | 595,935 (13.6%) | 0 | The logger wrote the same row twice. |
| Duplicate timestamps | 645,514; **48,841 with conflicting values** (spread usually 0–3 W) | 0 | |
| Interval | median 5 s, p99 15 s | median 7 s, p99 9 s | Irregular, as the readme describes |
| Gaps > 1 min / > 1 h / > 1 day | 780 / 14 / 3 | 3,603 / 19 / 10 | |
| Aggregate > 20 kW | 53 rows (mostly exactly **24,000 W**, a ceiling value) | 74 rows | A 100 A supply gives at most ~23 kW |
| Aggregate = 0 | 12 rows | 0 | A home never draws 0 W (fridge, standby), so these are dropouts |
| IAM > 4 kW | 3,038 cells | 911 cells | Values are **4096, 8192, 16384, 32767, 65535, 98301 W**: powers of two and bit-overflow codes, not loads. A 13 A plug socket gives at most ~3.1 kW. |
| Negative values | 0 | 0 | |
| NaN cells | 0 | 5,284,083 | Part2 marks unavailable sensors as NaN |
| Sum of IAMs > aggregate + 50 W | 0.96% of rows | 0.28% | Sensor asynchrony or a mis-scaled channel |

**The two parts overlap from 27 Jun to 1 Oct 2014 (137,011 minutes).** They are on the same clock (best lag 0 min). They agree only moderately per minute (r = 0.80, MAE 58 W), and closely over 15 minutes (r = 0.97). They are separate logging runs of the same meters, sampled at different moments. Averaging the two would blur fast events, and stacking them would double-count rows.

## 3. Cleaning pipeline (`src/clean_raw.py`)

| Step | Rule | Effect (Part1 / Part2) |
|---|---|---|
| 1 Parse | `Time` → timestamp. Check against `Unix`. | 0 failures |
| 2 Clock | Wall clock → UTC (Europe/London). In the repeated October hour, rows **before** the backward jump are BST and rows after are GMT. Times that don't exist in March would be dropped. | 0 dropped |
| 3 Sort | Stable sort on UTC | 1 row moved |
| 4a Duplicates | Drop exact duplicate rows | −595,891 / 0 |
| 4b Conflicts | Same timestamp, different values: **keep the last row**. The logger writes on change, so the later row is the newer state. Conflicts differ by ≤ 3 W in 95% of cases, so the choice has almost no effect. | −49,305 / 0 |
| 5 Range | Aggregate > 20 kW or ≤ 0 → NaN. IAM > 4 kW or < 0 → NaN. The 4 kW limit matches the official cleaner. I use **NaN instead of 0**, so a spike is treated as unknown, not as "off". | 2,650 / 985 cells |
| 6 Resample | Mean of readings in each 1-minute bin. An empty bin stays NaN. | 513,322 / 543,720 minutes |
| 7 Merge parts | Use Part1's value for a minute when it has one, otherwise Part2's. Nothing is double-counted. | 920,031 minutes |
| 8 Short gaps | NaN runs **≤ 5 minutes** are forward-filled. The logger writes only on change, so a short silence most likely means "unchanged". | 861 aggregate minutes filled |
| 9 Long gaps | Runs **> 5 minutes** stay NaN and are flagged (`agg_outage`). I don't interpolate: a washing cycle or kettle use inside an outage can't be reconstructed. | 374 outages, 1,904 h |
| 10 Flag | `sub_gt_agg` = sum of IAMs > aggregate + 50 W. The flag is kept and nothing is removed. | 0.59% of minutes |

**Outages kept as NaN (aggregate):**
- 355 last 5 min – 1 h, 6 last 1–6 h, 1 lasts 6–24 h, and 12 last more than a day.
- The longest runs from **24 Jan to 6 Mar 2014 (41.6 days)**. This is the February 2014 outage the readme mentions.
- Other multi-day gaps: 10–19 May 2015, 10–18 Jun 2015 and 3–7 Apr 2015.
- **87.6%** of minutes have a usable aggregate.

**Validation:** I compared my cleaned 1-minute House 1 aggregate with the official cleaned file, after putting both on true UTC. Over 803,619 shared minutes they agree with **r = 0.996**, a mean absolute difference of **2.7 W** and a median difference of **0 W**. The mean power is the same: 477.7 W against 477.9 W. So the simple pipeline reproduces the official result for the aggregate, and it leaves outages as NaN where the official file forward-fills them.

Figures: `outputs/figures/s1_01_full_period_before_after.png` covers the whole period with hourly maxima on a log scale. `outputs/figures/s1_02_zoom_before_after.png` zooms on the day of the largest washing-machine spike (8,192 W, removed) and on the October 2013 clock change. In that second panel, the raw curve sits one hour later than the UTC-corrected curve until 01:00, then they coincide.

## 4. The official cleaned files mix two clocks (undocumented)

The readme says the data were corrected for daylight saving. That holds **only up to the end of each home's raw Part1 file (~1 Oct 2014; 14 Jul 2014 for House 16; 12 Sep 2014 for House 13)**. After that, the cleaned rows come from raw Part2 and are still **UK wall clock**.

**Evidence** (`outputs/tables/clock_check.csv`, `src/clock_check.py`): for every one of the 18 homes, I found the lag that best aligns the cleaned aggregate with the raw aggregate.

| Period | Lag, cleaned file as shipped | Lag after my correction | Meaning |
|---|---|---|---|
| Summer 2014 (raw Part1, BST) | −60 min in **18/18** homes | −60 min in 18/18 | Cleaned file is UTC |
| May 2015 (raw Part2, BST) | **0 min in 18/18** homes | −60 min in 18/18 | Shipped cleaned file is wall clock; corrected file is UTC |

**Fix** (`src/clock.py`): rows up to the Part1 end (converted to UTC) are kept as they are. Later rows are localised as Europe/London and converted to UTC. The repeated hour on 26 Oct 2014 can't be resolved in the cleaned file, so it is dropped: 1 hour per home.

**Why it matters:**
- Without the fix, hour-of-day patterns in summer 2015 are off by an hour.
- Joining the data with weather or tariff data would be misaligned.
- All analysis uses true UTC, and hour-of-day and day-of-week use local UK time, because people run the washing machine by the clock on the wall.

## 5. Cleaned-data quality across the 18 washing-machine homes

Full table: `outputs/tables/s2_availability.csv`. A minute is **valid** (used for EDA cycles and for model training and scoring) when all of these hold:
- the aggregate and washing-machine values are present (no timestamp gap);
- `Issues` = 0;
- the aggregate is **not flat-lined**, meaning it does not hold the exact same value for ≥ 60 minutes. A real home's aggregate always fluctuates, so long identical runs are the official forward-fill hiding an outage. This removes 0.1–11.7% of minutes per home: House 3 has the most, then Houses 10 and 21 at 7.8%.

| | Range across homes |
|---|---|
| Span | 392–648 days |
| Timestamp gaps | 5.8–24.6% of minutes (House 2 is worst) |
| `Issues` flag | 0.8–7.6% |
| Valid minutes | 71.5–89.3% |
| WM drawing exactly 0 W | 53–99% of valid minutes |
| WM standby (0–20 W) | 0.3–40.7%. House 5 idles at ~3 W. |
| WM active (> 20 W) | 1.0–8.6% |

## 6. Remaining concerns

- **Mislabelled or moved IAMs:**
  - House 1's fridge channel shows ~3 kW for several minutes on 6 Mar 2015. That is under the 4 kW limit, so it is kept, but it is not a fridge.
  - Houses 11 and 19 almost never heat water: the median cycle peak is ~370 W against ~2 kW elsewhere. Their cycles look like genuine cold washes (regular 100-min programmes with motor and pump only), not a sensor fault, so I keep them.
  - House 21 washes cold in summer and hot in winter.
- **Part1 zeros** may hide unavailable sensors. Part1 has no NaN, so an IAM reading 0 W might be off or offline.
- **Sensor asynchrony.** The IAMs can lead or lag the aggregate by up to ~8 s, which is why the sum of IAMs sometimes exceeds the aggregate.
- **Unmetered loads.** Only 9 appliances per home are metered, so most of the aggregate is unexplained. That doesn't affect the target, but it limits what the failure analysis can attribute.
- **The cleaned files' spike removal sets values to 0**, which looks the same as "off". A spike during a wash would make the cycle look shorter. This is rare: < 0.01% of minutes.
