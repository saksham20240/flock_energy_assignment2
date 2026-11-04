# Recommendation: washing-machine disaggregation for a utility

*One-page summary for a distribution utility. Evidence and numbers: [RESULTS.md](RESULTS.md) and [DATA.md](DATA.md).*

## What the model can and cannot do

The model was tested on 4 UK homes it never saw in training. From **1-minute whole-home data alone**, it:

| Can do | Evidence |
|---|---|
| Estimate a home's **total washing-machine energy** over months | Within **+0.3%** pooled. Per home: −30% to +45% |
| Tell a home **roughly how much it spent on washing each day** | Daily error is **35% lower than assuming zero**. Daily correlation is 0.79–0.85 in 3 of 4 homes. |
| **Detect heated washes**, and so count washes and see when they happen | Finds **89% of heated washes**; 83% of all washes |
| Rank homes by **how often they wash in the evening peak** | Gets the order of the two heaviest peak-time washers right |

| Cannot do (yet) | Why |
|---|---|
| Minute-exact timing | Minute-level error is no better than "always off". It is fine per hour or day. |
| Bill-grade or per-home accuracy guarantees | About half of predicted washes are false. 45% of those coincide with a kettle or dishwasher, and most of the rest with unmetered cooking or water heating. |
| See **cold or eco washes** | Only **49%** of washes with no heater are found |
| Work on 15- or 30-minute smart-meter data | In a simple-model test, interval detection drops by ~40–50% and daily tracking by a third |

**Suitable uses:** portfolio-level insight, segmenting customers into targeting groups, and app messages framed as estimates ("you probably washed 4 times this week, mostly 17:00–19:00"). Results should be shown as ranges.

**Not suitable for:** billing, penalties, single-home guarantees, or as the *only* measure of a behaviour-change programme.

## Section 4: Practical opportunity

**Shift heated washes out of the 16:00–20:00 network peak, targeting the homes that wash then.**

- Across 18 homes, **20% of washing-machine energy is used 16:00–20:00**.
- Homes differ enormously:
  - House 10 starts 43% of its washes in the peak, about **3.7 peak washes a week**.
  - Houses 16, 9, 5 and 7 start 23–39% of their washes in the peak.
  - Houses 1, 8 and 18 start 3% or fewer there. House 8 already runs its washes on a night timer.
- **Heating is 76% of a wash's energy**, about 2 kW for 15–20 minutes. So each shifted wash removes a clear block of peak demand.
- The programme:
  1. Use the model to find the households with frequent evening washes.
  2. Offer them a delay-start nudge or a time-of-use tariff message.
  3. Add a second message about 30°C washing. Heating energy scales with the temperature rise, so 40°C → 30°C cuts heater energy by roughly 40%, and some REFIT homes already wash cold.

**How to measure whether it worked:**
- Run a **randomised trial**: targeted homes split into treatment and control, with 3 months of baseline and 3 months of intervention.
- **Primary outcome: metered aggregate demand 16:00–20:00**, as a difference-in-differences between the arms. It needs no model, so model errors cannot bias it.
- The expected effect is small. One shifted wash a week is about 0.5 kWh, roughly −20 W on a home's average peak-window demand. Detecting it at the household level needs a trial of **a few thousand homes**.
- **Secondary outcome: the model-estimated share of heated washes in the window.** Model errors affect both arms equally, so the difference is fair. It is more sensitive, but it cannot see cold washes. If customers switch to cold washes, this measure would over-credit the programme for "moved" washes.
- Fewer than 100 homes can only show the direction of the effect, not its size.
- **Caution:** the model's false alarms cluster at dinner time. It overstates every test home's peak-wash share: for example 32% predicted against 20% actual for House 2. Targeting should use a stricter threshold, or be confirmed with a short customer survey.

## Section 4: Limitations

- **Overlapping appliance signatures.** Dishwashers, kettles, ovens and electric showers use the same kind of resistive heating. About **half of predicted washes are false**: 25% coincide with a kettle, 20% with a dishwasher, and 48% with no metered load at all, which points to cooking or water heating. Cold washes, at 50–300 W, hide under fridges and TVs.
- **Missing data.**
  - Homes lose 6–25% of minutes to gaps. February 2014 and spring 2015 are the worst periods.
  - Up to 12% of minutes are flat-lined forward-fills, which look like real load.
  - The official cleaned files also mix UTC and wall-clock time after Oct 2014. Without the fix, hour-of-day results are shifted by an hour.
- **Household differences.**
  - Energy per wash varies **8×** between homes (89–693 Wh).
  - Some homes always wash cold. Others use night timers.
  - Recall ranges from 70% to 99% across the 4 test homes. A model trained on 12 homes is a starting point, not a calibrated product.
- **UK homes → India.** The model should not be transferred without retraining on local data.
  - **Machines:** semi-automatic and top-loading machines are common, and many have no heater. That is the cold-wash case this model misses half the time. Hand washing is also widespread.
  - **Confusers:** an **electric geyser** (water heater) has a 2 kW block signature just like a wash heater. Air conditioners, water pumps, inverter/UPS charging, rooftop solar and voltage fluctuation all distort the aggregate.
  - **Grid and data:** outages and load-shedding create gaps, and peak hours and tariffs differ.
  - **Meters:** India's smart-meter rollout mostly records at **15–30 minutes**, not 1 minute.

## Section 4: What changes with 15- or 30-minute data

A washing machine is visible in the aggregate mainly through its heater, a rectangular ~2 kW block lasting 15–20 minutes with sharp on and off edges. At 15-minute resolution that block becomes one or two averages of a few hundred watts. At 30 minutes it is indistinguishable from a kettle boiled twice or an oven on low. Event-level questions get much harder: *did a wash happen, when did it start, how many cycles were there, and was it hot or cold?*

In my experiment, the same simple model at 15 and 30 minutes, compared with 1 minute:
- detecting whether the machine ran in an interval fell from **F1 0.30 to 0.19 and 0.16**;
- the correlation of daily washing energy fell from **0.53 to 0.35**;
- total energy went from a −22% bias to an **overestimate of +44% and +59%**.

Coarse data still supports **monthly or seasonal totals across many homes** and statistical segmentation, such as "homes with frequent evening heating blocks". But peak-time nudges, wash counting and cold-wash detection would need either 1-minute or 10-second data from the meter's local consumer port (the HAN, home area network), or a model trained on labelled data at that coarse resolution with much more modest goals.
