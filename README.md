# Predicting F1 lap times across the 2026 regulation change

2026 is the biggest rules reset Formula 1 has seen in years, and it gives us a really clean natural experiment. The cars switch to brand-new power units with a roughly 50/50 split between the combustion engine and electric deployment, they gain active aerodynamics that trim drag down the straights and add downforce in the corners, and DRS goes away entirely. All of that changes how a lap is actually driven, so if you train a lap-time model on the 2022-2025 cars and point it at 2026, you'd expect it to break, and the interesting question is by how much, why, and how you recover. This project treats the regulation change as a real-world distribution shift, the same problem a credit model faces after a policy change or a demand model after a pandemic, but with incredibly well-instrumented data.

It's a supervised regression project built on real race data, and the whole pipeline (download, cleaning, features, tuning, every experiment and figure) is reproducible from a fresh clone.

## The short answer

- **How much does error rise?** About **+30%**: a model trained on 2022-2025 predicts 2026 laps with an MAE of ~0.63 s, against ~0.48 s for the two normal year-to-year transitions (2024, 2025). That's with our best features; a model that only knows team names jumps +36%.
- **How many 2026 races to recover?** With the right features, retraining barely matters. A model that *measures* pace every weekend (qualifying gaps, recent lap times) recovers most of what's recoverable before a single 2026 race is added to training; eleven races of retraining buy only ~2% more. The rest of the gap never closes, because the new rules changed **how a race unfolds** (soft tyres wear half as fast, the pace differences between compounds vanished, and the straight-line boost from following another car more than halved), not just which car is fast.

## Research question

How much does lap-time prediction error increase when a model trained on 2022-2025 races is applied to 2026 (new power units, active aero, no DRS), and how many 2026 races does it need to recover?

- **Task:** supervised regression, one row per clean racing lap.
- **Target:** `lap_delta_s`, the lap time minus that race's median clean lap time, in seconds. Subtracting the race median takes out the circuit (a 70 s Red Bull Ring lap vs a 105 s Silverstone lap) so the model learns pace *within* a race rather than memorising track length.
- **Evaluation:** MAE (main metric, in readable seconds) and RMSE (penalises big misses), always split by race or by season, never by random laps, and broken down by compound, team and circuit.

## Data

The data comes from the [TracingInsights](https://github.com/TracingInsights) season repositories (one repo per year, `TracingInsights/2022` through `TracingInsights/2026`), which publish [FastF1](https://github.com/theOehrly/Fast-F1)-derived lap tables: lap time, tyre compound, stint, tyre age, track status, pit flags, sector times, speed traps, and per-lap weather for every session. Notebook 01 sparse-checks out only the files we use (race and qualifying lap tables, plus each race's session info and corner map, ~20 MB per season) rather than the full repos, which also hold GBs of telemetry and radio. Each season is **pinned to an exact TracingInsights commit** (frozen on 2026-11-26, see `src/f1lap/download.py`), so a fresh clone reproduces our numbers even as new races get added upstream.

The raw data funnels from **125,909 laps down to 90,488 clean, representative green-flag laps across 113 races** (22 + 22 + 24 + 24 + 21), of which 21 are held in 2026 (up to and including Las Vegas) and 19 are dry. Full credit for the underlying data goes to TracingInsights and the FastF1 project. **No data is committed to this repo** (the `data/` and `ti20*/` folders are gitignored); everything is downloaded at runtime by notebook 01.

A few data quirks the pipeline handles explicitly:

- **Event names are not circuits.** In 2026 the "Spanish GP" moved to a brand-new Madrid circuit, Barcelona hosts the "Barcelona GP", and the "Bahrain GP" was actually held at Sepang. Races are matched by circuit, and we check that against the official circuit keys.
- **Team lineages are unified across rebrands** (AlphaTauri/RB -> Racing Bulls, Alfa Romeo/Kick Sauber/Audi -> Sauber/Audi). Cadillac is genuinely new in 2026 and flagged as unseen, like the Madrid circuit.
- **Wet races are dropped** (more than 5% of laps on intermediate or wet tyres), since rain pace is a different problem; wet *qualifying* sessions are flagged the same way.
- **Sensor glitches are repaired**, e.g. a 0.0 °C track temperature on one lap of the 2026 Las Vegas GP.

## Methods

**Models**, in increasing complexity: a **physics baseline** (Heilmeier-style lap time = compound base + tyre wear + fuel mass, as a linear regression), **XGBoost**, and a **physics-informed residual model** (XGBoost on the physics baseline's errors).

**Features** (notebook 03): we went from 6 numeric features plus one-hot team identity to 25 features in six groups, built on the idea of *measuring* a car's pace every weekend instead of *naming* it. The biggest new source is qualifying (gap to pole, team best, teammate gap, knockout-aware rank, speed trap), plus in-race context (gap to the car ahead and behind, recent pace, positions gained vs qualifying), driver racecraft from earlier races, and circuit character from the corner maps. **Every feature only uses information available when the lap starts**, and that's tested, not just promised: scrambling every outcome from lap L onwards must leave lap L's features bit-for-bit unchanged.

**Tuning and selection** (notebook 04): every feature set gets its own grouped-CV grid search on 2022-2024 only, and the final feature set is chosen by cross-validation, so 2025 (in-era test) and 2026 (the shift) never influence a choice.

**Evaluation design**, which is where we think the project is most novel:
1. **A normal-drift control**: every season transition is measured the same way, so only the error *beyond* ordinary year-to-year change is attributed to the new rules.
2. **Names vs measurements**: every feature set evaluated across the rule change, to see which kinds of information survive it.
3. **Error decomposition** into getting a driver's overall pace wrong vs getting the lap-to-lap evolution wrong.
4. **Adversarial validation and conformal prediction intervals** to measure how visible the shift is and whether the model stays honest about its uncertainty.
5. **Adaptation strategies on a fixed test set** (naive append, upweighting, warm-start fine-tuning, 2026-only), plus reading the race vs retraining.
6. **Physical diagnosis**: sectors and speed traps, the DRS-removal effect, the qualifying-to-race gap, and a fixed-effects regression that separates fuel burn from tyre wear.

## Results at a glance

- **Measuring pace beats naming teams.** The richer features cut in-era error by ~20% (2025 test MAE 0.59 -> 0.47 s). A model with *no team names at all* does just as well, because qualifying pace and recent lap times already tell it how fast each car is.
- **Saturday predicts Sunday, and far more so in 2026.** A team's qualifying gap to pole correlates with its race pace at r = 0.62 in 2025 and r = 0.91 in 2026, as the field spread from ~3% to ~6% off pole.
- **The 2026 regime shift adds ~30%** (0.63 s vs ~0.48 s for normal years). The unseen **Cadillac** team (1.01 s) and **Madrid** circuit (1.02 s) are still the worst-hit, though much improved from the first iteration (1.54 and 1.29 s).
- **Measuring pace helps most exactly where the rules changed.** Qualifying and driver features cut 2026 error ~4x more than in a normal year (-5% vs -1%), but no feature set closes the gap: the relative jump only falls from +36% to ~+30%.
- **What's left is how the race is driven.** *Pace-level* error (wrong overall pace for a driver in a race) is 49% of 2026 error for the team-names model vs 21% in a normal year; measurement features cut it to 32%. But even with it removed entirely, 2026 laps stay ~20% harder to predict.
- **Wrong and overconfident.** 90% prediction intervals cover 90.5% of laps in a normal year but only 81.7% in 2026 (55% at Monaco and the new Madrid circuit). Recalibrating the width on the 2026 races already run restores ~90% average coverage (+/-1.0 -> +/-1.4 s). A shift classifier only reaches AUC 0.59: no single 2026 lap looks obviously different, which is what makes the shift easy to miss.
- **Reading the race beats retraining.** On the last 8 races of 2026, a model with in-race features that is *never retrained* (0.59 s) beats a model without them even after retraining on 11 races of 2026 (0.67 s). Upweighting 2026, warm-start fine-tuning and training on 2026 alone do no better, or worse, than simply appending the new races.
- **What physically changed:** soft tyres degrade about half as fast and the pace gap between compounds vanishes (medium/hard were ~0.3-0.4 s slower than soft; now ~0); removing DRS more than halved the straight-line boost of running within 1 s of another car (~7.7 -> ~3.3 kph); the cars lost time mainly in corners rather than on straights; and race pace sits closer to qualifying pace (6.0% vs 7.1-7.6% before).
- **Champions don't ease off.** Races run after both titles were decided show no slowdown from the champion team, and the regime-shift result holds with or without them.
- **Random splits flatter models.** The same model scores ~18% better under a random lap-level split than under our race-grouped split, which matters when comparing against published numbers.

The full story of how we got here, including the mistakes we caught along the way, is in [docs/process_log.md](docs/process_log.md).

## How this compares to existing work

Peer-reviewed ML work on F1 is thin, so we reviewed papers, theses and open-source projects and read their numbers critically.

| Work | What it does | How our results relate |
|---|---|---|
| Heilmeier, Graf & Lienkamp (2018) | Race simulation built on a lap-time decomposition: base time + tyre degradation + fuel mass | Our physics baseline is this decomposition. It's a clear floor (0.76 s in-era), and the physics-informed residual model gained nothing over plain XGBoost across the rule change. More importantly, our fixed-effects estimates show the decomposition's *parameters* changed in 2026 (soft degradation halved, compound offsets vanished), so physical models need re-fitting after a rule change too. |
| Cappello & Hoegh (2025) | Bayesian state-space model of tyre degradation, fit per race | Closest to our target. Fitting per race is a strength under a regime shift: our results show that tyre behaviour learned from earlier seasons stopped applying in 2026, which a per-race model re-learns automatically. |
| F1 StratLab, Vega Sobral (2026) | XGBoost lap-time model trained on 2023-2025, MAE 0.392 s | Our main benchmark. Our in-era 0.47 s is on a harder, race-normalised target with a whole season held out, and we measured that a random lap-level split makes our own model look ~18% better (0.41 s), about the size of the gap. Since it's trained entirely pre-2026, our results suggest its error would rise ~30% on 2026 cars unless it's given measurement features. |
| Mistry & Smith (2025), *AutoF1* | RNNs on FastF1 data to simulate strategy decisions, with gap-to-car-behind features | Supports our gap features. Our DRS analysis shows following-car dynamics changed in 2026 (the boost from running within 1 s more than halved), so a strategy simulator trained on 2022-2025 overtaking behaviour would need re-learning. |
| Sasikumar et al. (2025) | Bi-LSTM pit-stop prediction on FastF1, 2020-2024 | The strongest published FastF1 work. It spans the 2022 regulation change without testing for era shift; a ~30% error jump at the 2026 boundary is a concrete reason to evaluate F1 models by era. |
| Garcia Tejada (Aalto, MSc thesis) | SVM vs random forest vs neural network for pit-stop timing | A useful model-comparison template, and its conclusion that pit-stop prediction is hard is why we restricted ourselves to clean green-flag laps. |
| schilamkur, *Predict-TireDeg* | XGBoost predicting the lap a tyre "falls off" | Its label comes from a rolling threshold, a design choice that shapes the result; our 2026 finding that soft tyres degrade half as fast means any fixed fall-off threshold learned pre-2026 would be miscalibrated. |

The general distribution-shift literature shaped the evaluation: the covariate-vs-concept shift framing (Quiñonero-Candela et al., 2009) behind our adversarial validation, split-conformal prediction (Angelopoulos & Bates, 2021) for the intervals, and the caveat that conformal guarantees assume exchangeable data (Barber et al., 2023), which laps clustered by race violate, hence our normal-year control. SHAP values come from TreeSHAP (Lundberg et al., 2020) via XGBoost (Chen & Guestrin, 2016).

## Limitations

- **Clean, dry, green-flag laps only.** Results apply to normal racing laps, not to full-race prediction with safety cars, pit stops or rain.
- **The target hides the absolute slowdown.** Subtracting the race median removes the ~2% overall 2026 slowdown, so the model sees relative pace only; we report the absolute change separately (notebook 02).
- **2026 is a small sample.** 19 dry races, so the adaptation experiments use 11 races to adapt and 8 to test, and some per-circuit comparisons rest on 14 circuits.
- **Some effects are associations, not causal.** Tyre age isn't random (teams choose stint lengths), being within 1 s of a car mixes DRS with traffic, and speed-trap locations can move between seasons.
- **Missing pieces of data.** Two 2026 circuits (Madrid, Sepang) have no corner map, eight late-2024 qualifying sessions have no Q1/Q2/Q3 labels, and wet qualifying sessions make qualifying pace a wet-weather measurement.
- **One model family.** We compared feature sets and adaptation strategies rather than architectures; a sequence model (e.g. an LSTM over stints) is a natural next step.

## Setup

You'll need Python 3.12 (that's what this was developed and verified against). From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then run the notebooks **in order**, since each one reads what the previous ones wrote (01 writes the clean table, 03 the model table, 04 the chosen features and tuned parameters):

```bash
jupyter lab      # or: jupyter notebook, then run 01 -> 07 with the .venv kernel
```

Notebook 01 downloads the data on its first run (a few minutes, network required) and is a no-op on reruns once the data is present. To run everything headless instead:

```bash
for nb in notebooks/0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```

The full headless run took ~37 minutes on a laptop in our from-scratch check (fresh clone, new virtual environment, fresh download), most of it the per-feature-set hyperparameter search in notebook 04 and the adaptation experiments in notebook 06. That check reproduced every number in the notebooks exactly. If your Jupyter has a custom `python3` kernel pointing at a different Python, select the virtual environment's kernel instead.

## Repo structure

```
final_project/
├── notebooks/
│   ├── 01_data_acquisition_validation.ipynb   # download, validate, cleaning funnel, target
│   ├── 02_eda.ipynb                           # target by season, stint pace, pace change by circuit, do champions ease off?
│   ├── 03_feature_engineering.ipynb           # qualifying, driver, in-race, circuit, weather features + leakage tests
│   ├── 04_baselines_tuning.ipynb              # three models, per-set tuning, feature ablation, in-era check, random-split leak
│   ├── 05_regime_shift.ipynb                  # drift vs 2026, names vs measurements, error decomposition, shift detection, conformal intervals
│   ├── 06_adaptation.ipynb                    # adaptation strategies, reading the race vs retraining
│   └── 07_diagnosis.ipynb                     # SHAP, sectors/speed traps, DRS removal, quali vs race, fuel vs tyre
├── src/f1lap/                 # the pipeline + models, imported by the notebooks
│   ├── download.py            # pinned TracingInsights sparse checkout
│   ├── clean.py               # loaders (race, qualifying, session info), cleaning funnel, target
│   ├── features.py            # all feature groups, leakage self-check, encoding
│   └── models.py              # physics baseline, XGBoost, residual, metrics, experiment helpers
├── figures/                   # committed plots, written by the notebooks
├── results/                   # model_config.json (final features + params), ablation.json (every set's params)
├── docs/process_log.md        # how the project evolved, and why
├── data/                      # gitignored; regenerated by notebook 01
├── requirements.txt           # pinned versions
├── LICENSE                    # MIT
└── README.md
```

The heavy data and modelling logic lives in `src/f1lap/` so the notebooks stay readable and the rules have a single source of truth. Each module has a small synthetic self-check you can run directly, e.g. `python src/f1lap/features.py` (which includes the leakage tests).

## References

**Data**
- TracingInsights. F1 season data repositories, 2022-2026. https://github.com/TracingInsights
- Schäfer, P. FastF1 Python library. https://github.com/theOehrly/Fast-F1

**Formula 1 modelling**
- Heilmeier, A., Graf, M., & Lienkamp, M. (2018). A Race Simulation for Strategy Decisions in Circuit Motorsports. *21st IEEE International Conference on Intelligent Transportation Systems (ITSC)*. doi:10.1109/ITSC.2018.8570012
- Cappello, C., & Hoegh, A. (2025). A State-Space Approach to Modeling Tire Degradation in Formula 1 Racing. arXiv:2512.00640.
- Vega Sobral, V. (2026). *F1 StratLab: AI Models for Strategy Recommendations in Formula 1 Races*. Bachelor's thesis, UIE Campus Coruña.
- Mistry, V., & Smith, W. A. P. (2025). AutoF1: An RNN-Based Approach to Simulating Strategic Decision-Making in Formula 1. *IEEE ITSC 2025* (in press). https://eprints.whiterose.ac.uk/id/eprint/229478
- Sasikumar, A., Leema, A. A., & Balakrishnan, P. (2025). Data-driven pit stop decision support for Formula 1 using deep learning models. *Frontiers in Artificial Intelligence*, 8. doi:10.3389/frai.2025.1673148
- Garcia Tejada, L. *Applying Machine Learning to Forecast Formula 1 Race Outcomes*. Master's thesis, Aalto University.
- schilamkur. Predicting Formula 1 Tire Degradation. Open-source project.

**Methods**
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56-67.
- Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (Eds.) (2009). *Dataset Shift in Machine Learning*. MIT Press.
- Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.
- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. *The Annals of Statistics*, 51(2).

**Race facts used** (championship clinch dates for the post-clinch robustness check): Formula1.com and ESPN race reports for the 2026 São Paulo GP, and Sky Sports for the 2026 Mexico City GP.

## License

MIT, see [LICENSE](LICENSE).
