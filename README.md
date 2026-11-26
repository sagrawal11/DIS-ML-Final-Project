# Predicting F1 lap times across the 2026 regulation change

2026 is the biggest rules reset Formula 1 has seen in years, and it gives us a really clean natural experiment. The cars switch to brand-new power units with a roughly 50/50 split between the combustion engine and electric deployment, they gain active aerodynamics that trim drag down the straights and add downforce in the corners, and DRS goes away entirely. All of that changes how a lap is actually driven, so if you train a lap-time model on the 2022-2025 cars and point it at 2026, you'd expect it to break, and the interesting question is by how much and for how long. This project is my attempt to measure exactly that: treat the regulation change as a distribution shift, quantify how much a model's error jumps when it crosses into 2026, and see how many 2026 races it takes before the model recovers once it starts seeing the new cars.

It's a supervised regression project built on real race data, and the whole pipeline (download, cleaning, feature engineering, and a first modelling iteration) is reproducible from a fresh clone.

## Research question

How much does lap-time prediction error increase when a model trained on 2022-2025 races is applied to 2026 (new power units, active aero, no DRS), and how many 2026 races does it need to recover?

- **Task:** supervised regression.
- **Target:** `lap_delta_s`, the lap time minus that race's median clean lap time, in seconds. Subtracting the race median takes out the circuit (a 70 s Red Bull Ring lap vs a 105 s Silverstone lap) so the model learns pace *within* a race rather than memorising track length.
- **Evaluation:** MAE and RMSE in seconds, always split by race or by season (never random lap-level splits), reported overall and broken down by compound, team lineage, and circuit.

## Data

The data comes from the [TracingInsights](https://github.com/TracingInsights) season repositories (one repo per year, `TracingInsights/2022` through `TracingInsights/2026`), which publish [FastF1](https://github.com/theOehrly/Fast-F1)-derived lap tables: lap time, tyre compound, stint, tyre age, track status, pit flags, sector times, speed traps, and per-lap weather for every session. Notebook 01 sparse-checks out only the files we use (race and qualifying lap tables, plus each race's session info and corner map, ~20 MB per season) rather than the full repos, which also hold GBs of telemetry and radio. Each season is **pinned to an exact TracingInsights commit** (frozen on 2026-11-26, see `src/f1lap/download.py`), so a fresh clone reproduces our numbers even as new races get added upstream.

The raw data funnels from **125,909 laps down to 90,488 clean, representative green-flag laps across 113 races** (22 + 22 + 24 + 24 + 21), of which 21 are held in 2026 (up to and including Las Vegas) and 19 are dry. Full credit for the underlying data goes to TracingInsights and the FastF1 project. **No data is committed to this repo** (the `data/` and `ti20*/` folders are gitignored); everything is downloaded at runtime by notebook 01.

A few data quirks the pipeline handles explicitly, because event names are not circuits:

- In 2026 the "Spanish GP" moved to a brand-new Madrid circuit, Barcelona hosts the "Barcelona GP", and the "Bahrain GP" was actually held at Sepang in Malaysia. Races are matched by circuit, not event name.
- Team lineages are unified across rebrands: AlphaTauri/RB become Racing Bulls, and Alfa Romeo/Kick Sauber/Audi become Sauber/Audi. Cadillac is genuinely new in 2026 and is flagged as an unseen team, just like the Madrid circuit is flagged as an unseen track.
- Wet races (more than 5% of laps on intermediate or wet tyres) are dropped, since rain pace is a different problem.

## Results at a glance

- **Measuring pace beats naming teams.** Going from 6 numeric features plus team one-hots to 25 features built from qualifying, in-race context and circuit character cuts in-era error by ~20% (2025 test MAE 0.59 -> 0.47 s). A model with *no team names at all* does just as well, because qualifying pace and recent lap times already tell it how fast each car is.
- **Saturday predicts Sunday, and far more so in 2026.** A team's qualifying gap to pole correlates with its race pace at r = 0.62 in 2025 and r = 0.91 in 2026, as the field spread from ~3% to ~6% off pole.
- **The 2026 regime shift adds ~30%**: XGBoost MAE is ~0.63 s on 2026 vs ~0.48 s on average for the two normal years (2024, 2025). The richer features lower error in every era by about the same proportion, so the *relative* jump barely moves from the first iteration (+31%). The unseen **Cadillac** team (1.01 s) and **Madrid** circuit (1.02 s) are still the worst-hit, though much improved (from 1.54 and 1.29 s).
- **Measuring pace helps most exactly where the rules changed.** Qualifying and driver features cut 2026 error ~4x more than in a normal year (-5% vs -1%), but no feature set closes the gap: the relative jump only falls from +36% to ~+30%.
- **What's left is how the race is driven.** We split error into *pace-level* error (wrong overall pace for a driver in a race) and *within-race* error. Pace-level error is 49% of 2026 error for the team-names model vs 21% in a normal year; measurement features cut it to 32%. But even with it removed entirely, 2026 laps stay ~20% harder to predict, so the new rules changed lap-to-lap dynamics, not just the pecking order.
- **Wrong and overconfident.** Split-conformal 90% intervals cover 90.5% of laps in a normal year but only 81.7% in 2026 (55% at Monaco and the new Madrid circuit). Recalibrating the interval width on the 2026 races already run restores ~90% average coverage, at the cost of a wider band (+/-1.0 -> +/-1.4 s). A shift classifier only reaches AUC 0.59: no single 2026 lap looks obviously different, which is what makes the shift easy to miss.
- **Reading the race beats retraining.** On a fixed test set of the last 8 races of 2026, a model with in-race features (recent pace, gaps) that is *never retrained* (0.59 s) beats a model without them even after retraining on 11 races of 2026 (0.67 s). With measurement features, adding 2026 races to training buys only ~2%, and the cleverer strategies (upweighting 2026, warm-start fine-tuning, training on 2026 alone) do no better or worse than simply appending the new races.
- **Champions don't ease off.** Races run after both titles were decided show no slowdown from the champion team, and the regime-shift result holds with or without them.
- **What physically changed** (notebook 07): soft tyres degrade about half as fast in 2026 and the pace gap between compounds vanishes (medium/hard were ~0.3-0.4 s slower than soft; now ~0); removing DRS more than halved the straight-line boost of running within 1 s of another car (~7.7 -> ~3.3 kph); the cars lost time mainly in corners rather than on straights; and race pace sits closer to qualifying pace (6.0% vs 7.1-7.6% before). SHAP shows the 2026 model leaning on measured pace (`recent_pace`, `team_quali_gap_pct`).

## Setup

You'll need Python 3.12 (that's what this was developed and verified against). From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then run the notebooks **in order**, since each one reads what the previous ones wrote (01 writes the clean table, 03 the model table, 04 the tuned parameters):

```bash
jupyter lab      # or: jupyter notebook, then run 01 -> 07
```

Notebook 01 downloads the data on its first run (a few minutes, network required) and is a no-op on reruns once the data is present. To run everything headless instead (about 2 minutes after the download):

```bash
for nb in notebooks/0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```

## Repo structure

```
final_project/
├── notebooks/
│   ├── 01_data_acquisition_validation.ipynb   # download, validate, cleaning funnel, target
│   ├── 02_eda.ipynb                           # target by season, stint pace, pace change by circuit
│   ├── 03_feature_engineering.ipynb           # qualifying, driver, in-race, circuit, weather features + leakage tests
│   ├── 04_baselines_tuning.ipynb              # three models, per-set tuning, feature ablation, in-era check
│   ├── 05_regime_shift.ipynb                  # drift vs 2026, names vs measurements, error decomposition, shift detection, conformal intervals
│   ├── 06_adaptation.ipynb                    # adaptation strategies, reading the race vs retraining
│   └── 07_diagnosis.ipynb                     # SHAP, sectors/speed traps, DRS removal, quali vs race, fuel vs tyre
├── src/f1lap/                 # the pipeline + models, imported by the notebooks
│   ├── download.py            # TracingInsights sparse checkout
│   ├── clean.py               # loaders (race, qualifying, session info), cleaning funnel, target
│   ├── features.py            # all feature groups, leakage self-check, encoding
│   └── models.py              # physics baseline, XGBoost, residual, metrics, experiment helpers
├── figures/                   # committed plots, written by the notebooks
├── results/                   # model_config.json: chosen feature set + tuned params
├── docs/process_log.md        # how the project evolved, and why
├── data/                      # gitignored; regenerated by notebook 01
├── requirements.txt           # pinned versions
├── LICENSE                    # MIT
└── README.md
```

The heavy data and modelling logic lives in `src/f1lap/` so the notebooks stay readable and the rules have a single source of truth. Each module has a small synthetic self-check you can run directly, e.g. `python src/f1lap/clean.py` or `python src/f1lap/models.py`.

## References and AI use

- **Data:** [TracingInsights](https://github.com/TracingInsights) season repositories, derived from [FastF1](https://github.com/theOehrly/Fast-F1).
- **AI use:** I used Claude (Anthropic) as a collaborator throughout this project. It helped with finding and vetting the data source, drafting the project proposal, building the data validation and preprocessing notebook, and setting up this repository (extracting the `f1lap` package, writing the README, and building the baselines and regime-shift modelling notebook). All modelling decisions, the research question, and the final analysis are our own, and every result in the notebooks is reproduced from the committed code.

## License

MIT, see [LICENSE](LICENSE).
