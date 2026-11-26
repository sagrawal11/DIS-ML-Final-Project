# Process log

The course grades the process, not just the final numbers, so this is the honest record of how the project evolved, what we found at each stage, and why we changed direction when we did. Newest entries are at the bottom.

## Nov 9 - proposal

We picked lap-time prediction across the 2026 regulation change because it's a really clean natural experiment: brand-new power units with a roughly 50/50 combustion/electric split, active aero, and no DRS, all landing on one dated weekend. That makes it a real-world distribution shift, the same problem a credit model faces after a policy change, but with incredibly well-instrumented data. The proposal (`F1_ML_Project_Proposal.pdf`) set the research question, the race-median target, the three models (physics baseline, XGBoost, physics-informed residual) and four experiments (in-era check, regime shift, few-shot recovery, SHAP diagnosis).

## Nov 19 - first end-to-end iteration (status update)

**What was done.** The full first modelling iteration: the cleaning funnel (124,920 raw laps down to 89,737 clean laps from 112 races), all three models wired through the `f1lap` package, race-grouped GroupKFold tuning, the in-era check (~0.58 s MAE), the normal-drift control vs the 2026 regime shift (a ~+25% jump in error) with per-compound, per-team and per-circuit breakdowns, the few-shot recovery curve over 5 race orderings, and a TreeSHAP before/after diagnosis.

**What was hardest.** Keeping the splits honest. Everything is split by race or by season, the feature matrix is built once so columns line up across eras, and tuning is grouped by race so nothing leaks. Getting SHAP working without the `shap` library (which broke the numpy stack) by using XGBoost's native `pred_contribs` was the other fiddly bit.

**What surprised us.**
1. The physics-informed residual model gave no edge over plain XGBoost. They tie everywhere, so physical priors didn't transfer across the regime any better than learned patterns.
2. Few-shot recovery plateaus well above the in-era bar: the first 2026 race helps a lot, but eight races still leave ~0.75 s vs ~0.59 s in-era.
3. SHAP said the relationship that changed most was `position` and the new team, not tyre degradation, even though the tyre curves in the EDA looked very different.

## Nov 26 - self-review and restructure

Before polishing for the final presentation we went back through the work critically, the way a tough reviewer would, and found real holes:

- **The model can't tell what year it is.** There's no season information, so `team_Aston Martin` in 2026 is averaged with four seasons of Aston Martin, and one 2026 race (~900 laps) is drowned out by ~73,000 older laps. That's a likely explanation for the few-shot plateau, and it's testable.
- **The few-shot design was muddled.** Random race orderings instead of the chronological "first k races" the proposal described, and a test set that shrinks as k grows, so the points on the curve aren't directly comparable.
- **We over-read `position` in SHAP.** Running position is mostly a stand-in for car pace, so its rise in 2026 more likely means the pecking order reshuffled than that "where you run" matters more.
- **We promised features we never built** (gap to the car ahead, driver), and we were using 18 of the ~42 columns in the raw data, leaving sector times, four speed traps and session timestamps on the table.
- **Our features named teams instead of measuring them.** One-hot team identity carries no information about the current season, which is exactly what breaks under a regulation change. Qualifying pace, on the other hand, is re-measured every weekend.

What we changed as a result:

1. **Split the two notebooks into seven**, one job each (data, EDA, features, baselines and tuning, regime shift, adaptation, diagnosis), with the shared experiment helpers moved into `f1lap.models`. We verified the split changed no results: all 83 numeric output lines from the old notebooks reappear verbatim in the new ones.
2. **Added "what you see / what it means" notes** after every plot and table, and corrected the `position` interpretation.
3. **Pinned the data and added Las Vegas.** The downloader used to clone whatever was newest upstream, so a fresh clone would have silently produced different numbers from ours (TracingInsights added the Las Vegas GP on Nov 20-22, after our Nov 13 snapshot). Every season is now pinned to an exact commit, and we took the newer 2026 snapshot because 2026 races are the scarcest data we have. We checked that the only change was the new race (no earlier 2026 race was revised), and the headline results barely moved (2026 MAE 0.786 -> 0.787 s), so every finding from Nov 19 holds. The downloader now also fetches qualifying lap tables, each race's session info and the corner map, which the new features need.
4. **Fixed a weather glitch** the new race exposed: on lap 5 of the 2026 Las Vegas GP every car's track temperature reads 0.0 °C (the laps either side read ~14 °C). Non-positive temperatures are now treated as missing and filled with the race median.
5. **Tested whether champions ease off once both titles are decided.** The idea was to drop races run after the drivers' and constructors' championships were both mathematically settled, on the logic that the champion team has nothing left to race for. That covers 10 races (2022: Mexico City, São Paulo, Abu Dhabi; 2023: Qatar through Abu Dhabi; 2026: Las Vegas). We first reconstructed the championship standings from the race and sprint lap data to find the clinch points, which matched the real history for 2022-2025 (lap data can't see post-race penalties, which can shift a clinch by one race but never changed *which* races were run after both titles were settled), and confirmed 2026 against race reports. Then, instead of assuming, we tested it: the champion team's pace gap to its best rival didn't get worse after the clinch (2022 Red Bull actually got faster, 2023 was unchanged, 2026 is a single race inside normal noise). So we kept the races, saved ourselves from throwing away 30% of 2023 and one of our scarce 2026 races, and added a robustness check showing the regime-shift jump holds either way (+29% with them, +32% without). Writing that check also caught an imprecision we'd been repeating since Nov 19: the jump is +29% against the average of the two normal years, not +25%.

## Nov 26 (cont.) - feature engineering

We went from 6 numeric features to 25, organised into groups (base, qualifying, driver, in-race, circuit, weather) so the next step can ablate them. The big idea is to replace features that *name* a car (one-hot team) with features that *measure* it every weekend, mostly from qualifying, which we hadn't been using at all.

Building them properly surfaced several things worth recording:

- **Two leaks in our original features.** `position` was the position at the *end* of the lap, which an overtake during that very lap changes, so it partly encoded the answer. It's now the position at the start of the lap. Fixing it made the 2026 error slightly *worse* (0.787 -> 0.797 s), which is exactly what removing a leak should look like. `fuel_proxy` also counted down from the last *clean* lap instead of the scheduled race distance, so it was wrong whenever a race finished behind the safety car.
- **A leakage test, not just a leakage rule.** Every in-race feature is built only from earlier laps, and we prove it: scramble every lap time, timing-line time and position from lap L onwards and check that lap L's features don't change by a single bit. That runs on a synthetic race in the package self-check and on the real 2026 Italian GP in notebook 03.
- **Qualifying is a knockout, so "fastest lap" isn't "pole".** In 2022 Brazil, Verstappen set the session's fastest lap in dry Q1/Q2 while Magnussen took pole in wet Q3. `quali_rank` sorts by furthest session reached, then time, and matches 9 out of 9 poles we checked by hand. Eight late-2024 sessions have no Q1/Q2/Q3 labels in the data, so those fall back to best-lap order.
- **F1's 107% rule cleans qualifying pace.** A driver whose best lap was a crash or an abort showed up as 38% off pole. Treating anything beyond 107% of pole as non-representative (85 of 2,289 driver-weekends) doubled the correlation between qualifying gap and race pace in 2022-2025, from 0.18 to 0.34.
- **The biggest finding so far:** at team level, qualifying gap predicts race pace with r = 0.62 in 2025 and **r = 0.91 in 2026**, as the field spread from ~3% to ~6% off pole. In 2026, Saturday pace basically *is* Sunday pace, which is precisely the information a team-name feature can't carry across a rule change.

## Nov 26 (cont.) - tuning and the feature ablation

**Tightened the protocol first.** Our first iteration tuned on 2022-2025, which quietly included 2025, the season we then reported as the "unseen" in-era test. Tuning and feature selection now only ever see 2022-2024, and the final feature set is chosen by grouped cross-validation, never by how it does on 2026 (picking whichever set looked best on 2026 would be test-set leakage dressed up as analysis).

**Ablation.** We added the feature groups one at a time and gave every set its own 54-point grid search (GroupKFold by race, 5 folds), so no set was handicapped by parameters tuned for another. Then one extra set: all features but **no team names**.
- The **in-race** group is where the in-era gain is (CV MAE 0.595 -> 0.515 s). Recent pace and gaps are most of what makes a lap predictable.
- **Qualifying barely moved the in-era numbers.** Within a stable era the team columns already carry car pace, so qualifying's value only shows when team identity goes stale.
- **Circuit features helped more than their near-zero correlations suggested**, the signature of interactions.
- **Dropping the team names cost nothing** (0.500 vs 0.498, a twentieth of the fold-to-fold spread), which is the cleanest support yet for "measure, don't name".

**Tuning matters far less than features.** The top grid points were within ~0.002 s of each other against a ~0.039 s fold-to-fold spread, a flat plateau. The best point again sat on a grid edge (`min_child_weight`), so instead of shrugging we extended past it: it bought 0.001 s, confirming the plateau.

**The result that surprised us.** The full feature set cut error by ~20% everywhere (2025: 0.59 -> 0.47 s; 2026: 0.80 -> 0.63 s), with the biggest gains exactly where team names failed (Cadillac 1.54 -> 1.01 s, Aston Martin 1.34 -> 0.97 s). But the *relative* regime-shift jump barely moved (+31% -> +30%). We expected measuring pace to specifically close the 2026 gap; instead it lifted both eras by about the same proportion. Understanding why is the next step.

## Nov 26 (cont.) - dissecting the regime shift

The ablation left us with a puzzle: measuring pace made the model ~20% better in every era but barely moved the relative 2026 jump. We added four analyses to notebook 05 to understand the shift itself rather than just its size.

- **Names vs measurements across the rule change.** Running every ablation set on 2026 (each with its own CV-tuned parameters, chosen before we looked at 2026) showed qualifying and driver features help 2026 about four times more than a normal year (-5% vs -1%): exactly the "team names go stale, measurements don't" effect we hypothesised. Circuit features pull the other way, likely because two 2026 circuits (Madrid, Sepang) have no corner map in the data. Dropping team names doesn't add robustness; they're redundant, not harmful.
- **Error decomposition.** Splitting error into a per-driver-per-race *level* and the lap-to-lap remainder was the most clarifying thing we've done. Half of the first model's 2026 error was simply not knowing how fast each car was (49% vs 21% in 2025); our features cut that to 32%. But with level error removed entirely, 2026 is still ~20% harder lap to lap, so the remaining shift is in race dynamics (energy management is the obvious suspect), which shapes what we look for in notebook 07.
- **Adversarial validation.** A 2025-vs-2026 classifier only reached AUC 0.59, and giving it the lap time didn't help. The shift is subtle and spread across every lap, which is what makes it dangerous.
- **Conformal intervals under shift**, with a normal-year control: 90.5% coverage in 2025 vs 81.7% in 2026. Recalibrating the width on earlier 2026 races restores ~90% on average but not at the oddest circuits. Building the control first mattered: it's what lets us say the 2026 drop is the regime shift and not the method (laps cluster by race, so coverage guarantees only roughly hold even in a normal year).

Small things caught along the way: the coverage figure labelled the 2026 Madrid race "Spanish" (event names lie, which is why we keep a separate circuit column), and notebook 04 now saves every ablation set's tuned parameters so notebook 05 can compare them on 2026 without re-tuning or peeking.

## Nov 26 (cont.) - adaptation, redesigned

We rebuilt the few-shot experiment around the flaws we found in the self-review: 2026 races are now added in **calendar order** (with 5 random orders as a robustness check), and every point is scored on the **same fixed test set** (the last 8 dry races of 2026), so the curve finally compares like with like. We also compared four adaptation strategies, each set by a rule fixed in advance, because tuning them on the test races would be leakage: naive append, upweighting 2026 to the weight of a full season, warm-start fine-tuning (continuing to boost the pre-2026 model on new races) and training on 2026 alone.

The results changed our answer to the research question. With measurement features, retraining barely matters: 11 races of 2026 data buy ~2%, the clever strategies are no better (fine-tuning on one race actually hurts, as 300 trees learn that circuit's quirks), and 2026-only training never catches up with using the old seasons. The most striking result is that a model which **reads the race** through in-race features, never retrained, beats a model without them even after 11 races of retraining (0.59 vs 0.67 s). In our first iteration the story was "the first 2026 race helps a lot, then it plateaus"; now we can say *why*: what that first race was teaching the model, the features now deliver every weekend, and the rest of the gap is lap dynamics that more of the same data can't teach.
