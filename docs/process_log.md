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
