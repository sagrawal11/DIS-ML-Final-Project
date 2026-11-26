"""Feature engineering: from the clean lap table to the model table.

Leakage rule for every feature: a feature for lap L may only use information
available when lap L *starts*. Nothing from lap L itself, later laps, or the race
result. Qualifying happens before the race, so it is always allowed. The
self-check below enforces this by perturbing lap L onwards and asserting that
lap L's features don't move.

Data quirks handled here:
  * Team lineages: AlphaTauri/RB -> Racing Bulls; Alfa Romeo/Kick Sauber/Audi ->
    Sauber/Audi. Cadillac is new in 2026 (an unseen team).
  * Event names are not circuits. In 2026 the 'Spanish GP' moved to a new Madrid
    circuit, Barcelona hosts the 'Barcelona GP', and the 'Bahrain GP' was
    relocated to Sepang, Malaysia. Races are matched by circuit, not event name.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

LINEAGE = {"AlphaTauri": "Racing Bulls", "RB": "Racing Bulls",
           "Alfa Romeo": "Sauber/Audi", "Kick Sauber": "Sauber/Audi", "Audi": "Sauber/Audi"}

# Races run after BOTH world titles were mathematically decided. Clinches:
#   2022 drivers Japan, constructors USA        2023 constructors Japan, drivers Qatar sprint
#   2024 / 2025 constructors or drivers went to the Abu Dhabi finale -> no such races
#   2026 drivers Mexico City, constructors Sao Paulo (Formula1.com / ESPN race reports)
# Used as a robustness check (notebooks 02 and 05), not a filter: champions did not ease off.
POST_CLINCH = {
    (2022, "Mexico City Grand Prix"), (2022, "São Paulo Grand Prix"), (2022, "Abu Dhabi Grand Prix"),
    (2023, "Qatar Grand Prix"), (2023, "United States Grand Prix"), (2023, "Mexico City Grand Prix"),
    (2023, "São Paulo Grand Prix"), (2023, "Las Vegas Grand Prix"), (2023, "Abu Dhabi Grand Prix"),
    (2026, "Las Vegas Grand Prix"),
}

# Numeric features by group; encode_features appends one-hot compound and team to every set.
FEATURE_GROUPS = {
    "base": ["tyre_age", "fuel_proxy", "stint", "position", "air_temp_c", "track_temp_c"],
    "quali": ["quali_gap_pct", "team_quali_gap_pct", "teammate_quali_gap_pct", "quali_rank",
              "quali_speed_trap_delta"],
    "driver": ["racecraft"],
    "race": ["gap_ahead_s", "gap_behind_s", "pos_vs_quali", "recent_pace", "stint_lap", "tyre_fresh"],
    "circuit": ["lap_length_m", "n_corners", "frac_straight", "longest_straight_m"],
    "weather": ["humidity", "wind_speed", "track_temp_delta"],
}
NUM_FEATURES = FEATURE_GROUPS["base"]
RACE = ["season", "event"]
DRIVER_RACE = RACE + ["driver"]
LAP = DRIVER_RACE + ["lap"]


def add_features(df):
    """Add team_lineage, unseen_team, circuit and post_clinch columns."""
    df = df.copy()
    df["team_lineage"] = df.team.replace(LINEAGE)
    train_teams = set(df.loc[df.season < 2026, "team_lineage"])
    df["unseen_team"] = ~df.team_lineage.isin(train_teams)

    # Circuit = event name minus "Grand Prix", with the 2026 relocations fixed.
    df["circuit"] = df.event.str.replace(" Grand Prix", "")
    df.loc[(df.season < 2026) & (df.circuit == "Spanish"), "circuit"] = "Barcelona"
    df.loc[(df.season == 2026) & (df.circuit == "Spanish"), "circuit"] = "Madrid"
    df.loc[(df.season == 2026) & (df.circuit == "Bahrain"), "circuit"] = "Sepang"

    df["post_clinch"] = [(s, e) in POST_CLINCH for s, e in zip(df.season, df.event)]
    return df


def race_context(raw):
    """In-race features for every raw lap, built only from earlier laps.

    position          running position at the start of the lap (= end of the previous lap)
    gap_ahead_s       time to the car one place ahead when this lap starts (NaN for the leader)
    gap_behind_s      time to the car one place behind when this lap starts
    stint_lap         laps since this stint began (1 = out-lap)
    recent_pace       driver's median lap time over the previous 5 laps minus the field's
                      median over the same laps (negative = quicker than the field lately)
    track_temp_delta  track temperature now minus at the start of the race
    """
    r = raw.drop_duplicates(LAP).copy()

    # State at the end of lap L-1 is the state at the start of lap L.
    prev = r[LAP + ["position", "lap_end_time_s"]].copy()
    prev["lap"] += 1
    prev.columns = LAP + ["start_pos", "start_time"]
    at_pos = prev.dropna(subset=["start_pos"]).drop_duplicates(RACE + ["lap", "start_pos"])
    at_pos = at_pos[RACE + ["lap", "start_pos", "start_time"]]
    out = r[LAP].merge(prev, on=LAP, how="left")
    key = RACE + ["lap", "start_pos"]
    out = out.merge(at_pos.assign(start_pos=at_pos.start_pos + 1).rename(columns={"start_time": "t_ahead"}),
                    on=key, how="left")
    out = out.merge(at_pos.assign(start_pos=at_pos.start_pos - 1).rename(columns={"start_time": "t_behind"}),
                    on=key, how="left")
    out["gap_ahead_s"] = out.start_time - out.t_ahead
    out["gap_behind_s"] = out.t_behind - out.start_time
    out = out.rename(columns={"start_pos": "position"}).drop(columns=["start_time", "t_ahead", "t_behind"])

    r["stint_lap"] = r.lap - r.groupby(DRIVER_RACE + ["stint"]).lap.transform("min") + 1
    first_temp = r.sort_values("lap").groupby(RACE).track_temp_c.transform("first")
    r["track_temp_delta"] = r.track_temp_c - first_temp

    # Recent pace uses only per-lap properties to pick representative laps (no stint
    # medians, which would peek at later laps).
    ok = ((r.lap > 1) & r.lap_time_s.notna() & r.pit_in.isna() & r.pit_out.isna()
          & (r.track_status.astype(str) == "1") & (r.is_accurate == True) & (r.deleted != True))
    r["t_ok"] = r.lap_time_s.where(ok)
    pieces = []
    for (season, event), g in r.groupby(RACE):
        piv = g.pivot(index="lap", columns="driver", values="t_ok")
        piv = piv.reindex(range(1, int(piv.index.max()) + 1))
        mine = piv.rolling(5, min_periods=1).median().shift(1)
        field = piv.median(axis=1).rolling(5, min_periods=1).median().shift(1)
        rel = mine.sub(field, axis=0).reset_index().melt(id_vars="lap", var_name="driver", value_name="recent_pace")
        pieces.append(rel.assign(season=season, event=event))
    recent = pd.concat(pieces, ignore_index=True)

    out = out.merge(r[LAP + ["stint_lap", "track_temp_delta"]], on=LAP, how="left")
    return out.merge(recent, on=LAP, how="left")


def quali_features(q):
    """Per driver per race, from that weekend's qualifying (always before the race).

    quali_gap_pct           driver's best lap vs pole, in %
    team_quali_gap_pct      the team's best car vs pole, in % (car pace)
    teammate_quali_gap_pct  driver's best vs teammate's best, in % of pole (driver vs same car)
    quali_rank              knockout order: furthest session reached, then lap time in it
    quali_speed_trap_delta  speed-trap reading on the driver's best lap minus the field median (kph)
    """
    ok = q.lap_time_s.notna() & (q.deleted != True)
    best_idx = q[ok].groupby(DRIVER_RACE).lap_time_s.idxmin()
    b = q.loc[best_idx, DRIVER_RACE + ["team", "lap_time_s", "speed_trap_kph"]].rename(columns={"lap_time_s": "q_best"})
    pole = b.groupby(RACE).q_best.transform("min")
    # F1's own 107% rule: a best lap slower than 107% of pole isn't representative pace
    # (a crash, an abort, or wet running), so the pace features are left missing.
    rep = b.q_best.where(b.q_best <= 1.07 * pole)
    b["quali_gap_pct"] = (rep / pole - 1) * 100
    b["team_quali_gap_pct"] = b.groupby(RACE + ["team"]).quali_gap_pct.transform("min")
    n = rep.groupby([b[c] for c in RACE + ["team"]]).transform("count")
    mate = (rep.groupby([b[c] for c in RACE + ["team"]]).transform("sum") - rep) / (n - 1).where(n > 1)
    b["teammate_quali_gap_pct"] = (rep - mate) / pole * 100
    b["quali_speed_trap_delta"] = (b.speed_trap_kph - b.groupby(RACE).speed_trap_kph.transform("median")).where(rep.notna())

    sess = q.session.map({"Q1": 1, "Q2": 2, "Q3": 3})
    by_driver = [q[c] for c in DRIVER_RACE]
    reached = sess.groupby(by_driver).max().rename("reached")
    q_r = q.assign(sess=sess, reached=sess.groupby(by_driver).transform("max"))
    t_reached = q_r[ok & (q_r.sess == q_r.reached)].groupby(DRIVER_RACE).lap_time_s.min().rename("t_reached")
    k = pd.concat([reached, t_reached, b.set_index(DRIVER_RACE).q_best], axis=1).reset_index()
    # 8 late-2024 sessions carry no Q1/Q2/Q3 labels at all: rank those purely by best lap.
    unlabelled = k.reached.isna()
    k["reached"] = k.reached.fillna(0)
    k["t_reached"] = k.t_reached.fillna(k.q_best.where(unlabelled)).fillna(np.inf)  # no time -> last
    k = k.sort_values(RACE + ["reached", "t_reached"], ascending=[True, True, False, True])
    k["quali_rank"] = k.groupby(RACE).cumcount() + 1
    return b.merge(k[DRIVER_RACE + ["quali_rank"]], on=DRIVER_RACE, how="left")


def racecraft(raw, qf, info, window=10):
    """Driver's average positions gained (quali rank minus finishing position) over
    their previous `window` races, classified finishes only. Shifted by one race,
    so a race's own result never feeds its own feature."""
    last = raw.dropna(subset=["position"]).sort_values("lap").groupby(DRIVER_RACE).tail(1)
    classified = last.lap >= 0.9 * last.groupby(RACE).lap.transform("max")
    fin = last[DRIVER_RACE + ["position"]].assign(classified=classified)
    fin = fin.merge(qf[DRIVER_RACE + ["quali_rank"]], on=DRIVER_RACE, how="left")
    fin = fin.merge(info[RACE + ["race_date"]], on=RACE).sort_values("race_date")
    fin["gained"] = (fin.quali_rank - fin.position).where(fin.classified)
    fin["racecraft"] = fin.groupby("driver").gained.transform(
        lambda s: s.shift(1).rolling(window, min_periods=3).mean())
    return fin[DRIVER_RACE + ["racecraft"]]


def circuit_features(seasons, root="."):
    """Per race, from that race's corner map: lap length (from the track outline,
    stored in decimetres), corner count, share of the lap that is straight, and the
    longest straight section. NaN where the map is missing or inconsistent."""
    rows = []
    for y in seasons:
        for f in glob.glob(os.path.join(root, f"ti{y}", "*", "Race", "corners.json")):
            with open(f) as fh:
                c = json.load(fh)
            row = {"season": y, "event": f.split(os.sep)[-3]}
            x, yy = np.asarray(c.get("TrackX") or [], float), np.asarray(c.get("TrackY") or [], float)
            if len(x) > 1:
                row["lap_length_m"] = np.hypot(np.diff(np.r_[x, x[:1]]), np.diff(np.r_[yy, yy[:1]])).sum() / 10
            if c.get("CornerNumber"):
                row["n_corners"] = len(c["CornerNumber"])
            typ, a, b = (c.get(k) or [] for k in ["SectionType", "SectionStartDistance", "SectionEndDistance"])
            if typ and len(typ) == len(a) == len(b):
                seg = pd.DataFrame({"type": typ, "len": np.subtract(b, a)})
                straight = seg.len[seg.type == "Straight"]
                row["frac_straight"] = straight.sum() / seg.len.sum()
                row["longest_straight_m"] = straight.max()
            rows.append(row)
    return pd.DataFrame(rows)


def build_model_table(clean, raw, info, quali_laps, circuits):
    """Join every feature group onto the clean laps. One row per clean lap, always."""
    df = add_features(clean).rename(columns={"position": "end_position"})
    df = df.merge(info, on=RACE, how="left")
    df["fuel_proxy"] = df.scheduled_laps - df.lap            # laps of fuel still on board
    df = df.merge(race_context(raw), on=LAP, how="left")
    qf = quali_features(quali_laps)
    df = df.merge(qf.drop(columns=["team", "q_best", "speed_trap_kph"]), on=DRIVER_RACE, how="left")
    df["pos_vs_quali"] = df.quali_rank - df.position         # + = running ahead of where they qualified
    df = df.merge(racecraft(raw, qf, info), on=DRIVER_RACE, how="left")
    df = df.merge(circuits, on=RACE, how="left")
    df["tyre_fresh"] = df.tyre_fresh.astype(float)
    assert len(df) == len(clean), "a merge duplicated laps"
    return df


def encode_features(df, groups=("base",), team=True):
    """Numeric features from `groups`, plus one-hot compound and (if `team`) one-hot
    team lineage. team=False gives a 'measurements only' model that never sees
    which car it is, only how fast that car has been measured to be."""
    cols = [c for g in groups for c in FEATURE_GROUPS[g]]
    cats = ["compound", "team_lineage"] if team else ["compound"]
    X = pd.get_dummies(df[cats], prefix=["cmp", "team"][:len(cats)], dtype=int)
    return pd.concat([df[cols].astype(float), X], axis=1)


def _synthetic_race(rng):
    """Three cars, 10 laps; lap end times are cumulative so gaps are consistent."""
    rows = []
    for d, base in [("AAA", 90.0), ("BBB", 90.5), ("CCC", 91.0)]:
        t = 0.0
        for lap in range(1, 11):
            lt = base + rng.normal(0, 0.2)
            t += lt
            rows.append({"season": 2024, "event": "R", "driver": d, "lap": lap, "lap_time_s": lt,
                         "lap_end_time_s": t, "stint": 1.0 if lap < 6 else 2.0, "track_status": "1",
                         "pit_in": np.nan, "pit_out": np.nan, "is_accurate": True, "deleted": False,
                         "track_temp_c": 30.0 + lap / 10})
    r = pd.DataFrame(rows)
    r["position"] = r.groupby("lap").lap_end_time_s.rank().astype(float)
    return r


def _selfcheck():
    raw = pd.DataFrame({
        "season": [2024, 2026, 2026, 2024, 2024],
        "event": ["Spanish Grand Prix", "Spanish Grand Prix", "Bahrain Grand Prix",
                  "British Grand Prix", "British Grand Prix"],
        "team": ["AlphaTauri", "Cadillac", "Ferrari", "RB", "RB"],
    })
    d = add_features(raw)
    assert d.loc[d.team == "AlphaTauri", "team_lineage"].iloc[0] == "Racing Bulls"
    assert d.loc[(d.season == 2026) & (d.event == "Spanish Grand Prix"), "circuit"].iloc[0] == "Madrid"
    assert d.loc[(d.season == 2024) & (d.event == "Spanish Grand Prix"), "circuit"].iloc[0] == "Barcelona"
    assert d.loc[(d.season == 2026) & (d.event == "Bahrain Grand Prix"), "circuit"].iloc[0] == "Sepang"
    assert bool(d.loc[d.team == "Cadillac", "unseen_team"].iloc[0]) is True
    assert not d.post_clinch.any() and len(POST_CLINCH) == 10

    # Race context: gaps come from the previous lap's end times.
    rng = np.random.RandomState(0)
    r = _synthetic_race(rng)
    ctx = race_context(r).set_index(["driver", "lap"])
    e = r.set_index(["driver", "lap"]).lap_end_time_s
    lead2 = r[(r.lap == 2) & (r.position == 1)].driver.iloc[0]
    second2 = r[(r.lap == 2) & (r.position == 2)].driver.iloc[0]
    assert np.isclose(ctx.loc[(second2, 3), "gap_ahead_s"], e[(second2, 2)] - e[(lead2, 2)])
    assert np.isnan(ctx.loc[(lead2, 3), "gap_ahead_s"]) and ctx.loc[(lead2, 3), "position"] == 1
    assert ctx.loc[("AAA", 7), "stint_lap"] == 2              # stint 2 starts on lap 6

    # LEAKAGE CHECK: scramble every outcome from lap L onwards; lap L's features must not move.
    L = 6
    r2 = r.copy()
    later = r2.lap >= L
    r2.loc[later, "lap_time_s"] *= rng.uniform(0.9, 1.3, later.sum())
    r2.loc[later, "lap_end_time_s"] += rng.uniform(0, 50, later.sum())
    r2.loc[later, "position"] = rng.permutation(r2.loc[later, "position"].values)
    a = race_context(r).set_index(LAP).sort_index()
    b = race_context(r2).set_index(LAP).sort_index()
    at_L = a.index.get_level_values("lap") == L
    pd.testing.assert_frame_equal(a[at_L], b[at_L])

    # Qualifying: a Q3 driver ranks above a Q2 driver even with a slower lap.
    q = pd.DataFrame({"season": 2024, "event": "R", "driver": ["AAA", "BBB", "CCC", "CCC"],
                      "team": ["T1", "T1", "T2", "T2"], "lap_time_s": [80.0, 79.5, 80.2, 80.5],
                      "deleted": False, "speed_trap_kph": [320, 322, 318, 318],
                      "session": ["Q3", "Q2", "Q1", "Q3"]})
    qf = quali_features(q).set_index("driver")
    assert qf.quali_rank.to_dict() == {"AAA": 1, "CCC": 2, "BBB": 3}, qf.quali_rank.to_dict()
    assert np.isclose(qf.loc["BBB", "quali_gap_pct"], 0) and qf.loc["AAA", "team_quali_gap_pct"] == 0
    assert np.isclose(qf.loc["AAA", "teammate_quali_gap_pct"], 0.5 / 79.5 * 100)
    # No session labels -> rank by best lap; a lap beyond 107% of pole gets no pace features.
    q2 = q.assign(session=np.nan, lap_time_s=[80.0, 79.5, 90.0, 95.0])
    qf2 = quali_features(q2).set_index("driver")
    assert qf2.quali_rank.to_dict() == {"BBB": 1, "AAA": 2, "CCC": 3}, qf2.quali_rank.to_dict()
    assert np.isnan(qf2.loc["CCC", "quali_gap_pct"]) and np.isnan(qf2.loc["CCC", "quali_speed_trap_delta"])

    # Racecraft never sees the current race's result.
    races = pd.concat([_synthetic_race(np.random.RandomState(i)).assign(event=f"R{i}") for i in range(6)])
    info = pd.DataFrame({"season": 2024, "event": [f"R{i}" for i in range(6)],
                         "race_date": pd.date_range("2024-03-01", periods=6, freq="7D")})
    qr = pd.DataFrame([{"season": 2024, "event": f"R{i}", "driver": d, "quali_rank": k}
                       for i in range(6) for k, d in enumerate(["CCC", "BBB", "AAA"], 1)])
    rc = racecraft(races, qr, info).set_index(["event", "driver"]).racecraft
    flipped = races.copy()
    last5 = (flipped.event == "R5") & (flipped.lap == 10)
    flipped.loc[last5, "position"] = flipped.loc[last5, "position"].values[::-1]
    rc2 = racecraft(flipped, qr, info).set_index(["event", "driver"]).racecraft
    assert rc.loc["R5"].equals(rc2.loc["R5"]) and rc.loc["R2"].isna().all() and rc.loc["R3"].notna().all()

    X = encode_features(pd.DataFrame({c: [1.0] for c in FEATURE_GROUPS["base"]}
                                     | {"compound": ["SOFT"], "team_lineage": ["Ferrari"]}))
    assert list(X.columns[:6]) == NUM_FEATURES and "team_Ferrari" in X.columns
    X_no_team = encode_features(pd.DataFrame({c: [1.0] for c in FEATURE_GROUPS["base"]}
                                             | {"compound": ["SOFT"], "team_lineage": ["Ferrari"]}), team=False)
    assert not any(c.startswith("team_") for c in X_no_team.columns) and "cmp_SOFT" in X_no_team.columns
    print("features.py self-check passed")


if __name__ == "__main__":
    _selfcheck()
