"""Load the raw lap table, run the cleaning funnel, and build the target.

The cleaning logic here is the source of truth for the project: notebook 01
calls these functions and reports the funnel. The headline numbers it produces
(125,909 raw laps -> 90,488 clean laps across 113 races) must not change.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

# Raw JSON columns we keep, in order, and the names we give them.
# Sector times and speed traps are outcomes of the lap itself, so they are kept for
# analysis (notebooks 02 and 07) but never used as features for the same lap.
RAW_COLS = ["season", "event", "drv", "team", "lap", "time", "compound", "stint", "life",
            "pos", "status", "pin", "pout", "iacc", "del", "wAT", "wTT", "wR",
            "sesT", "fresh", "wH", "wWS", "s1", "s2", "s3", "vi1", "vi2", "vfl", "vst"]
NUMERIC_COLS = ["time", "lap", "stint", "life", "pos", "wAT", "wTT",
                "sesT", "wH", "wWS", "s1", "s2", "s3", "vi1", "vi2", "vfl", "vst"]
RENAME = {"drv": "driver", "time": "lap_time_s", "life": "tyre_age", "pos": "position",
          "status": "track_status", "pin": "pit_in", "pout": "pit_out", "iacc": "is_accurate",
          "del": "deleted", "wAT": "air_temp_c", "wTT": "track_temp_c", "wR": "rainfall",
          "sesT": "lap_end_time_s", "fresh": "tyre_fresh", "wH": "humidity", "wWS": "wind_speed",
          "s1": "sector1_s", "s2": "sector2_s", "s3": "sector3_s", "vi1": "speed_i1_kph",
          "vi2": "speed_i2_kph", "vfl": "speed_fl_kph", "vst": "speed_trap_kph"}

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
WET_COMPOUNDS = ["INTERMEDIATE", "WET"]


def load_raw(seasons, root="."):
    """Load every race's ``session_laptimes.json`` into one driver-lap table.

    One row per driver-lap, columns renamed to readable names and numeric
    columns coerced. ``root`` is the directory holding the ``ti<year>/`` folders.
    """
    frames = []
    for y in seasons:
        pattern = os.path.join(root, f"ti{y}", "*", "Race", "session_laptimes.json")
        for f in sorted(glob.glob(pattern)):
            with open(f) as fh:
                df = pd.DataFrame(json.load(fh))
            df["season"] = y
            df["event"] = f.split(os.sep)[-3]  # ti<year>/<event>/Race/session_laptimes.json
            frames.append(df)
    raw = pd.concat(frames, ignore_index=True).replace("None", np.nan).infer_objects()
    raw = raw[RAW_COLS]
    for c in NUMERIC_COLS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return fix_temps(raw.rename(columns=RENAME))


def load_session_info(seasons, root="."):
    """One row per race: start date (for chronological order), round number within
    the season, circuit key (a stable circuit ID, unlike event names) and the
    scheduled race distance in laps."""
    rows = []
    for y in seasons:
        for f in glob.glob(os.path.join(root, f"ti{y}", "*", "Race", "session_info.json")):
            with open(f) as fh:
                d = json.load(fh)
            rows.append({"season": y, "event": f.split(os.sep)[-3],
                         "race_date": pd.Timestamp(d["session_info"]["StartDate"]),
                         "circuit_key": d["session_info"]["Meeting"]["Circuit"]["Key"],
                         "scheduled_laps": d.get("total_laps")})
    info = pd.DataFrame(rows).sort_values("race_date", ignore_index=True)
    info["round"] = info.groupby("season").cumcount() + 1
    return info


def load_quali(seasons, root="."):
    """Every qualifying lap: season, event, driver, team, lap time, deleted flag,
    speed-trap reading, knockout session (Q1/Q2/Q3) and tyre compound."""
    frames = []
    for y in seasons:
        for f in glob.glob(os.path.join(root, f"ti{y}", "*", "Qualifying", "session_laptimes.json")):
            with open(f) as fh:
                q = pd.DataFrame(json.load(fh))
            q["season"], q["event"] = y, f.split(os.sep)[-3]
            frames.append(q)
    q = pd.concat(frames, ignore_index=True).replace("None", np.nan).infer_objects()
    q = q[["season", "event", "drv", "team", "time", "del", "vst", "qs", "compound"]]
    q.columns = ["season", "event", "driver", "team", "lap_time_s", "deleted", "speed_trap_kph", "session", "compound"]
    for c in ["lap_time_s", "speed_trap_kph"]:
        q[c] = pd.to_numeric(q[c], errors="coerce")
    return q


def fix_temps(raw):
    """Weather-sensor glitches: no F1 race runs at or below 0 °C, yet e.g. lap 5 of
    the 2026 Las Vegas GP reads 0.0 °C track temperature for every car while the laps
    either side read ~14 °C. Treat non-positive temperatures as missing and fill
    them with that race's median."""
    raw = raw.copy()
    for c in ["air_temp_c", "track_temp_c"]:
        raw.loc[raw[c] <= 0, c] = np.nan
        raw[c] = raw[c].fillna(raw.groupby(["season", "event"])[c].transform("median"))
    return raw


def wet_races(raw):
    """Set of (season, event) with >5% of laps on intermediate/wet tyres."""
    frac = raw.groupby(["season", "event"]).compound.apply(
        lambda s: s.isin(WET_COMPOUNDS).mean() > 0.05
    )
    return set(frac[frac].index)


def clean_laps(raw):
    """Run the cleaning funnel and return ``(df, funnel, wet_races_set)``.

    Keeps clean, representative green-flag laps:
      recorded lap time -> not lap 1 -> no pit in/out -> green flag ->
      not deleted & accurate -> dry race + dry compound -> no rainfall & tyre
      age known -> within 107% of the stint median.
    """
    funnel = [("All laps", len(raw))]
    df = raw.copy()

    def step(name, mask):
        nonlocal df
        df = df[mask(df)]
        funnel.append((name, len(df)))

    step("Lap time recorded", lambda d: d.lap_time_s.notna())
    step("Not lap 1", lambda d: d.lap > 1)
    step("No pit in/out lap", lambda d: d.pit_in.isna() & d.pit_out.isna())
    step("Green flag only", lambda d: d.track_status.astype(str) == "1")
    step("Not deleted, accurate", lambda d: (d.deleted != True) & (d.is_accurate == True))
    wet = wet_races(raw)
    step("Dry races, dry compounds",
         lambda d: ~d.set_index(["season", "event"]).index.isin(wet) & d.compound.isin(DRY_COMPOUNDS))
    step("No rainfall on lap, tyre age known",
         lambda d: (d.rainfall != True) & d.tyre_age.notna())
    stint_med = df.groupby(["season", "event", "driver", "stint"]).lap_time_s.transform("median")
    step("Within 107% of stint median", lambda d: d.lap_time_s <= 1.07 * stint_med.loc[d.index])

    funnel = pd.DataFrame(funnel, columns=["step", "laps"])
    funnel["% kept"] = (funnel.laps / funnel.laps[0] * 100).round(1)
    return df, funnel, wet


def add_target(df):
    """Add ``race_median_s`` and the target ``lap_delta_s`` (lap time minus the
    race's median clean lap time)."""
    df = df.copy()
    df["race_median_s"] = df.groupby(["season", "event"]).lap_time_s.transform("median")
    df["lap_delta_s"] = df.lap_time_s - df.race_median_s
    return df


def _selfcheck():
    """Synthetic funnel check: each dirty row is dropped for the right reason."""
    def row(**kw):
        base = {"season": 2024, "event": "R1", "driver": "VER", "team": "Ferrari", "lap": 3,
                    "lap_time_s": 90.0, "compound": "MEDIUM", "stint": 1, "tyre_age": 3, "position": 1,
                    "track_status": "1", "pit_in": np.nan, "pit_out": np.nan, "is_accurate": True,
                    "deleted": False, "air_temp_c": 25, "track_temp_c": 35, "rainfall": False}
        base.update(kw)
        return base

    rows = [
        row(lap=2, lap_time_s=90), row(lap=3, lap_time_s=91),
        row(lap=4, lap_time_s=90), row(lap=5, lap_time_s=92),   # 4 clean laps -> kept
        row(lap=1),                                             # lap 1 -> dropped
        row(lap=6, pit_in=1234.5),                              # pit lap -> dropped
        row(lap=7, track_status="4"),                           # safety car -> dropped
        row(lap=8, deleted=True),                               # deleted -> dropped
        row(lap=9, lap_time_s=200),                             # >107% stint median -> dropped
        # a fully wet race -> whole race dropped
        row(event="R2", lap=2, compound="INTERMEDIATE"),
        row(event="R2", lap=3, compound="INTERMEDIATE"),
    ]
    raw = pd.DataFrame(rows)
    df, funnel, wet = clean_laps(raw)
    assert (2024, "R2") in wet, wet
    assert len(df) == 4, (len(df), df.lap.tolist())
    assert (df.lap > 1).all() and df.compound.isin(DRY_COMPOUNDS).all()
    assert list(funnel.laps) == sorted(funnel.laps, reverse=True), "funnel must be non-increasing"
    dft = add_target(df)
    assert abs(dft.groupby(["season", "event"]).lap_delta_s.median()).max() < 1e-9

    # A 0.0 °C glitch is replaced by its own race's median, never another race's.
    t = fix_temps(pd.DataFrame({"season": [2026] * 3 + [2025], "event": ["LV"] * 3 + ["X"],
                                "air_temp_c": [12.0, 12.0, 12.0, 30.0],
                                "track_temp_c": [14.0, 0.0, 15.0, 50.0]}))
    assert t.track_temp_c.tolist() == [14.0, 14.5, 15.0, 50.0], t.track_temp_c.tolist()
    print("clean.py self-check passed")


if __name__ == "__main__":
    _selfcheck()
