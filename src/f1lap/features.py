"""Feature encoding: team-lineage and circuit mapping, fuel proxy, one-hot encoding.

Data quirks handled here:
  * Team lineages: AlphaTauri/RB -> Racing Bulls; Alfa Romeo/Kick Sauber/Audi ->
    Sauber/Audi. Cadillac is new in 2026 (an unseen team).
  * Event names are not circuits. In 2026 the 'Spanish GP' moved to a new Madrid
    circuit, Barcelona hosts the 'Barcelona GP', and the 'Bahrain GP' was
    relocated to Sepang, Malaysia. Races are matched by circuit, not event name.
"""

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

# Numeric features fed to the models (one-hot columns are appended by encode_features).
NUM_FEATURES = ["tyre_age", "fuel_proxy", "stint", "position", "air_temp_c", "track_temp_c"]


def add_features(df):
    """Add team_lineage, unseen_team, circuit, and fuel_proxy columns."""
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

    # Laps of fuel remaining: the race's final lap number minus the current lap.
    df["fuel_proxy"] = df.groupby(["season", "event"]).lap.transform("max") - df.lap
    return df


def encode_features(df):
    """One-hot encode compound and team lineage, concatenated with NUM_FEATURES."""
    X = pd.get_dummies(df[["compound", "team_lineage"]], prefix=["cmp", "team"], dtype=int)
    return pd.concat([df[NUM_FEATURES], X], axis=1)


def _selfcheck():
    raw = pd.DataFrame({
        "season": [2024, 2026, 2026, 2024, 2024],
        "event": ["Spanish Grand Prix", "Spanish Grand Prix", "Bahrain Grand Prix",
                  "British Grand Prix", "British Grand Prix"],
        "team": ["AlphaTauri", "Cadillac", "Ferrari", "RB", "RB"],
        "lap": [5, 5, 10, 3, 5],
        "compound": ["SOFT"] * 5,
        "tyre_age": [5, 5, 10, 3, 5], "stint": [1] * 5, "position": [1, 2, 3, 4, 5],
        "air_temp_c": [25] * 5, "track_temp_c": [35] * 5,
    })
    d = add_features(raw)
    assert d.loc[d.team == "AlphaTauri", "team_lineage"].iloc[0] == "Racing Bulls"
    assert d.loc[d.team == "RB", "team_lineage"].iloc[0] == "Racing Bulls"
    assert d.loc[(d.season == 2026) & (d.event == "Spanish Grand Prix"), "circuit"].iloc[0] == "Madrid"
    assert d.loc[(d.season == 2024) & (d.event == "Spanish Grand Prix"), "circuit"].iloc[0] == "Barcelona"
    assert d.loc[(d.season == 2026) & (d.event == "Bahrain Grand Prix"), "circuit"].iloc[0] == "Sepang"
    assert bool(d.loc[d.team == "Cadillac", "unseen_team"].iloc[0]) is True
    # fuel_proxy: British GP has laps {3,5} -> max 5 -> lap 3 has 2 laps of fuel left.
    assert d.loc[(d.event == "British Grand Prix") & (d.lap == 3), "fuel_proxy"].iloc[0] == 2
    assert not d.post_clinch.any()  # none of the synthetic races are post-clinch
    assert (2023, "Las Vegas Grand Prix") in POST_CLINCH and len(POST_CLINCH) == 10
    assert list(encode_features(d).columns[:6]) == NUM_FEATURES
    print("features.py self-check passed")


if __name__ == "__main__":
    _selfcheck()
