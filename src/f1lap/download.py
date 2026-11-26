"""Download lap tables and session metadata from the TracingInsights season repositories.

Each season repo (github.com/TracingInsights/<year>) also holds telemetry and
radio (GBs), so we sparse-checkout only the files listed in FILES. Each season is
pinned to an exact commit, so a fresh clone reproduces our numbers even though
TracingInsights keeps adding races to the current season.
"""

import os
import subprocess

# Snapshot per season, frozen on 2026-11-26 (2026 includes the Las Vegas GP).
SNAPSHOT = {
    2022: "df396d091c4a40a3a6b733631da3497f8c91d89f",
    2023: "737c84d3ae14b9883a671ca248d015b69cf1cb58",
    2024: "d1543a71370d9b617b627319f8ad80b2f85dfd5e",
    2025: "dfb3ea3f50b69857fe8e40cabe5e899743c59f6a",
    2026: "f2c7692c87817a48fa7d85749f2767743d62be70",
}

FILES = [
    "*/Race/session_laptimes.json",        # the race laps we model
    "*/Race/session_info.json",            # round number, circuit key, scheduled laps
    "*/Race/corners.json",                 # corner positions and speed class per circuit
    "*/Qualifying/session_laptimes.json",  # one-lap pace for each car and driver
]


def _git(*args, check=True):
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True).stdout.strip()


def download(year, dest=None):
    """Sparse-checkout FILES for one season at its pinned commit into ``ti<year>/``.

    Idempotent: re-running on an up-to-date checkout fetches nothing.
    Returns the destination directory.
    """
    d = dest or f"ti{year}"
    sha = SNAPSHOT[year]
    if not os.path.isdir(os.path.join(d, ".git")):
        _git("init", "-q", d)
        _git("-C", d, "remote", "add", "origin", f"https://github.com/TracingInsights/{year}")
    _git("-C", d, "sparse-checkout", "set", "--no-cone", *FILES)
    if _git("-C", d, "rev-parse", "HEAD", check=False) != sha:
        _git("-C", d, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", sha)
        _git("-C", d, "checkout", "-q", "--detach", sha)
    return d


def download_all(seasons):
    """Download every season in ``seasons`` (idempotent)."""
    for y in seasons:
        download(y)
