"""Download race lap tables from the TracingInsights season repositories.

Each season repo (github.com/TracingInsights/<year>) also holds telemetry and
radio (GBs), so we sparse-checkout only the race-level lap tables
(``*/Race/session_laptimes.json``, ~20 MB per season).
"""

import glob
import subprocess


def download(year, dest=None):
    """Sparse-checkout the race lap tables for one season into ``ti<year>/``.

    No-op if the data is already present. Returns the destination directory.
    """
    d = dest or f"ti{year}"
    if glob.glob(f"{d}/*/Race/session_laptimes.json"):
        return d
    url = f"https://github.com/TracingInsights/{year}"
    subprocess.run(
        ["git", "clone", "-q", "--filter=blob:none", "--no-checkout", "--depth", "1", url, d],
        check=True,
    )
    subprocess.run(["git", "-C", d, "sparse-checkout", "init", "--no-cone"], check=True)
    subprocess.run(["git", "-C", d, "sparse-checkout", "set", "*/Race/session_laptimes.json"], check=True)
    subprocess.run(["git", "-C", d, "checkout", "-q", "main"], check=True)
    return d


def download_all(seasons):
    """Download every season in ``seasons`` (idempotent)."""
    for y in seasons:
        download(y)
