"""f1lap: data pipeline for F1 lap-time prediction across the 2026 regulation change.

Modules
-------
download : sparse-checkout race lap tables from the TracingInsights season repos.
clean    : load the raw lap table, run the cleaning funnel, build the target.
features : team-lineage and circuit mapping, fuel proxy, feature encoding.

The notebooks import from here so the cleaning/feature logic lives in one place.
"""

from . import download, clean, features  # noqa: F401
