"""Command-line entry point for historical downloads and weekly refreshes."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
from pathlib import Path

import polars as pl

from fantasy_football.config import PROJECT_ROOT, load_settings
from fantasy_football.data.columns import load_column_selection, select_columns
from fantasy_football.data.sources import (
    REFERENCE_DATASETS, SEASON_DATASETS, configure_source, fetch_dataset,
)
from fantasy_football.data.storage import save_parquet, validate_download

LOGGER = logging.getLogger(__name__)


def download_one(name: str, season: int | None, root: Path, refresh: bool) -> dict:
    """Report each result explicitly; a failed refresh never means success."""
    label = str(season) if season is not None else "latest"
    path = root / "data" / "raw" / name / f"{label}.parquet"
    result = {"dataset": name, "season": season, "path": str(path.relative_to(root))}
    try:
        cached = path.exists() and not refresh
        frame = pl.read_parquet(path) if cached else fetch_dataset(name, season)
        validate_download(frame, season)
        source_columns = frame.columns
        frame = select_columns(frame, name, load_column_selection())
        if not cached or frame.columns != source_columns:
            save_parquet(frame, path)
        result |= {
            "status": "cached" if cached else "downloaded", "rows": frame.height,
            "removed_columns": [c for c in source_columns if c not in frame.columns],
            "schema": {key: str(value) for key, value in frame.schema.items()},
        }
        if not cached:
            result["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
        return result
    except Exception as error:
        # Keep downloading independent tables, but mark the run incomplete.
        LOGGER.warning("%s %s: %s", name, label, error)
        return result | {
            "status": "failed", "error": str(error),
            "previous_file_preserved": path.exists(),
        }


def main(argv: list[str] | None = None) -> int:
    """Choose seasons, download each table, and save an audit report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["history", "current", "all"], default="current")
    parser.add_argument("--refresh", action="store_true", help="Replace cached historical data too")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = load_settings()["data"]
    seasons = []
    if args.mode in {"history", "all"}:
        seasons.extend(range(settings["historical_start"], settings["historical_end"] + 1))
    if args.mode in {"current", "all"}:
        seasons.append(settings["current_season"])

    configure_source()
    results = []
    for season in seasons:
        for name in SEASON_DATASETS:
            refresh = args.refresh or season == settings["current_season"]
            result = download_one(name, season, PROJECT_ROOT, refresh)
            results.append(result)
            LOGGER.info("%s %s: %s", name, season, result["status"])
    for name in REFERENCE_DATASETS:
        results.append(download_one(name, None, PROJECT_ROOT, refresh=True))

    now = datetime.now(timezone.utc)
    report = {
        "run_at_utc": now.isoformat(), "mode": args.mode,
        "nflreadpy_version": version("nflreadpy"),
        "source": "https://github.com/nflverse/nflverse-data",
        "results": results,
    }
    folder = PROJECT_ROOT / "data" / "manifests"
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / f"download_{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    failures = sum(result["status"] == "failed" for result in results)
    LOGGER.info("Report: %s; failed datasets: %s", report_path, failures)
    # Before games are published, current-season stats may not exist yet.
    # Keep that visible instead of filling missing results with invented zeros.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
