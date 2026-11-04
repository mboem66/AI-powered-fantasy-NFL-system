"""Apply current availability to future predictions, never historical training."""

from datetime import datetime
import json
from pathlib import Path
import re
import tomllib
from zoneinfo import ZoneInfo

import polars as pl

DETAILS = {"api_player_id": pl.Int64, "availability_status": pl.String,
           "injury_description": pl.String, "injury_report_date": pl.String,
           "availability_checked_at": pl.String, "availability_decision": pl.String}
REFRESH = "Run python -m fantasy_football.data.refresh_availability first."


def report_applies(player, entry, now):
    """Only use an explicit, recent upcoming-week report to withhold a forecast."""
    description = entry.get("injury_description") or ""
    weeks = re.findall(r"\bweek\s+(\d+)\b", description, re.I)
    if not weeks or any(int(week) != player.get("week") for week in weeks):
        return False
    # A return estimate or negated phrase is not an explicit absence report.
    absent = re.search(r"(?:^|[.;]\s*| - )(?:ruled out|out|inactive|suspended)\s+for\s+week\s+(\d+)\b",
                       description, re.I)
    if not absent or int(absent[1]) != player.get("week"):
        return False
    try:
        report_date = datetime.fromisoformat(entry.get("injury_date") or "").date()
    except ValueError:
        return False
    today = now.astimezone(ZoneInfo("America/New_York")).date()
    return report_date <= today and 0 <= (player["game_date"] - report_date).days <= 7


def decision(player, entry, now, options):
    if (player["game_date"] - now.astimezone(ZoneInfo("America/New_York")).date()).days > options["max_game_days_ahead"]:
        return "review_distant_game"
    if entry is None or entry["mapping_status"] != "matched":
        return "review_identity_or_roster"
    if entry["team"] != player["team"]:
        return "review_team_change"
    group = (entry.get("roster_group") or "").strip().lower()
    status = entry["injury_status"].strip().lower()
    if status in {"out", "inactive", "injured reserve", "ir", "suspended"}:
        return "reported_unavailable" if report_applies(player, entry, now) else "review_report_timing"
    # Broad roster groups and unfamiliar statuses are context, not confirmed absences.
    if group != "offense":
        return "review_roster_group"
    if status not in {"not_listed", "questionable", "doubtful", "probable", "day-to-day", "day to day"}:
        return "review_injury_status"
    return "include"


def apply_availability(upcoming, root, settings, now):
    config = root / "config/api_nfl.toml"
    if not config.exists():
        return upcoming, {"enabled": False}
    with config.open("rb") as stream:
        options = tomllib.load(stream)
    if not options["enabled"]:
        return upcoming, {"enabled": False}
    parent = root / "data/api_nfl"
    if not (parent / "latest.json").exists():
        raise ValueError("No API-NFL availability snapshot. " + REFRESH)
    name = json.loads((parent / "latest.json").read_text(encoding="utf-8"))["snapshot"]
    if not isinstance(name, str) or not name or name in {".", ".."} or name != Path(name).name:
        raise ValueError("Invalid availability snapshot path")
    folder = parent / name
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    age = (now - datetime.fromisoformat(summary["started_at_utc"])).total_seconds() / 3600
    if summary["season"] != settings["data"]["current_season"] or summary["teams"] != 32:
        raise ValueError("Availability snapshot has incorrect season or incomplete team coverage. " + REFRESH)
    if not 0 <= age <= options["max_snapshot_age_hours"]:
        raise ValueError(f"Availability snapshot is stale ({age:.1f} hours) or future-dated. " + REFRESH)
    source = pl.read_parquet(folder / "players.parquet")
    entries = {}
    for row in source.filter(pl.col("player_id").is_not_null()).to_dicts():
        pid = row["player_id"]
        entries[pid] = row if pid not in entries else row | {"mapping_status": "needs_review"}
    audit = []
    for player in upcoming.to_dicts():
        entry = entries.get(player["player_id"])
        outcome = decision(player, entry, now, options)
        # Never attach an arbitrary duplicate or another team's injury report.
        if entry and (entry["mapping_status"] != "matched" or entry["team"] != player["team"]):
            entry = None
        audit.append(player | {
            "api_player_id": entry["api_player_id"] if entry else None,
            "availability_status": entry["injury_status"] if entry else "unknown",
            "injury_description": entry.get("injury_description") if entry else None,
            "injury_report_date": entry.get("injury_date") if entry else None,
            "availability_checked_at": entry["fetched_at_utc"] if entry else None,
            "availability_decision": outcome,
        })
    frame = pl.DataFrame(audit, schema=dict(upcoming.schema) | DETAILS)
    review = parent / "reviews" / f"{now.strftime('%Y%m%dT%H%M%S%fZ')}.csv"
    review.parent.mkdir(parents=True, exist_ok=True)
    frame.select("player_id", "player_display_name", "team", "position", "game_id", "game_date",
                 *DETAILS).write_csv(review)
    # Keep review flags in forecasts. Uncertainty alone must not remove a player.
    included = frame.filter(~pl.col("availability_decision").is_in(["reported_unavailable", "review_distant_game"]))
    print(f"Availability: {included.height}/{upcoming.height} candidates included; review: {review}")
    return included, {"enabled": True, "snapshot": str(folder), "age_hours": age,
                      "candidate_count": upcoming.height, "included_count": included.height,
                      "decisions": frame.group_by("availability_decision").len().to_dicts(),
                      "review_file": str(review), "note": "Uncertain records are retained with flags. Not listed is not confirmed active. Only explicit recent upcoming-week unavailable reports are withheld."}


def prediction_columns(upcoming):
    return [c for c in DETAILS if c in upcoming.columns]
