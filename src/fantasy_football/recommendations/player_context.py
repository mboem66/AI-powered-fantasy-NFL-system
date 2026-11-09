"""Assemble outlook facts from a saved forecast, without rebuilding predictions."""

from datetime import datetime, timezone
from math import isfinite, isclose
from statistics import mean
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from fantasy_football.features.build_features import history_indexes
from fantasy_football.features.recent import previous, average
from fantasy_football.modeling.sequences import SEQUENCE_FIELDS


def unstarted(schedule, now):
    if not schedule or not schedule.get("gametime") or not schedule.get("gameday"):
        return False
    kickoff = datetime.fromisoformat(f"{schedule['gameday']}T{schedule['gametime']}").replace(
        tzinfo=ZoneInfo("America/New_York"))
    return (schedule.get("home_score") is None and schedule.get("away_score") is None
            and kickoff > now)


def matchup_reference(root, cutoff):
    """Reconstruct only pre-cutoff games; retain the source version in the run report."""
    games = pl.read_parquet(root / "data/processed/player_games.parquet")
    schedules = pl.concat([pl.read_parquet(p) for p in
                           sorted((root / "data/raw/schedules").glob("*.parquet"))],
                          how="diagonal_relaxed").filter(pl.col("game_type") == "REG")
    _, defenses, _ = history_indexes(games, schedules, ["QB", "RB", "WR", "TE"], cutoff)
    values = {}
    for key in defenses:
        recent = previous(defenses, key, cutoff, 3, 365)
        available = [r for r in recent if r["allowed_ppr"] is not None]
        values[key] = (average(available, "allowed_ppr"), len(available))
    return values, {r["game_id"]: r for r in schedules.to_dicts()}


def number(value):
    if value is None or not isfinite(float(value)):
        raise ValueError("Missing required player statistic")
    return f"{float(value):.1f}"


def availability_sentence(row, now=None, max_age_hours=24):
    """A retrieval timestamp does not establish when an injury status changed."""
    now = now or datetime.now(timezone.utc)
    status = (row.get("availability_status") or "unknown").strip()
    decision = row.get("availability_decision") or "unknown"
    checked = row.get("availability_checked_at") or "unknown"
    if "identity" in decision or "roster" in decision or "team_change" in decision:
        return "Availability is unconfirmed because the saved roster or identity match needs review."
    if status.lower() in {"not_listed", "not listed"}:
        # Measure freshness when writing the outlook, not when the forecast ran.
        try:
            retrieved = datetime.fromisoformat(str(checked).replace("Z", "+00:00"))
            age_hours = (now - retrieved).total_seconds() / 3600 if retrieved.tzinfo else None
        except (ValueError, TypeError):
            age_hours = None
        if age_hours is None or age_hours < 0:
            return "The saved availability snapshot has no reliable retrieval time; check for updates before kickoff."
        if age_hours > max_age_hours:
            return f"The saved availability snapshot was retrieved {checked} and is outdated; check for updates before kickoff."
        if decision == "include" and not row.get("injury_description"):
            return ""
        return "The saved availability information needs review; check for updates before kickoff."
    if status.lower() in {"unknown", ""}:
        return "The saved snapshot does not confirm game-day availability; check for updates before kickoff."
    detail = str(row.get("injury_description") or "no injury detail supplied").strip().rstrip(".")
    reported = row.get("injury_report_date") or "unknown"
    # Keep the source description, including any older week it names, visibly quoted.
    return (f'The saved report dated {reported} lists {status} ("{detail}"; retrieved {checked}), '
            "which does not confirm current game-day availability.")


def context(prediction, row, sequence, reference, cutoff, now=None, max_age_hours=24):
    if prediction["prediction_method"] != "lstm":
        raise ValueError("Only LSTM predictions are supported")
    for field in ("player_id", "game_id", "season", "week", "position", "sequence_length"):
        if prediction[field] != row[field]:
            raise ValueError(f"Saved forecast mismatch: {field}")
    if not len(sequence) or len(sequence) != row["sequence_length"]:
        raise ValueError("Saved sequence length mismatch")
    last = dict(zip(SEQUENCE_FIELDS, sequence[-1]))
    if not isclose(last["calculated_ppr_points"], row["ppr_last_game"], abs_tol=0.001):
        raise ValueError("Saved last-game PPR mismatch")
    fields = {"QB": ["attempts", "rushing_yards"], "RB": ["carries", "targets", "receptions"],
              "WR": ["targets", "receptions", "receiving_yards"],
              "TE": ["targets", "receptions", "receiving_yards"]}[row["position"]]
    labels = {"attempts": "passing attempts", "rushing_yards": "rushing yards",
              "receiving_yards": "receiving yards"}
    usage = {labels.get(f, f): number(np.mean(sequence[-3:, SEQUENCE_FIELDS.index(f)])) for f in fields}
    projected, recent = number(prediction["predicted_ppr_points"]), number(row["ppr_recent_avg"])
    comparison = "above" if float(projected) > float(recent) else "below" if float(projected) < float(recent) else "equal to"
    facts = {"player": row["player_display_name"], "projected_ppr": projected,
             "recent_ppr_average": recent, "comparison": comparison,
             "last_game_ppr": number(row["ppr_last_game"]),
             "usage_window_appearances": str(min(3, len(sequence))), "usage_averages": usage,
             "season_ppr_average": number(row["ppr_season_avg"])}
    opponent = row["opponent_team"]
    saved = row["opponent_ppr_allowed_recent_avg"]
    rebuilt, count = reference.get((opponent, row["position"]), (None, 0))
    consistent = (saved is None and rebuilt is None) or (
        saved is not None and rebuilt is not None and isclose(saved, rebuilt, abs_tol=0.001))
    if not consistent or count != row["opponent_history_games"]:
        raise ValueError("Matchup history changed since forecast; regenerate predictions before outlooks")
    league = [v for (team, pos), (v, n) in reference.items() if pos == row["position"] and n == 3]
    if count < 3 or len(league) < 20:
        matchup = f"There is insufficient comparable recent history to rate the matchup against {opponent}."
    else:
        avg = mean(league)
        direction = "more favorable" if saved > avg else "more difficult" if saved < avg else "neutral"
        matchup = (f"{opponent} allowed {saved:.1f} PPR per game to opposing {row['position']} groups over its last three games "
                   f"versus a {avg:.1f} league reference, a {direction} matchup indicator rather than a guarantee.")
    return {"writing_facts": facts, "matchup_sentence": matchup,
            "availability_sentence": availability_sentence(row, now, max_age_hours), "feature_cutoff": str(cutoff),
            "availability_source": {k: row.get(k) for k in ("availability_status", "injury_description",
                "injury_report_date", "availability_checked_at", "availability_decision")},
            "league_reference_teams": len(league), "history_game_ids": row["history_game_ids"]}
