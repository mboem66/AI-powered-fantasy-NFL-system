from bisect import bisect_left
from collections import defaultdict
from datetime import timedelta
from statistics import mean
from math import isfinite


def make_index(rows, key):
    """Store sorted histories once instead of scanning all rows per player"""
    grouped = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    index = {}
    for group, values in grouped.items():
        rows = sorted(values, key=lambda r: r["game_date"])
        index[group] = (rows, [r["game_date"] for r in rows])
    return index


def previous(index, key, cutoff, count, history_days):
    rows, dates = index.get(key, ([], []))
    end = bisect_left(dates, cutoff)
    start = bisect_left(dates, cutoff - timedelta(days=history_days))
    return rows[max(start, end - count):end]


def average(rows, field):
    values = [r[field] for r in rows if r.get(field) is not None]
    return mean(values) if values else None


def usage_trend(rows, field):
    """Last appearance minus the preceding two; require all three values."""
    if len(rows) < 3:
        return None
    values = [r.get(field) for r in rows[-3:]]
    if any(value is None or not isfinite(value) for value in values):
        return None
    return values[-1] - mean(values[:2])


def features_for(row, player_index, defense_index, team_index, settings, as_of_date):
    """Return only pregame features, actual points are added separately"""
    count, days = settings["recent_games"], settings["history_days"]
    cutoff = min(row["game_date"], as_of_date)
    recent = previous(player_index, row["player_id"], cutoff, count, days)
    opponents = previous(defense_index, (row["opponent_team"], row["position"]), cutoff, count, days)
    team_games = previous(team_index, row["team"], cutoff, count, days)
    team_game_ids = {r["game_id"] for r in team_games}
    # Look through recent calendar history for participation in those team games.
    appearances = previous(player_index, row["player_id"], cutoff, 100, days)
    participated = any(
        r["game_id"] in team_game_ids and r["team"] == row["team"] and
        any((r.get(c) or 0) > 0 for c in ("offense_snaps", "attempts", "targets", "carries"))
        for r in appearances
    )
    return {
        "ppr_last_game": recent[-1]["calculated_ppr_points"] if recent else None,
        "ppr_recent_avg": average(recent, "calculated_ppr_points"),
        "targets_recent_avg": average(recent, "targets"),
        "carries_recent_avg": average(recent, "carries"),
        "passing_attempts_recent_avg": average(recent, "attempts"),
        "passing_yards_recent_avg": average(recent, "passing_yards"),
        "passing_tds_recent_avg": average(recent, "passing_tds"),
        "snap_share_recent_avg": average(recent, "offense_pct"),
        "targets_trend": usage_trend(appearances, "targets"),
        "carries_trend": usage_trend(appearances, "carries"),
        "snap_share_trend": usage_trend(appearances, "offense_pct"),
        "history_games": len(recent),
        "snap_history_games": sum(r.get("offense_pct") is not None for r in recent),
        "history_crosses_season": any(r["season"] != row["season"] for r in recent),
        "opponent_ppr_allowed_recent_avg": average(opponents, "allowed_ppr"),
        "opponent_history_games": sum(r["allowed_ppr"] is not None for r in opponents),
        "eligible_recent_participation": participated,
    }
