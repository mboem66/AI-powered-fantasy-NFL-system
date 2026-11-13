"""Combine saved pregame forecasts and available results for one NFL week."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from math import isfinite, sqrt
import json
from zoneinfo import ZoneInfo

import polars as pl

from fantasy_football.config import PROJECT_ROOT, load_settings


def timestamp(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Forecast timestamp must include a timezone")
    return parsed


def kickoff(row):
    if not row.get("gameday") or not row.get("gametime"):
        return None
    return datetime.fromisoformat(f"{row['gameday']}T{row['gametime']}").replace(
        tzinfo=ZoneInfo("America/New_York"))


EXPERIMENT_MODELS = ("linear_control", "lstm_control", "deep_lstm", "deep_transformer", "transformer_lstm")


def select_forecasts(root, season, week, schedules, now, experiments=False, version=None):
    """Choose independently for each pipeline; never scan backtest folders."""
    selected, audit = {}, Counter()
    sources = []
    if experiments:
        manifests = sorted((root / "artifacts/model_experiments/weekly").glob("*/summary.json"))
        versions = {json.loads(p.read_text())["experiment_version"] for p in manifests if not p.parent.name.startswith("pending_")}
        if version is None and len(versions) > 1:
            raise ValueError(f"Multiple experiment versions: {sorted(versions)}. Select one with --version.")
        for manifest in manifests:
            info = json.loads(manifest.read_text())
            if manifest.parent.name.startswith("pending_") or (version and info["experiment_version"] != version):
                continue
            for model in EXPERIMENT_MODELS:
                sources.append((model, manifest.parent / model / "predictions.parquet"))
    else:
        for task, model in [("task2", "linear"), ("task3", "lstm")]:
            sources.extend((model, p) for p in sorted((root / f"artifacts/{task}/weekly").glob("*/predictions.parquet"))
                           if (p.parent / "summary.json").exists() and not p.parent.name.startswith("pending_"))
    for model, path in sources:
        if path.exists():
            frame = pl.read_parquet(path).filter((pl.col("season") == season) & (pl.col("week") == week))
            if frame.unique(["player_id", "game_id"]).height != frame.height:
                raise ValueError(f"Duplicate player-game records in {path}")
            for row in frame.to_dicts():
                audit["saved_forecast_rows"] += 1
                game = schedules.get(row["game_id"])
                start = kickoff(game) if game else None
                if start is None:
                    audit["missing_kickoff"] += 1
                    continue
                try:
                    saved = timestamp(row["predicted_at_utc"])
                    valid = isfinite(row["predicted_ppr_points"])
                except (ValueError, TypeError, KeyError):
                    audit["invalid_forecast"] += 1
                    continue
                if saved >= start or saved > now or not valid:
                    audit["post_kickoff_future_or_invalid"] += 1
                    continue
                key = (row["player_id"], row["game_id"], model)
                if key not in selected or saved > timestamp(selected[key]["predicted_at_utc"]):
                    selected[key] = row | {"source_run": path.parent.parent.name if experiments else path.parent.name}
    return selected, dict(audit)


def metric(rows, model, scope, position):
    errors = [r[f"{model}_projected_ppr"] - r["actual_ppr"] for r in rows]
    return {"scope": scope, "model": model, "position": position, "records": len(errors),
            "mae": sum(abs(e) for e in errors) / len(errors) if errors else None,
            "rmse": sqrt(sum(e * e for e in errors) / len(errors)) if errors else None}


def build(root=PROJECT_ROOT, season=2026, week=9, now=None, experiments=False, version=None):
    now = now or datetime.now(timezone.utc)
    model_names = EXPERIMENT_MODELS if experiments else ("linear", "lstm")
    schedule_frame = pl.read_parquet(root / f"data/raw/schedules/{season}.parquet").filter(
        (pl.col("season") == season) & (pl.col("week") == week) & (pl.col("game_type") == "REG"))
    schedules = {r["game_id"]: r for r in schedule_frame.to_dicts()}
    selected, audit = select_forecasts(root, season, week, schedules, now, experiments, version)
    if not selected:
        raise ValueError("No eligible saved pregame predictions found for this week")
    actual_frame = pl.read_parquet(root / "data/processed/player_games.parquet").filter(
        (pl.col("season") == season) & (pl.col("week") == week))
    if actual_frame.unique(["player_id", "game_id"]).height != actual_frame.height:
        raise ValueError("Duplicate player-game actual results")
    actuals = {(r["player_id"], r["game_id"]): r["calculated_ppr_points"] for r in actual_frame.to_dicts()}
    rows = []
    for player, game_id in sorted({(p, g) for p, g, _ in selected}):
        forecasts = {model: selected.get((player, game_id, model)) for model in model_names}
        identity = max((f for f in forecasts.values() if f), key=lambda f: timestamp(f["predicted_at_utc"]))
        game = schedules[game_id]
        start = kickoff(game)
        finished = game.get("home_score") is not None and game.get("away_score") is not None and start <= now
        actual = actuals.get((player, game_id))
        # Do not convert an absent player record into a zero or score a future game.
        actual = actual if finished and actual is not None and isfinite(actual) else None
        row = {k: identity[k] for k in ("player_id", "player_display_name", "position", "team", "opponent_team", "game_id")}
        row |= {"season": season, "week": week, "game_date": game["gameday"],
                "kickoff_utc": start.astimezone(timezone.utc).isoformat(), "actual_ppr": actual,
                "result_status": "result_available" if actual is not None else "upcoming" if start > now else "awaiting_statistics"}
        for model, forecast in forecasts.items():
            value = forecast["predicted_ppr_points"] if forecast else None
            row[f"{model}_projected_ppr"] = value
            row[f"{model}_absolute_error"] = abs(value - actual) if value is not None and actual is not None else None
            for output, source in [("saved_at", "predicted_at_utc"), ("run", "source_run"),
                                   ("availability_status", "availability_status"), ("availability_flag", "availability_decision"),
                                   ("injury_report_date", "injury_report_date"), ("availability_checked_at", "availability_checked_at")]:
                row[f"{model}_{output}"] = forecast.get(source) if forecast else None
            row[f"{model}_method"] = (forecast.get("prediction_method", "linear") if forecast else None)
        rows.append(row)
    metrics = []
    matched = [r for r in rows if r["actual_ppr"] is not None]
    # Task 3 can contain Linear fallback predictions. Keep them visible, but do
    # not count them as LSTM performance or as genuine head-to-head LSTM rows.
    eligible = {m: [r for r in matched if r[f"{m}_projected_ppr"] is not None and
                    (m != "lstm" or r["lstm_method"] == "lstm")] for m in model_names}
    common = [r for r in matched if all(r[f"{m}_projected_ppr"] is not None for m in model_names)
              and (experiments or r["lstm_method"] == "lstm")]
    for scope in ("all_available", "same_player_games"):
        for model in model_names:
            pool = common if scope == "same_player_games" else eligible[model]
            for position in ("overall", "QB", "RB", "WR", "TE"):
                group = pool if position == "overall" else [r for r in pool if r["position"] == position]
                metrics.append(metric(group, model, scope, position))
    report = {"season": season, "week": week, "created_at_utc": now.isoformat(),
              "experimental_comparison": experiments, "experiment_version_filter": version,
              "unique_player_games": len(rows), "result_status_counts": dict(Counter(r["result_status"] for r in rows)),
              "forecast_counts": {m: sum(r[f"{m}_projected_ppr"] is not None for r in rows) for m in model_names},
              "lstm_fallback_rows": sum(r.get("lstm_method") == "linear_fallback" for r in rows),
              "selection_audit": audit, "metrics": metrics,
              "note": "Latest saved pre-kickoff prediction per pipeline; current saved schedules define kickoff. Same-player comparisons may use different forecast times. Missing actuals remain missing; retrospective backtests excluded."}
    output = root / ("artifacts/model_experiments/reports" if experiments else "artifacts/weekly_reports") / f"{season}_week_{week:02d}" / now.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    pl.DataFrame(rows, infer_schema_length=None).sort("game_date", "team", "position", "player_display_name").write_csv(output / "players.csv")
    pl.DataFrame(metrics).write_csv(output / "metrics.csv")
    misses = [{"player": r["player_display_name"], "position": r["position"], "model": m,
               "projected_ppr": r[f"{m}_projected_ppr"], "actual_ppr": r["actual_ppr"],
               "absolute_error": r[f"{m}_absolute_error"]} for m, pool in eligible.items() for r in pool]
    report["largest_misses"] = sorted(misses, key=lambda r: r["absolute_error"], reverse=True)[:10]
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{season} Week {week}: {len(rows)} unique player-games")
    print("Forecasts:", report["forecast_counts"], "| Results:", report["result_status_counts"])
    with pl.Config(float_precision=3, tbl_rows=20, tbl_cols=-1, tbl_width_chars=150, tbl_formatting="ASCII_MARKDOWN"):
        print(pl.DataFrame(metrics).filter(pl.col("position") == "overall"))
        if report["largest_misses"]:
            print("Largest prediction misses (available results):")
            print(pl.DataFrame(report["largest_misses"]))
    print("Report:", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--season", type=int, default=load_settings()["data"]["current_season"])
    parser.add_argument("--experiments", action="store_true", help="Compare the five models from shared experimental runs")
    parser.add_argument("--version", help="Experimental version ID when multiple configurations exist")
    args = parser.parse_args()
    build(season=args.season, week=args.week, experiments=args.experiments, version=args.version)
