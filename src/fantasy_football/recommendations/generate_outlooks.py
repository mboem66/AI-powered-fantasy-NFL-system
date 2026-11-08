"""Generate resumable local player outlooks from one saved LSTM weekly run."""

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np
import polars as pl
import requests

from fantasy_football.config import PROJECT_ROOT
from fantasy_football.config import load_settings
from fantasy_football.recommendations import local_llm
from fantasy_football.recommendations.player_context import context, matchup_reference, unstarted


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def save(path, value):
    # Publish each completed result atomically so interrupted batches can resume.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--season", type=int, default=load_settings()["data"]["current_season"])
    parser.add_argument("--run", help="Saved LSTM weekly run folder name")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.run and Path(args.run).name != args.run:
        parser.error("--run expects a folder name inside artifacts/task3/weekly")
    root = PROJECT_ROOT
    settings = tomllib.loads((root / "config/outlooks.toml").read_text())
    prompt = (root / "config/prompts/player_outlook.txt").read_text(encoding="utf-8")
    base = root / "artifacts/task3/weekly"
    folders = [base / args.run] if args.run else sorted(base.iterdir(), reverse=True)
    for folder in folders:
        if not all((folder / f).is_file() for f in ("summary.json", "predictions.parquet",
                                                  "upcoming_rows.parquet", "upcoming_sequences.npz")):
            continue
        predictions = pl.read_parquet(folder / "predictions.parquet").filter(
            (pl.col("season") == args.season) & (pl.col("week") == args.week))
        if predictions.height:
            break
    else:
        raise ValueError("No completed LSTM run found for this season/week")
    summary = json.loads((folder / "summary.json").read_text())
    cutoff = date.fromisoformat(summary["feature_cutoff_exclusive"])
    reference, schedules = matchup_reference(root, cutoff)
    rows = pl.read_parquet(folder / "upcoming_rows.parquet").to_dicts()
    lookup = {(r["player_id"], r["game_id"]): i for i, r in enumerate(rows)}
    if len(lookup) != len(rows) or predictions.unique(["player_id", "game_id"]).height != predictions.height:
        raise ValueError("Duplicate saved player-game records")
    arrays = np.load(folder / "upcoming_sequences.npz", allow_pickle=False)
    candidates = predictions.filter((pl.col("prediction_method") == "lstm") &
                                    (pl.col("predicted_ppr_points") >= settings["minimum_ppr"])).to_dicts()
    above_threshold = len(candidates)
    now = datetime.now(timezone.utc)
    candidates = [r for r in candidates if unstarted(schedules.get(r["game_id"]), now)]
    unstarted_count = len(candidates)
    if args.limit:
        candidates = candidates[:args.limit]
    model = local_llm.model_digest(settings) if candidates else None
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "artifacts/task4/outlooks" / stamp
    cache = output.parent / "cache"
    output.mkdir(parents=True)
    cache.mkdir(exist_ok=True)
    report = {"prediction_run": str(folder), "created_at_utc": now.isoformat(),
              "season": args.season, "week": args.week, "model_digest": model,
              "counts": {"saved_weekly_predictions": predictions.height, "lstm_above_threshold": above_threshold,
                         "unstarted": unstarted_count, "selected": len(candidates)},
              "settings": settings, "prompt": prompt, "feature_cutoff": str(cutoff),
              "league_reference_note": "Equal-team mean of last-three-game positional totals; reconstructed from current sources before cutoff, not a historical snapshot.",
              "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [root / "data/processed/player_games.parquet", *sorted((root / "data/raw/schedules").glob("*.parquet"))]},
              "results": []}
    save(output / "manifest.json", report)
    for prediction in candidates:
        name = prediction["player_display_name"]
        if not unstarted(schedules.get(prediction["game_id"]), datetime.now(timezone.utc)):
            continue
        try:
            index = lookup[(prediction["player_id"], prediction["game_id"])]
            for game_id in rows[index]["history_game_ids"]:
                if game_id not in schedules or date.fromisoformat(schedules[game_id]["gameday"]) >= min(cutoff, prediction["game_date"]):
                    raise ValueError("Saved sequence contains unknown or post-cutoff game")
            facts = context(prediction, rows[index], arrays["values"][index, :arrays["lengths"][index]], reference, cutoff)
            key = digest({"facts": facts, "prompt": prompt, "settings": settings, "model": model,
                          "prediction_run": str(folder), "validator_version": 2})
            cached = cache / f"{key}.json"
            result = json.loads(cached.read_text()) if cached.exists() else None
            if result is not None:
                result["errors"] = sorted(set(result["errors"] + local_llm.validate(result["draft"], facts["writing_facts"])))
            if result is None or (args.retry_failed and result["errors"]):
                result = local_llm.generate(settings, prompt, facts["writing_facts"])
                if result["errors"]:
                    first = result
                    result = local_llm.generate(settings, prompt + "\nPrevious draft failed these checks: " +
                                                "; ".join(first["errors"]) + ". Correct those issues.", facts["writing_facts"])
                    result["first_attempt"] = first
                result |= {"facts": facts, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
                save(cached, result)
            result = dict(result, player=name, player_id=prediction["player_id"], game_id=prediction["game_id"], cache_key=key)
            if not unstarted(schedules.get(prediction["game_id"]), datetime.now(timezone.utc)):
                result["errors"].append("Game started during generation")
            result["outlook"] = "" if result["errors"] else " ".join([
                result["draft"]["projection"], result["draft"]["performance"],
                facts["matchup_sentence"], facts["availability_sentence"]])
        except (ValueError, KeyError, requests.RequestException) as exc:
            result = {"player": name, "errors": [str(exc)], "outlook": ""}
        report["results"].append(result)
        save(output / "manifest.json", report)
        print(f"{name}: {'REVIEW ' + str(result['errors']) if result['errors'] else 'saved'}", flush=True)
    accepted = [r for r in report["results"] if not r["errors"]]
    lines = [f"# Week {args.week} player outlooks", "", f"Forecast run: {folder.name}", "",
             "Statistical outlooks, not model feature attributions. Availability reflects the saved forecast snapshot.", ""]
    for r in accepted:
        lines.extend([f"## {r['player']}", "", r["outlook"], ""])
    (output / "OUTLOOKS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Accepted: {len(accepted)}; review: {len(report['results']) - len(accepted)}. Output: {output}")


if __name__ == "__main__":
    main()
