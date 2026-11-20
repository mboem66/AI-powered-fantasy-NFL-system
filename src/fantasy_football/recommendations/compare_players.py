"""Compare two players using one saved LSTM run; no forecasts are rebuilt."""

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import tomllib

import polars as pl
import requests

from fantasy_football.config import PROJECT_ROOT, load_settings
from fantasy_football.modeling import lstm
from fantasy_football.recommendations import attribution, comparison_llm
from fantasy_football.recommendations.comparison import load_inputs, take, resolve, prepare_pair
from fantasy_football.recommendations.generate_outlooks import digest, save
from fantasy_football.recommendations.player_context import context, matchup_reference, unstarted


def calibration_file(root):
    paths = sorted((root / "artifacts/task4/uncertainty").glob("[0-9]*/calibration.json"))
    if not paths:
        raise ValueError("First run: python -m fantasy_football.recommendations.uncertainty")
    return paths[-1]


def compare_players(player_a, player_b, week, season, root=PROJECT_ROOT, run=None, use_llm=True):
    """Return a JSON-ready report and save it for terminal or future web use."""
    if run and Path(run).name != run:
        raise ValueError("--run must be a folder name inside artifacts/task3/weekly")
    now = datetime.now(timezone.utc)
    base = root / "artifacts/task3/weekly"
    folders = [base / run] if run else sorted(base.glob("[0-9]*"), reverse=True)
    for folder in folders:
        if not (folder / "summary.json").exists():
            continue
        predictions = pl.read_parquet(folder / "predictions.parquet").filter(
            (pl.col("season") == season) & (pl.col("week") == week))
        if predictions.height:
            break
    else:
        raise ValueError("No saved LSTM forecast for this season/week; run predict_lstm first")
    # Do not silently fall back to older snapshots if either player is missing.
    selected = predictions.to_dicts()
    selected = [selected[resolve(selected, name)] for name in (player_a, player_b)]
    if any(r["prediction_method"] != "lstm" for r in selected):
        raise ValueError("Comparison requires two LSTM predictions, not Linear fallbacks")
    summary = json.loads((folder / "summary.json").read_text())
    created = datetime.fromisoformat(summary["created_at_utc"])
    if created.tzinfo is None or created > now:
        raise ValueError("Forecast has an invalid or future timestamp")
    cutoff = date.fromisoformat(summary["feature_cutoff_exclusive"])
    reference, schedules = matchup_reference(root, cutoff)
    if any(not unstarted(schedules.get(r["game_id"]), now) for r in selected):
        raise ValueError("A selected game has started, finished, or lacks a known kickoff")
    data = load_inputs(folder, "upcoming")
    rows = data["rows"].to_dicts()
    indices = {(r["player_id"], r["game_id"]): i for i, r in enumerate(rows)}
    age = tomllib.loads((root / "config/api_nfl.toml").read_text())["max_snapshot_age_hours"]
    combined, slices, outlook_facts = [], [], []
    for prediction in selected:
        index = indices[(prediction["player_id"], prediction["game_id"])]
        one = take(data, index)
        row = rows[index]
        for game in one["history_ids"][0]:
            if game not in schedules or date.fromisoformat(schedules[game]["gameday"]) >= min(cutoff, row["game_date"]):
                raise ValueError("Sequence includes unknown or post-cutoff history")
        facts = context(prediction, row, one["sequences"][0], reference, cutoff, now, age)
        combined.append(row | prediction)
        slices.append(one)
        outlook_facts.append(facts)
    calibration_path = calibration_file(root)
    calibration = json.loads(calibration_path.read_text())
    if Path(summary["selection_run"]).name != calibration["source_run"]:
        raise ValueError("Calibration belongs to a different selected LSTM run")
    facts = prepare_pair(*combined, slices[0]["sequences"][0], slices[1]["sequences"][0], calibration,
                         now=now, max_age_hours=age)
    for side, outlook in zip(("A", "B"), outlook_facts):
        facts["evidence"][f"{side}.matchup"] = facts["players"][side] + ": " + outlook["matchup_sentence"]
    bundle = lstm.load(folder / "model")
    explanations = {side: attribution.explain(bundle, one, row["lstm_prediction"])
                    for side, one, row in zip(("A", "B"), slices, combined)}
    for side, explanation in explanations.items():
        top = sorted(explanation["contributions_ppr"].items(), key=lambda item: abs(item[1]), reverse=True)[:3]
        facts["evidence"][f"{side}.attribution"] = (facts["players"][side] + "'s largest model contributions relative to the synthetic "
            "training-mean reference are " + ", ".join(f"{k.replace('_', ' ')} ({v:+.1f} PPR)" for k, v in top) +
            "; these are model attributions, not causal explanations.")
    settings = tomllib.loads((root / "config/outlooks.toml").read_text())
    prompt = (root / "config/prompts/player_comparison.txt").read_text(encoding="utf-8")
    assistant = comparison_llm.recommend(settings, prompt, facts) if use_llm else None
    # Generation can cross kickoff; never publish a newly generated start/sit choice then.
    if any(not unstarted(schedules.get(r["game_id"]), datetime.now(timezone.utc)) for r in selected):
        raise ValueError("Game started during comparison; start/sit recommendation withheld")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "prediction_run": str(folder),
              "calibration_file": str(calibration_path), "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
              "prediction_sha256": hashlib.sha256((folder / "predictions.parquet").read_bytes()).hexdigest(),
              "player_ids": [r["player_id"] for r in selected], "facts": facts, "attributions": explanations,
              "settings": settings if use_llm else None, "prompt_sha256": digest(prompt), "assistant": assistant,
              "recommendation": comparison_llm.render(assistant, facts) if assistant else "Statistical comparison only; Qwen was not requested.",
              "availability": [f["availability_source"] for f in outlook_facts]}
    output = root / "artifacts/task4/comparisons" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    report["output"] = str(output)
    save(output / "comparison.json", report)
    (output / "COMPARISON.md").write_text(display(report), encoding="utf-8")
    return report


def display(report):
    facts = report["facts"]
    lines = [f"# Week {facts['week']} start/sit comparison", "", "| Player | LSTM PPR | Estimated 80% range |",
             "| --- | ---: | --- |"]
    for side in ("A", "B"):
        bounds = facts["ranges"][side]
        lines.append(f"| {facts['players'][side]} | {report['attributions'][side]['prediction']:.2f} | {bounds['lower']:.1f} to {bounds['upper']:.1f} |")
    preference = facts["players"].get(facts["model_preference"], "Tie")
    probs = facts["probabilities"]
    lines += ["", f"Higher-projection choice: {preference}.",
              f"Estimated scoring probabilities: A wins {probs['a_outscores_b']:.1%}; B wins {probs['b_outscores_a']:.1%}; tie {probs['tie']:.1%}.",
              "", report["recommendation"], "", "Availability notes:"]
    lines += [facts["evidence"].get(f"{side}.availability", facts["players"][side] + ": No designation in the fresh, matched snapshot.") for side in ("A", "B")]
    lines += ["", "Matchup notes:"] + [facts["evidence"][f"{side}.matchup"] for side in ("A", "B")]
    lines += ["", facts["uncertainty_note"], "", "Model contributions (PPR relative to a synthetic reference):"]
    for side, explanation in report["attributions"].items():
        top = sorted(explanation["contributions_ppr"].items(), key=lambda item: abs(item[1]), reverse=True)[:5]
        lines.append(f"{facts['players'][side]}: " + ", ".join(f"{name} {value:+.2f}" for name, value in top))
    lines += ["", f"Saved: {report.get('output', '')}"]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("player_a")
    parser.add_argument("player_b")
    parser.add_argument("--week", required=True, type=int)
    parser.add_argument("--season", type=int, default=load_settings()["data"]["current_season"])
    parser.add_argument("--run")
    parser.add_argument("--no-llm", action="store_true", help="Inspect statistical comparison without Ollama")
    args = parser.parse_args()
    try:
        print(display(compare_players(args.player_a, args.player_b, args.week, args.season, run=args.run, use_llm=not args.no_llm)))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, str(exc) + "\n")
    except requests.RequestException as exc:
        parser.exit(1, f"Ollama request failed: {exc}. Start Ollama or use --no-llm for the statistical comparison.\n")
