"""Ask one player/week question using the local knowledge snapshot and Qwen."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib

import polars as pl
import requests

from fantasy_football.config import PROJECT_ROOT, load_settings
from fantasy_football.rag.answer import generate
from fantasy_football.rag.build_index import source_run, hash_file
from fantasy_football.rag.retrieve import load_index, retrieve
from fantasy_football.recommendations.generate_outlooks import save, digest
from fantasy_football.recommendations.player_context import unstarted


def checked_index(root, season, week):
    """Shared source checks for single-player answers and conversational comparisons."""
    folder, manifest, records = load_index(root, season, week)
    latest, _ = source_run(root, season, week)
    if latest.resolve() != Path(manifest["prediction_run"]).resolve():
        raise ValueError("Newer forecasts exist for this week. Rebuild the Task 5 index first")
    for relative, expected in manifest["sources"].items():
        path = root / relative
        if not path.exists() or hash_file(path) != expected:
            raise ValueError("An indexed source changed or disappeared. Rebuild the Task 5 index")
    return folder, manifest, records


def ask(question, week, season, player=None, retrieval_only=False, root=PROJECT_ROOT, now=None):
    now = now or datetime.now(timezone.utc)
    options = tomllib.loads((root / "config/rag.toml").read_text())
    if not question.strip() or len(question) > options["max_question_chars"]:
        raise ValueError("Question is empty or exceeds the configured length limit")
    folder, manifest, records = checked_index(root, season, week)
    age = tomllib.loads((root / "config/api_nfl.toml").read_text())["max_snapshot_age_hours"]
    facts = retrieve(records, question, season, week, player, options["top_k"], now, age)
    pointer = root / "data/api_nfl/latest.json"
    if pointer.exists():
        snapshot = json.loads(pointer.read_text())["snapshot"]
        if Path(manifest.get("availability_snapshot") or "").name != snapshot:
            facts["warnings"].append("A different availability snapshot exists. Refresh LSTM forecasts and rebuild the index to use it.")
    row = next(r for r in records if r["player_id"] == facts["player_id"])
    path = root / f"data/raw/schedules/{season}.parquet"
    schedule = pl.read_parquet(path).filter(pl.col("game_id") == row["game_id"]) if path.exists() else None
    if schedule is None or schedule.height != 1 or not unstarted(schedule.row(0, named=True), now):
        facts["warnings"].append("This game has started, finished, or has no verified future kickoff. Answers describe a saved pregame snapshot, not live results.")
    settings = tomllib.loads((root / "config/outlooks.toml").read_text())
    prompt = (root / "config/prompts/rag_answer.txt").read_text(encoding="utf-8")
    result = None if retrieval_only else generate(question, facts, settings, prompt, options["max_answer_sentences"])
    report = {"question": question, "created_at_utc": now.isoformat(), "index": str(folder),
              "index_manifest_sha256": hash_file(folder / "manifest.json"), "retrieval": facts, "generation": result,
              "settings": settings if result else None, "prompt_sha256": digest(prompt), "retrieval_options": options}
    output = root / "artifacts/task5/answers" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    report["output"] = str(output)
    save(output / "answer.json", report)
    (output / "ANSWER.txt").write_text(display(report), encoding="utf-8")
    return report


def display(report):
    facts, result = report["retrieval"], report["generation"]
    lines = [f"{facts['player']} | {facts['season']} Week {facts['week']}",
             f"Forecast saved: {facts['forecast_created_at']}; feature cutoff: {facts['feature_cutoff']}", ""]
    if result is None:
        lines += [f"[{s['id']}] {s['text']}" for s in facts["evidence"]]
    elif not result["accepted"]:
        lines.append("Answer withheld: generated wording failed validation. See answer.json for review.")
    elif not result["draft"]["sentences"]:
        lines.append("The retrieved evidence is insufficient to answer this question.")
    else:
        lines += [f"{s['text']} [{s['source_id']}]" for s in result["draft"]["sentences"]]
    lines += ["", "Saved availability: " + facts["availability_note"],
              f"Retrieved: {facts['availability_retrieved_at'] or 'unknown'}; report date: {facts['injury_report_date'] or 'unknown'}."]
    lines += ["NOTE: " + warning for warning in facts["warnings"]]
    lines += ["", "Sources (retrieved evidence):"]
    lines += [f"[{s['id']}] {s['category']} - {s['source']} ({s['kind']})" for s in facts["evidence"]]
    lines += ["", f"Saved: {report['output']}"]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--season", type=int, default=load_settings()["data"]["current_season"])
    parser.add_argument("--player", help="Exact indexed player name or ID")
    parser.add_argument("--retrieve-only", action="store_true", help="Inspect evidence without calling Ollama")
    args = parser.parse_args()
    try:
        print(display(ask(args.question, args.week, args.season, args.player, args.retrieve_only)))
    except (ValueError, FileNotFoundError, requests.RequestException) as exc:
        parser.exit(1, str(exc) + "\n")
