"""Build a small, versioned player-week knowledge snapshot from saved forecasts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import polars as pl

from fantasy_football.config import PROJECT_ROOT, load_settings
from fantasy_football.data.storage import pending_run, publish_run
from fantasy_football.recommendations.comparison import load_inputs, evidence_for


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_run(root, season, week, run=None):
    if run and Path(run).name != run:
        raise ValueError("--run expects a saved weekly folder name")
    base = root / "artifacts/task3/weekly"
    folders = [base / run] if run else sorted(base.glob("[0-9]*"), reverse=True)
    for folder in folders:
        if not all((folder / f).is_file() for f in ("summary.json", "predictions.parquet", "upcoming_rows.parquet", "upcoming_sequences.npz")):
            continue
        frame = pl.read_parquet(folder / "predictions.parquet").filter(
            (pl.col("season") == season) & (pl.col("week") == week))
        if frame.height:
            return folder, frame
    raise ValueError("No completed LSTM forecast found for this season/week")


def outlooks_for(root, folder):
    """Only accepted outlooks from this exact forecast run can be attached."""
    found = {}
    for path in sorted((root / "artifacts/task4/outlooks").glob("[0-9]*/manifest.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if Path(report["prediction_run"]).resolve() != folder.resolve():
            continue
        for result in report["results"]:
            if result.get("errors") or not result.get("outlook"):
                continue
            # Rebuild availability cautions when asking, rather than reusing old wording.
            draft = result.get("draft", {})
            text = " ".join(draft.get(k, "") for k in ("projection", "performance")).strip()
            if text:
                found[(result["player_id"], result["game_id"])] = (text, path)
    return found


def build(root=PROJECT_ROOT, season=2026, week=11, run=None, now=None):
    if not 1 <= week <= 18:
        raise ValueError("Regular-season week must be between 1 and 18")
    now = now or datetime.now(timezone.utc)
    folder, predictions = source_run(root, season, week, run)
    summary = json.loads((folder / "summary.json").read_text())
    created = datetime.fromisoformat(summary["created_at_utc"])
    if created.tzinfo is None or created > now:
        raise ValueError("Forecast timestamp is missing timezone or is in the future")
    if predictions.unique(["player_id", "game_id"]).height != predictions.height:
        raise ValueError("Duplicate saved predictions")
    data = load_inputs(folder, "upcoming")
    rows = data["rows"].to_dicts()
    lookup = {(r["player_id"], r["game_id"]): i for i, r in enumerate(rows)}
    outlooks = outlooks_for(root, folder)
    records, excluded = [], []
    sources = {folder / f for f in ("summary.json", "predictions.parquet", "upcoming_rows.parquet", "upcoming_sequences.npz")}
    availability_fields = ("availability_status", "injury_description", "injury_report_date",
                           "availability_checked_at", "availability_decision")
    for prediction in predictions.to_dicts():
        if prediction["prediction_method"] != "lstm" or prediction["sequence_length"] < 1:
            excluded.append(prediction["player_id"])
            continue
        index = lookup[(prediction["player_id"], prediction["game_id"])]
        row = rows[index]
        for key in ("player_id", "game_id", "season", "week", "position", "sequence_length"):
            if row[key] != prediction[key]:
                raise ValueError(f"Forecast/input mismatch: {key}")
        row = row | prediction
        evidence = evidence_for("player", row, data["sequences"][index], now=now)
        groups = {"projection": [], "performance": [], "usage": [], "matchup": []}
        for key, text in evidence.items():
            field = key.split(".")[-1]
            if field == "availability":
                continue
            group = "projection" if field == "projection" else "performance" if field.startswith("ppr_") else "matchup" if field == "matchup" else "usage"
            groups[group].append(text)
        # This raw saved status is deliberately separate from generated outlook prose.
        availability = {key: row.get(key) for key in availability_fields}
        groups["availability"] = [f"Saved API status/injury label: {availability['availability_status'] or 'unknown'}. "
            f"Injury description: {availability['injury_description'] or 'none supplied'}. "
            f"Report date: {availability['injury_report_date'] or 'unknown'}. "
            f"Retrieved: {availability['availability_checked_at'] or 'unknown'}. "
            "This saved report does not confirm current game-day participation."]
        attached = outlooks.get((row["player_id"], row["game_id"]))
        if attached:
            groups["outlook"] = [attached[0]]
            sources.add(attached[1])
        chunks = []
        for category, texts in groups.items():
            if not texts:
                continue
            source = attached[1] if category == "outlook" else folder / ("predictions.parquet" if category == "projection" else "upcoming_rows.parquet")
            identity = f"{folder.name}:{row['player_id']}:{row['game_id']}:{category}"
            chunks.append({"id": hashlib.sha256(identity.encode()).hexdigest()[:16], "category": category,
                           "text": " ".join(texts), "source": str(source.relative_to(root)),
                           "kind": "generated_summary" if category == "outlook" else "saved_data"})
        records.append({key: row[key] for key in ("player_id", "player_display_name", "position", "season", "week", "game_id", "game_date", "team", "opponent_team")} |
                       {"forecast_created_at": summary["created_at_utc"], "feature_cutoff": summary["feature_cutoff_exclusive"],
                        "availability": availability, "chunks": chunks})
    if not records:
        raise ValueError("No eligible LSTM players to index")
    parent = root / "artifacts/task5/indexes"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / now.strftime("%Y%m%dT%H%M%S%fZ")
    with pending_run(parent) as temporary:
        staging = temporary / "result"
        staging.mkdir()
        payload = json.dumps(records, indent=2, default=str)
        (staging / "records.json").write_text(payload, encoding="utf-8")
        manifest = {"schema_version": 1, "created_at_utc": now.isoformat(), "season": season, "week": week,
                    "prediction_run": str(folder), "availability_snapshot": summary.get("availability", {}).get("snapshot"),
                    "players": len(records), "chunks": sum(len(r["chunks"]) for r in records), "excluded_non_lstm": excluded,
                    "records_sha256": hash_file(staging / "records.json"),
                    "sources": {str(p.relative_to(root)): hash_file(p) for p in sorted(sources)},
                    "note": "One saved forecast run; no current outcomes, external news, or live injury lookup."}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        publish_run(staging, output)
    print(f"Indexed {len(records)} players for {season} Week {week}. Saved: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", required=True, type=int)
    parser.add_argument("--season", type=int, default=load_settings()["data"]["current_season"])
    parser.add_argument("--run")
    args = parser.parse_args()
    build(season=args.season, week=args.week, run=args.run)
