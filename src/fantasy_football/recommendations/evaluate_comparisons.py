"""Retrospective smoke evaluation of assistant choices against higher projections."""

import argparse
from datetime import datetime, timezone
import json
import hashlib
import random
import tomllib

import numpy as np
import polars as pl

from fantasy_football.config import PROJECT_ROOT
from fantasy_football.recommendations import comparison_llm
from fantasy_football.recommendations.compare_players import calibration_file
from fantasy_football.recommendations.comparison import load_inputs, prepare_pair
from fantasy_football.recommendations.generate_outlooks import save


def evaluate(root=PROJECT_ROOT, limit=12):
    if limit < 1:
        raise ValueError("--limit must be positive")
    calibration_path = calibration_file(root)
    calibration = json.loads(calibration_path.read_text())
    source = root / "artifacts/task3/runs" / calibration["source_run"]
    data = load_inputs(source, "retrospective")
    predictions = pl.read_parquet(source / "retrospective_2025_predictions.parquet")
    predictions = predictions.filter(pl.col("lstm_prediction").is_finite() & (pl.col("lstm_prediction") >= 5))
    lookup = {(r["player_id"], r["game_id"]): i for i, r in enumerate(data["rows"].to_dicts())}
    pairs = []
    for group in predictions.sort("player_id").partition_by(["season", "week", "position"]):
        rows = group.to_dicts()
        pairs.extend(zip(rows[::2], rows[1::2]))
    random.Random(42).shuffle(pairs)
    settings = tomllib.loads((root / "config/outlooks.toml").read_text())
    prompt = (root / "config/prompts/player_comparison.txt").read_text(encoding="utf-8")
    output = root / "artifacts/task4/comparison_evaluations" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    results = []
    for a, b in pairs[:limit]:
        indices = [lookup[(r["player_id"], r["game_id"])] for r in (a, b)]
        combined = [data["rows"].row(i, named=True) | pred for i, pred in zip(indices, (a, b))]
        facts = prepare_pair(*combined, *[data["sequences"][i] for i in indices], calibration, historical=True)
        # Targets stay outside the prompt. Names are masked to reduce retrospective recall.
        for side in ("A", "B"):
            name = facts["players"][side]
            facts["players"][side] = "Player " + side
            facts["evidence"] = {k: text.replace(name, "Player " + side) for k, text in facts["evidence"].items()}
        assistant = comparison_llm.recommend(settings, prompt, facts)
        choice = assistant["draft"]["choice"] if assistant["accepted"] else "review"
        actual = {"A": a["target_ppr_points"], "B": b["target_ppr_points"]}
        baseline = facts["model_preference"]
        result = {"players": [a["player_display_name"], b["player_display_name"]], "facts": facts,
                  "assistant": assistant, "choice": choice, "actual": actual,
                  "baseline_choice": baseline,
                  "assistant_points": actual.get(choice), "baseline_points": actual.get(baseline),
                  "actual_winner": "A" if actual["A"] > actual["B"] else "B" if actual["B"] > actual["A"] else "tie"}
        results.append(result)
        save(output / "cases.json", results)
        print(f"Pair {len(results)}/{min(limit, len(pairs))}: assistant={choice}, projection={baseline}", flush=True)
    decided = [r for r in results if r["assistant_points"] is not None and r["baseline_points"] is not None]
    mean = lambda values: float(np.mean(values)) if values else None
    report = {"cases": len(results), "decided_cases": len(decided),
              "abstentions": sum(r["choice"] in {"too_close", "insufficient_evidence"} for r in results),
              "review_cases": sum(r["choice"] == "review" for r in results),
              "assistant_win_rate_on_decided": mean([r["choice"] == r["actual_winner"] for r in decided]),
              "projection_win_rate_on_same_decided": mean([r["baseline_choice"] == r["actual_winner"] for r in decided]),
              "assistant_mean_points_on_decided": mean([r["assistant_points"] for r in decided]),
              "projection_mean_points_on_same_decided": mean([r["baseline_points"] for r in decided]),
              "source_run": str(source), "calibration_file": str(calibration_path), "settings": settings,
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
              "note": "Small retrospective 2025 smoke test, not evidence of superiority. Names masked; outcomes withheld from "
                      "Qwen. Same-position, disjoint pairs with both projections >=5, deterministic sample seed 42. "
                      "No historical injury snapshots or attribution evidence in these test prompts. Live comparisons include those; "
                      "prospective evaluation is still needed. Ties count as neither player winning."}
    save(output / "evaluation.json", report)
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    evaluate(limit=args.limit)
