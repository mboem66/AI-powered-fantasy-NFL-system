"""Twenty repeatable retrieval checks, with optional local Qwen answer review."""

import argparse
from datetime import datetime, timezone
import json

from fantasy_football.config import PROJECT_ROOT, load_settings
from fantasy_football.rag.ask import ask
from fantasy_football.rag.retrieve import load_index, retrieve
from fantasy_football.recommendations.generate_outlooks import save


def evaluate(root=PROJECT_ROOT, season=2026, week=11, with_llm=False):
    folder, _, records = load_index(root, season, week)
    preferred = next((r for r in records if r["player_display_name"] == "CeeDee Lamb"), records[0])
    qb = next(r for r in records if r["position"] == "QB")
    injury = next((r for r in records if r["availability"].get("availability_status") not in {None, "unknown", "not_listed"}), preferred)
    name = preferred["player_display_name"]
    cases = [(f"What is {name}'s outlook?", "projection"),
             (f"What is {name}'s projected PPR?", "projection"),
             (f"How has {name}'s target usage changed?", "usage"),
             (f"What did {name} score last game?", "performance"),
             (f"What is {name}'s season average?", "performance"),
             (f"How difficult is {name}'s matchup?", "matchup"),
             (f"What is {name}'s saved injury status?", "availability"),
             (f"Is {name} confirmed healthy?", "availability"),
             (f"What is {qb['player_display_name']}'s projection?", "projection"),
             (f"What is {qb['player_display_name']}'s recent passing usage?", "usage"),
             (f"What is {injury['player_display_name']}'s injury designation?", "availability"),
             (f"How does {name}'s projection compare to his average?", "projection"),
             (f"What is {name}'s week {week + 1} outlook?", None),
             (f"What is {name}'s {season - 1} outlook?", None),
             ("What is Imaginary Example Person's outlook?", None),
             (f"Should I start {name}?", None),
             (f"What is {name}'s latest news?", None),
             (f"What is {name}'s weather?", None),
             (f"What was {name}'s outlook last week?", None),
             (f"Give {name} and {qb['player_display_name']} outlooks", None)]
    output = root / "artifacts/task5/evaluations" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    results = []
    for question, expected in cases:
        row = {"question": question, "expected_category": expected}
        try:
            evidence = retrieve(records, question, season, week)
            row["passed"] = expected in {e["category"] for e in evidence["evidence"]}
            if with_llm and expected:
                report = ask(question, week, season, root=root)
                row.update(answer_report=report["output"], accepted=report["generation"]["accepted"],
                           sentences=report["generation"]["draft"])
        except ValueError as exc:
            row.update(passed=expected is None, error=str(exc))
        results.append(row)
        save(output / "cases.json", results)
        print(f"{len(results)}/20: {'PASS' if row['passed'] else 'FAIL'} - {question}", flush=True)
    summary = {"index": str(folder), "cases": len(results), "retrieval_passed": sum(r["passed"] for r in results),
               "generated": sum("accepted" in r for r in results), "answers_accepted": sum(r.get("accepted", False) for r in results),
               "note": "Development smoke checks, not an independent benchmark. Review saved answers against evidence; "
                       "citation/number validation does not establish complete factual accuracy."}
    save(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--season", type=int, default=load_settings()["data"]["current_season"])
    parser.add_argument("--with-llm", action="store_true")
    args = parser.parse_args()
    evaluate(season=args.season, week=args.week, with_llm=args.with_llm)
