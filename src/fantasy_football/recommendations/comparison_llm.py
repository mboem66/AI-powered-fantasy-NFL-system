"""Let Qwen choose and prioritize evidence; Python retains control of the facts."""

import json
import requests

from fantasy_football.recommendations.local_llm import model_digest


def validate(draft, facts):
    if not isinstance(draft, dict) or set(draft) != {"choice", "primary_factor_id", "evidence_ids", "disagreement_evidence_ids"}:
        return ["Expected choice, primary_factor_id, evidence_ids, disagreement_evidence_ids"]
    errors = []
    if not isinstance(draft["choice"], str) or draft["choice"] not in {"A", "B", "too_close", "insufficient_evidence"}:
        return ["Invalid choice"]
    ids, disagreement = draft["evidence_ids"], draft["disagreement_evidence_ids"]
    if not isinstance(ids, list) or not all(isinstance(k, str) for k in ids):
        return errors + ["Expected evidence ID list"]
    if len(set(ids)) != len(ids) or not 2 <= len(ids) <= 4 or any(k not in facts["evidence"] for k in ids):
        errors.append("Select 2-4 unique supplied evidence IDs")
    if not all(any(k.startswith(side + ".") for k in ids) for side in ("A", "B")):
        errors.append("Include evidence for both players")
    scoring = {"projection", "ppr_last_game", "ppr_recent_avg", "ppr_season_avg"}
    primary = draft["primary_factor_id"]
    if not isinstance(primary, str) or primary not in facts["evidence"] or primary.split(".")[-1] in scoring:
        errors.append("Primary factor must be usage, trend, matchup, availability, or attribution evidence")
    if not isinstance(disagreement, list) or not all(isinstance(k, str) and k in ids and
                                                   not k.endswith(".projection") for k in disagreement):
        errors.append("Invalid disagreement evidence")
    differs = draft["choice"] in {"A", "B"} and facts["model_preference"] in {"A", "B"} and draft["choice"] != facts["model_preference"]
    if differs and not disagreement:
        errors.append("Explain disagreement with the projection preference")
    if not differs and disagreement:
        errors.append("Disagreement evidence is only for a contrary selection")
    return errors


def recommend(settings, prompt, facts):
    digest = model_digest(settings)
    keys = list(facts["evidence"])
    factors = [k for k in keys if k.split(".")[-1] not in {"projection", "ppr_last_game", "ppr_recent_avg", "ppr_season_avg"}]
    schema = {"type": "object", "properties": {
        "choice": {"type": "string", "enum": ["A", "B", "too_close", "insufficient_evidence"]},
        "primary_factor_id": {"type": "string", "enum": factors},
        "evidence_ids": {"type": "array", "items": {"type": "string", "enum": keys}, "minItems": 2, "maxItems": 4},
        "disagreement_evidence_ids": {"type": "array", "items": {"type": "string", "enum": keys}}},
        "required": ["choice", "primary_factor_id", "evidence_ids", "disagreement_evidence_ids"], "additionalProperties": False}
    attempts = []
    for _ in range(2):
        response = requests.post(settings["url"] + "/api/chat", json={
            "model": settings["model"], "stream": False, "think": False, "keep_alive": "10m", "format": schema,
            "options": {k: settings[k] for k in ("temperature", "seed", "num_ctx", "num_predict")},
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(facts)}]},
            timeout=settings["timeout_seconds"])
        response.raise_for_status()
        body = response.json()
        raw = body["message"]["content"]
        try:
            draft = json.loads(raw)
        except ValueError:
            draft = None
        errors = validate(draft, facts)
        if body.get("done_reason") != "stop":
            errors.append("Generation did not finish normally")
        attempts.append({"draft": draft, "errors": errors, "raw_response": raw})
        if not errors:
            break
        prompt += "\nCorrect these validation issues: " + "; ".join(errors)
    return {"accepted": not errors, "draft": draft, "attempts": attempts, "model_digest": digest,
            "model": settings["model"]}


def render(result, facts):
    if not result["accepted"]:
        return "Assistant recommendation withheld: generated evidence needs review."
    draft = result["draft"]
    names = facts["players"]
    choice = draft["choice"]
    opening = (f"Lean toward {names[choice]} for this lineup slot." if choice in names else
               "This comparison is too close for a clear recommendation." if choice == "too_close" else
               "There is insufficient evidence for a start/sit recommendation.")
    selected = list(dict.fromkeys([draft["primary_factor_id"], *draft["evidence_ids"]]))
    lines = [opening, *[facts["evidence"][k] for k in selected]]
    if draft["disagreement_evidence_ids"]:
        lines.append("This differs from the higher-projection choice; the assistant prioritized: " +
                     " ".join(facts["evidence"][k] for k in draft["disagreement_evidence_ids"]))
    return " ".join(lines)
