"""Local Qwen answers with passage-level citations and conservative output checks."""

import json
import re
import requests

from fantasy_football.recommendations.local_llm import model_digest


def numbers(text):
    return {float(n) for n in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text)}


def validate(draft, retrieval, max_sentences=4):
    if not isinstance(draft, dict) or set(draft) != {"sentences"} or not isinstance(draft["sentences"], list):
        return ["Expected a sentences array"]
    if len(draft["sentences"]) > max_sentences:
        return ["Answer is too long"]
    sources = {e["id"]: e for e in retrieval["evidence"]}
    errors = []
    for sentence in draft["sentences"]:
        if not isinstance(sentence, dict) or set(sentence) != {"text", "source_id"}:
            errors.append("Each sentence needs text and source_id")
            continue
        text, source = sentence["text"], sentence["source_id"]
        if not isinstance(text, str) or not text.strip() or len(text) > 650 or not isinstance(source, str) or source not in sources:
            errors.append("Empty/long sentence or unknown citation")
            continue
        allowed = sources[source]["text"].replace(retrieval["player"], "Player")
        checked = text.replace(retrieval["player"], "Player")
        if not numbers(checked) <= numbers(allowed):
            errors.append("Sentence contains a number not in its cited source")
        if re.search(r"guarantee|definitely|confirmed (active|healthy)|will (play|score)|should (start|sit)|because|due to|after averaging", text, re.I):
            errors.append("Unsupported certainty, recommendation, or causal claim")
        if sources[source]["category"] == "availability" and not re.search(r"saved|snapshot|report|listed|retrieved", text, re.I):
            errors.append("Availability must be described as a saved report")
    return errors


def generate(question, retrieval, settings, prompt, max_sentences=4):
    model = model_digest(settings)
    schema = {"type": "object", "properties": {"sentences": {"type": "array", "maxItems": max_sentences,
        "items": {"type": "object", "properties": {"text": {"type": "string"},
            "source_id": {"type": "string", "enum": [e["id"] for e in retrieval["evidence"]]}},
            "required": ["text", "source_id"], "additionalProperties": False}}},
        "required": ["sentences"], "additionalProperties": False}
    attempts = []
    for _ in range(2):
        response = requests.post(settings["url"] + "/api/chat", json={
            "model": settings["model"], "stream": False, "think": False, "keep_alive": "10m", "format": schema,
            "options": {k: settings[k] for k in ("temperature", "seed", "num_ctx", "num_predict")},
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": json.dumps({"question": question, "context": retrieval})}]},
            timeout=settings["timeout_seconds"])
        response.raise_for_status()
        body = response.json()
        raw = body["message"]["content"]
        try:
            draft = json.loads(raw)
        except ValueError:
            draft = None
        errors = validate(draft, retrieval, max_sentences)
        if body.get("done_reason") != "stop":
            errors.append("Generation was truncated or did not finish normally")
        attempts.append({"draft": draft, "raw_response": raw, "errors": errors})
        if not errors:
            break
        prompt += "\nCorrect these checks: " + "; ".join(errors)
    return {"accepted": not errors, "draft": draft, "attempts": attempts, "model_digest": model,
            "validation_note": "Checks citations, numbers and selected unsafe wording; not a proof of semantic correctness."}
