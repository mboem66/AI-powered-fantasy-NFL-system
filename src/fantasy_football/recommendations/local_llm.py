"""Small Ollama client and conservative checks on generated wording."""

import json
import re
from urllib.parse import urlparse
import requests


def model_digest(settings):
    if urlparse(settings["url"]).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Outlooks are configured for a local Ollama server only")
    response = requests.get(settings["url"] + "/api/tags", timeout=15)
    response.raise_for_status()
    for model in response.json()["models"]:
        if model["name"] == settings["model"]:
            return model["digest"]
    raise ValueError(f"Install {settings['model']} in Ollama first")


def generate(settings, prompt, facts):
    response = requests.post(settings["url"] + "/api/chat", json={
        "model": settings["model"], "stream": False, "think": False, "keep_alive": "10m",
        "format": {"type": "object", "properties": {k: {"type": "string"}
                   for k in ("projection", "performance")},
                   "required": ["projection", "performance"], "additionalProperties": False},
        "options": {k: settings[k] for k in ("temperature", "seed", "num_ctx", "num_predict")},
        "messages": [{"role": "system", "content": prompt},
                     {"role": "user", "content": json.dumps(facts)}]},
        timeout=settings["timeout_seconds"])
    response.raise_for_status()
    body = response.json()
    raw = body["message"]["content"]
    try:
        draft = json.loads(raw)
    except ValueError:
        draft = {}
    errors = validate(draft, facts)
    if body.get("done_reason") != "stop":
        errors.append("Generation did not finish normally")
    return {"draft": draft, "raw_response": raw, "errors": errors}


def validate(draft, facts):
    errors = []
    if not isinstance(draft, dict) or set(draft) != {"projection", "performance"}:
        return ["Expected projection and performance sentences"]
    required = {"projection": [facts["projected_ppr"], facts["recent_ppr_average"]],
                "performance": [facts["last_game_ppr"], *facts["usage_averages"].values()]}
    for key, values in required.items():
        text = draft[key]
        if not isinstance(text, str):
            errors.append(f"{key}: expected text")
            continue
        # Remove the name first: names can contain initials or numeric suffixes.
        checked = text.replace(facts["player"], "Player")
        numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", checked)
        allowed = values + ([facts["usage_window_appearances"]] if key == "performance" else [])
        if any(float(n) not in [float(v) for v in allowed] for n in numbers):
            errors.append(f"{key}: unsupported number")
        if any(float(v) not in [float(n) for n in numbers] for v in values):
            errors.append(f"{key}: missing required number")
        if len(re.split(r"(?<=[.!?])\s+", checked.strip())) != 1:
            errors.append(f"{key}: expected one sentence")
        if re.search(r"injur|questionable|healthy|guarantee|will score|upside|confiden|must.start|because|due to", text, re.I):
            errors.append(f"{key}: unsupported interpretation")
    if isinstance(draft.get("projection"), str):
        if facts["player"] not in draft["projection"] or facts["comparison"] not in draft["projection"].lower():
            errors.append("Projection name or comparison missing")
        projection = draft["projection"].replace(facts["player"], "Player")
        ordered = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", projection)
        if [float(n) for n in ordered] != [float(facts["projected_ppr"]), float(facts["recent_ppr_average"])]:
            errors.append("Projection and recent-average values must appear in that order")
    if isinstance(draft.get("performance"), str):
        performance = draft["performance"].lower()
        if f"last {facts['usage_window_appearances']} appearances" not in performance or "averaged" not in performance or "season" in performance:
            errors.append("Usage window or averaging description is incorrect")
        for label, value in facts["usage_averages"].items():
            if f"{value} {label}" not in performance:
                errors.append(f"Usage label missing: {label}")
    return errors
