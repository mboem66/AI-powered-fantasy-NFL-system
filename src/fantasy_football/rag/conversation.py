"""Resolve conversational references without asking the language model to guess IDs."""

import re
import json
from pathlib import Path

from fantasy_football.config import PROJECT_ROOT
from fantasy_football.rag.ask import ask, checked_index, display
from fantasy_football.rag.retrieve import normalize
from fantasy_football.recommendations.compare_players import compare_players, display as display_comparison


def scope(question, state):
    weeks = {int(n) for n in re.findall(r"\bweek\s+(\d+)\b", question, re.I)}
    seasons = {int(n) for n in re.findall(r"\b20\d{2}\b", question)}
    if len(weeks) > 1 or len(seasons) > 1:
        raise ValueError("Choose one season and week per question")
    if re.search(r"\b(last|next|previous)\s+(week|season)\b", question, re.I):
        raise ValueError("Please give the explicit week/season number")
    week, season = next(iter(weeks), state["week"]), next(iter(seasons), state["season"])
    if not 1 <= week <= 18:
        raise ValueError("Choose a regular-season week from 1 through 18")
    return season, week


def mentioned_players(question, records):
    """Full names first; accept surnames only when they identify exactly one player."""
    remaining = " " + normalize(question) + " "
    matches = []
    for row in sorted(records, key=lambda r: len(r["player_display_name"]), reverse=True):
        for alias in (normalize(row["player_display_name"]), normalize(row["player_id"])):
            token = " " + alias + " "
            if token in remaining:
                matches.append((normalize(question).find(alias), row))
                remaining = remaining.replace(token, " ")
                break
    surnames = {}
    for row in records:
        parts = normalize(row["player_display_name"]).split()
        if parts[-1] in {"jr", "sr", "ii", "iii", "iv"}:
            parts.pop()
        surnames.setdefault(parts[-1], []).append(row)
    for surname, candidates in surnames.items():
        if " " + surname + " " not in remaining:
            continue
        if len(candidates) > 1:
            raise ValueError("Which player? Please use a full name: " + ", ".join(r["player_display_name"] for r in candidates))
        matches.append((normalize(question).find(surname), candidates[0]))
    return list({row["player_id"]: row for _, row in sorted(matches, key=lambda item: item[0])}.values())


def plan(question, state, records):
    """Store only resolved identifiers/topic, never model-generated claims as evidence."""
    q = normalize(question)
    players = mentioned_players(question, records)
    comparison = bool(re.search(r"\b(start|sit|versus|vs|better pick|who should)\b", q))
    stat_comparison = re.search(r"\b(average|season|recent|last game)\b", q)
    comparison |= bool(re.search(r"\bcompare\b", q) and (len(players) > 1 or
                       (not stat_comparison and re.search(r"\b(him|them|those)\b", q))))
    by_id = {r["player_id"]: r for r in records}
    previous = [by_id[p] for p in state.get("players", []) if p in by_id]
    pronoun = bool(re.search(r"\b(he|his|him)\b", q))
    if comparison:
        if len(players) == 1 and pronoun and len(previous) == 1 and previous[0]["player_id"] != players[0]["player_id"]:
            players = previous + players
        elif not players and len(previous) == 2 and re.search(r"\b(them|those|both|who)\b", q):
            players = previous
        if len(players) != 2:
            raise ValueError("Name two players to compare; 'him' requires one previously selected player")
        return {"route": "comparison", "players": [r["player_id"] for r in players], "topic": "outlook",
                "question": question}
    if len(players) > 1 or (not players and len(previous) > 1):
        raise ValueError("Which player do you mean? Name one player for this follow-up")
    if not players:
        # Do not attach an unknown named player to the previous answer silently.
        elliptical = re.fullmatch(r"(?:what about |and )?(?:the )?(?:matchup|usage|injury|projection|season average|recent usage)", q)
        if len(previous) != 1 or not (pronoun or elliptical):
            raise ValueError("Please name a player; use his/him only after selecting one player")
        players = previous
    topics = [("availability", r"injury|status|questionable|healthy|availability|designation"),
              ("usage", r"usage|targets?|carries|receptions|attempts|snaps|volume|trend"),
              ("matchup", r"matchup|opponent|defense|difficulty"),
              ("projection", r"project\w*|predict\w*|forecast"),
              ("performance", r"last game|recent|season average|performance|scored"),
              ("outlook", r"outlook|preview|summary")]
    topic = next((name for name, pattern in topics if re.search(r"\b(?:" + pattern + r")\b", q)), None)
    rewritten = question
    if topic is None:
        # Only a bare name or a short 'what about NAME' inherits the topic.
        remainder = q
        for alias in (normalize(players[0]["player_display_name"]), normalize(players[0]["player_id"])):
            remainder = remainder.replace(alias, "").strip()
        surname = normalize(players[0]["player_display_name"]).split()[-1]
        if remainder in {surname, "what about " + surname, "and " + surname}:
            remainder = ""
        if remainder not in {"", "what about", "and"}:
            raise ValueError("Ask about this player's outlook, projection, usage, matchup, performance, or availability")
        topic = state.get("topic", "outlook")
        rewritten = f"What is {players[0]['player_display_name']}'s {topic}?"
    return {"route": "player", "players": [players[0]["player_id"]], "topic": topic, "question": rewritten}


def respond(question, state, root=PROJECT_ROOT, retrieval_only=False):
    if not question.strip() or len(question) > 1000:
        raise ValueError("Use a question between 1 and 1000 characters")
    if re.search(r"\b(weather|news|betting|odds|half ppr|non ppr|standard scoring|superflex)\b", normalize(question)):
        raise ValueError("This chat uses saved full-PPR data only; external news/weather and other scoring formats are not supported")
    season, week = scope(question, state)
    _, manifest, records = checked_index(root, season, week)
    planned = plan(question, state, records)
    if planned["route"] == "comparison":
        report = compare_players(*planned["players"], week, season, root=root,
                                 run=Path(manifest["prediction_run"]).name, use_llm=not retrieval_only)
        text = display_comparison(report)
        pointer = root / "data/api_nfl/latest.json"
        if pointer.exists() and Path(manifest.get("availability_snapshot") or "").name != json.loads(pointer.read_text())["snapshot"]:
            text += "\nNOTE: A different availability snapshot exists. Refresh LSTM forecasts and rebuild the index to use it."
    else:
        report = ask(planned["question"], week, season, planned["players"][0], retrieval_only, root=root)
        text = display(report)
    memory = {"season": season, "week": week, "players": planned["players"], "topic": planned["topic"]}
    return {"text": text, "state": memory, "route": planned["route"], "report": report["output"],
            "resolved_question": planned["question"]}
