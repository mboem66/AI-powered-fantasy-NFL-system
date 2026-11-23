"""Small terminal chat with local session history and explicit player/week memory."""

import argparse
from datetime import datetime, timezone
import json
import re

import requests

from fantasy_football.config import PROJECT_ROOT, load_settings
from fantasy_football.rag.conversation import respond
from fantasy_football.recommendations.generate_outlooks import save


def load_session(root, session_id=None, season=None, week=None):
    base = root / "artifacts/task5/chats"
    base.mkdir(parents=True, exist_ok=True)
    if session_id:
        if not re.fullmatch(r"\d{8}T\d{12}Z", session_id):
            raise ValueError("--session expects the printed session ID, not a path")
        path = base / f"{session_id}.json"
        session = json.loads(path.read_text(encoding="utf-8"))
        if session.get("schema_version") != 1:
            raise ValueError("Unsupported chat session format")
        if (week is not None and week != session["state"]["week"]) or (season is not None and season != session["state"]["season"]):
            raise ValueError("Resume without conflicting scope flags; change week explicitly in a question")
        return path, session
    if week is None or not 1 <= week <= 18:
        raise ValueError("Start a chat with --week 1 through 18")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    state = {"season": season or load_settings()["data"]["current_season"], "week": week, "players": [], "topic": "outlook"}
    return base / f"{stamp}.json", {"schema_version": 1, "state": state, "turns": []}


def turn(question, session, root=PROJECT_ROOT, retrieval_only=False):
    """Failed/ambiguous turns are recorded but do not replace resolved memory."""
    entry = {"question": question, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        result = respond(question, session["state"], root, retrieval_only)
        session["state"] = result["state"]
        entry.update(result)
    except (ValueError, FileNotFoundError, requests.RequestException) as exc:
        entry.update(error=str(exc), text="Please check: " + str(exc))
    session["turns"].append(entry)
    return entry["text"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int)
    parser.add_argument("--season", type=int)
    parser.add_argument("--session", help="Resume a printed session ID")
    parser.add_argument("--retrieve-only", action="store_true", help="No Ollama: evidence or statistical comparison only")
    args = parser.parse_args()
    try:
        path, session = load_session(PROJECT_ROOT, args.session, args.season, args.week)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, str(exc) + "\n")
    save(path, session)
    print(f"Session: {path.stem}\nSeason {session['state']['season']}, Week {session['state']['week']}.")
    print("Ask about a player, then follow up. Commands: /context, /reset, /exit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == "/exit":
            break
        if question == "/context":
            print(json.dumps(session["state"], indent=2))
            continue
        if question == "/reset":
            session["state"].update(players=[], topic="outlook")
            session["turns"].append({"command": "/reset"})
            save(path, session)
            print("Player context cleared; season and week retained.")
            continue
        if question:
            print(turn(question, session, retrieval_only=args.retrieve_only))
            save(path, session)
    print(f"Session saved: {path}")


if __name__ == "__main__":
    main()
