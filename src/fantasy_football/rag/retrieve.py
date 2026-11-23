"""Filter by player/week first, then rank relevant passages with TF-IDF."""

from datetime import datetime, timezone
import hashlib
import json
import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer

from fantasy_football.recommendations.player_context import availability_sentence


def normalize(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def load_index(root, season, week):
    for folder in sorted((root / "artifacts/task5/indexes").glob("[0-9]*"), reverse=True):
        if not (folder / "manifest.json").exists():
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        if (manifest["season"], manifest["week"]) != (season, week):
            continue
        raw = (folder / "records.json").read_bytes()
        if manifest["schema_version"] != 1 or hashlib.sha256(raw).hexdigest() != manifest["records_sha256"]:
            raise ValueError("Index format or checksum changed; rebuild the Task 5 index")
        records = json.loads(raw)
        if any((r["season"], r["week"]) != (season, week) for r in records):
            raise ValueError("Index contains mixed seasons/weeks")
        return folder, manifest, records
    raise ValueError("No index for this season/week. Run fantasy_football.rag.build_index first")


def resolve_player(records, question, player=None):
    question = " " + normalize(question) + " "
    full = [r for r in records if " " + normalize(r["player_display_name"]) + " " in question]
    if len(full) > 1:
        raise ValueError("This version answers about one player. Use the Task 4 comparison command for two players")
    if full and " and " in question:
        remainder = question.replace(normalize(full[0]["player_display_name"]), "")
        others = [r for r in records if r["player_id"] != full[0]["player_id"] and
                  " " + normalize(r["player_display_name"]).split()[-1] + " " in remainder]
        if others:
            raise ValueError("This version answers about one player; another player name was found")
    if player:
        matches = [r for r in records if normalize(player) in (normalize(r["player_display_name"]), normalize(r["player_id"]))]
        if len(matches) != 1:
            raise ValueError("--player must be one exact indexed name or player ID")
        if full and full[0]["player_id"] != matches[0]["player_id"]:
            raise ValueError("The player in the question conflicts with --player")
        return matches[0]
    if full:
        return full[0]
    matches = []
    for row in records:
        parts = normalize(row["player_display_name"]).split()
        if parts[-1] in {"jr", "sr", "ii", "iii", "iv"}:
            parts.pop()
        if " " + parts[-1] + " " in question:
            matches.append(row)
    if len(matches) != 1:
        names = ", ".join(r["player_display_name"] for r in matches[:8])
        raise ValueError("Player missing or ambiguous. Supply --player with the full name." + (f" Matches: {names}" if names else ""))
    return matches[0]


def categories(question):
    q = normalize(question)
    if re.search(r"\b(start|sit|versus|vs|better pick)\b", q):
        raise ValueError("Start/sit comparisons belong to Task 4 for now; use recommendations.compare_players")
    if re.search(r"\b(weather|news|contract|trade rumors|bet|odds|touchdowns tonight|final score)\b", q):
        raise ValueError("This snapshot has no news, weather, betting, or live game results")
    if re.search(r"\b(did|actual|result|results|scored)\b", q) and not re.search(r"\b(last game|previous game|recent|prior)\b", q):
        raise ValueError("This index has pregame forecasts and prior statistics, not this week's actual results")
    if re.search(r"\b(injury|injured|designation|status|questionable|playing|active|healthy|availability|knee|hamstring)\b", q):
        return ["availability"]
    if re.search(r"\b(usage|target|targets|carries|receptions|attempts|snaps|volume|trend)\b", q):
        return ["usage"]
    if re.search(r"\b(matchup|opponent|defense|difficulty)\b", q):
        return ["matchup"]
    if re.search(r"\b(outlook|preview|summary)\b", q):
        return ["projection", "performance", "usage", "matchup", "outlook"]
    if re.search(r"\b(project|projected|projection|predict|predicted|forecast|expect)\b", q):
        return ["projection", "performance", "usage", "matchup"]
    if re.search(r"\b(last|recent|season|average|performance|scored)\b", q):
        return ["performance"]
    raise ValueError("Ask about an outlook, projection, prior performance, usage, matchup, or saved availability")


def retrieve(records, question, season, week, player=None, top_k=6, now=None, max_age_hours=24):
    now = now or datetime.now(timezone.utc)
    weeks = re.findall(r"\bweek\s+(\d+)\b", question, flags=re.I)
    years = re.findall(r"\b20\d{2}\b", question)
    if any(int(w) != week for w in weeks) or any(int(y) != season for y in years):
        raise ValueError("Question season/week conflicts with the explicit command options")
    if re.search(r"\b(last|next|previous)\s+(week|season)\b", question, re.I):
        raise ValueError("Use an explicit season/week for historical or future questions")
    row = resolve_player(records, question, player)
    if (row["season"], row["week"]) != (season, week):
        raise ValueError("Player belongs to a different season/week")
    selected_categories = categories(question)
    chunks = [dict(c) for c in row["chunks"] if c["category"] in selected_categories]
    if not chunks:
        raise ValueError("No saved evidence for this question")
    # Rank only inside the correct player/week. Similar names cannot cross this boundary.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
    matrix = vectorizer.fit_transform([c["category"] + " " + c["text"] for c in chunks] + [question])
    scores = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
    ranked = sorted(zip(chunks, scores), key=lambda item: (-item[1], item[0]["id"]))[:top_k]
    evidence = [c | {"score": float(score)} for c, score in ranked]
    status = availability_sentence(row["availability"], now, max_age_hours)
    status = status or "No designation was listed in the fresh matched snapshot; this does not confirm active status."
    checked = row["availability"].get("availability_checked_at")
    try:
        timestamp = datetime.fromisoformat(str(checked).replace("Z", "+00:00"))
        age = (now - timestamp).total_seconds() / 3600 if timestamp.tzinfo else None
    except ValueError:
        age = None
    warnings = []
    if age is None or age < 0 or age > max_age_hours:
        warnings.append("Availability retrieval time is missing, invalid, or older than the freshness limit; it is not current confirmation.")
    for chunk in evidence:
        if chunk["category"] == "availability":
            chunk["text"] += " " + status + " " + " ".join(warnings)
    return {"player": row["player_display_name"], "player_id": row["player_id"], "season": season, "week": week,
            "forecast_created_at": row["forecast_created_at"], "game_date": row["game_date"],
            "feature_cutoff": row["feature_cutoff"], "availability_note": status,
            "availability_retrieved_at": checked, "injury_report_date": row["availability"].get("injury_report_date"),
            "warnings": warnings, "evidence": evidence}
