"""Manually download complete roster/injury snapshots for all NFL teams."""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import tomllib
import unicodedata

import polars as pl

from fantasy_football.config import PROJECT_ROOT, load_settings
from fantasy_football.data.api_nfl import Client
from fantasy_football.data.cleaning import TEAM_ALIASES

SCHEMA = {"api_player_id": pl.Int64, "player_id": pl.String, "name": pl.String,
          "position": pl.String, "team": pl.String, "roster_group": pl.String,
          "mapping_status": pl.String, "injury_status": pl.String,
          "injury_date": pl.String, "injury_description": pl.String,
          "fetched_at_utc": pl.String}


def normalized(value):
    text = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def identity_index(root, season):
    # Exact full-name and position matches only; no initials or fuzzy matching.
    index = defaultdict(set)
    roster = pl.read_parquet(root / f"data/raw/rosters/{season}.parquet")
    players = pl.read_parquet(root / "data/raw/players/latest.parquet")
    for frame, name in [(roster, "full_name"), (players, "display_name")]:
        for row in frame.to_dicts():
            if row.get("gsis_id") and row.get(name) and row.get("position"):
                index[(normalized(row[name]), row["position"])].add(row["gsis_id"])
    return index


def normalize_players(players, injuries, team, index, fetched):
    injury_index = defaultdict(list)
    for injury in injuries:
        if not isinstance(injury.get("player", {}).get("id"), int) or not isinstance(injury.get("status"), str):
            raise ValueError("Unexpected injury schema; refusing to mark players as unlisted")
        injury_index[injury["player"]["id"]].append(injury)
    rows, seen = [], set()
    # Preserve injured players who are absent from the player list for review.
    entries = list(players)
    roster_ids = {p.get("id") for p in players}
    entries += [r[0]["player"] for pid, r in injury_index.items() if pid not in roster_ids]
    for player in entries:
        pid, name = player.get("id"), player.get("name")
        if not isinstance(pid, int) or not name or pid in seen:
            raise ValueError("Missing or duplicate API player identity")
        seen.add(pid)
        position = player.get("position")
        matches = index.get((normalized(name), position), set())
        reports = injury_index.get(pid, [])
        # Conflicting reports require review; do not choose a status silently.
        distinct = {(r.get("status"), r.get("date"), r.get("description")) for r in reports}
        injury = reports[0] if len(distinct) == 1 else {}
        rows.append({"api_player_id": pid, "player_id": next(iter(matches)) if len(matches) == 1 else None,
                     "name": name, "position": position, "team": team, "roster_group": player.get("group"),
                     "mapping_status": "matched" if len(matches) == 1 else "needs_review",
                     "injury_status": "conflicting_reports" if len(distinct) > 1 else injury.get("status", "not_listed"),
                     "injury_date": injury.get("date"), "injury_description": injury.get("description"),
                     "fetched_at_utc": fetched})
    return rows


def run(root=PROJECT_ROOT, settings=None, client=None, now=None):
    settings = settings or load_settings()
    season = settings["data"]["current_season"]
    with (root / "config/api_nfl.toml").open("rb") as stream:
        options = tomllib.load(stream)
    index = identity_index(root, season)
    team_names = {normalized(r["team_name"]): TEAM_ALIASES.get(r["team_abbr"], r["team_abbr"])
                  for r in pl.read_parquet(root / "data/raw/teams/latest.parquet").to_dicts()}
    owns_client = client is None
    client = client or Client(root)
    started = now or datetime.now(timezone.utc)
    parent = root / "data/api_nfl"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / started.strftime("%Y%m%dT%H%M%S%fZ")
    try:
        with TemporaryDirectory(dir=parent, prefix="pending_") as temporary:
            folder = Path(temporary) / "result"
            folder.mkdir()
            leagues = client.get("/leagues", id=options["league_id"], season=season)
            coverage = [s.get("coverage", {}) for league in leagues
                        if league.get("league", {}).get("id") == options["league_id"]
                        for s in league.get("seasons", []) if s.get("year") == season]
            if len(coverage) != 1 or not coverage[0].get("injuries") or not coverage[0].get("players"):
                raise ValueError(f"API-NFL does not confirm player and injury coverage for {season}; no snapshot published.")
            teams = client.get("/teams", league=options["league_id"], season=season)
            if len(teams) != 32 or len({t.get("id") for t in teams}) != 32:
                raise ValueError("Expected 32 distinct NFL teams")
            raw, rows, mapped_teams = {"leagues": leagues, "teams": teams, "team_data": []}, [], set()
            for team in teams:
                abbreviation = team_names.get(normalized(team.get("name")))
                if not abbreviation or abbreviation in mapped_teams:
                    raise ValueError(f"Unmatched or duplicate team: {team.get('name')}")
                mapped_teams.add(abbreviation)
                players = client.get("/players", team=team["id"], season=season)
                injuries = client.get("/injuries", team=team["id"])
                if not players:
                    raise ValueError(f"Empty player list for {abbreviation}; prior snapshot preserved")
                if any(r.get("team", {}).get("id") != team["id"] for r in injuries):
                    raise ValueError("Injury response contains a different team")
                fetched = (now or datetime.now(timezone.utc)).isoformat()
                rows.extend(normalize_players(players, injuries, abbreviation, index, fetched))
                raw["team_data"].append({"team": abbreviation, "fetched_at_utc": fetched,
                                         "players": players, "injuries": injuries})
                print(f"API-NFL: {abbreviation}: {len(players)} players, {len(injuries)} injury reports", flush=True)
            frame = pl.DataFrame(rows, schema=SCHEMA, strict=True)
            # Trades or duplicated team listings must be reviewed, not joined twice.
            frame = frame.with_columns(pl.when(pl.col("api_player_id").is_duplicated() |
                (pl.col("player_id").is_not_null() & pl.col("player_id").is_duplicated()))
                .then(pl.lit("needs_review")).otherwise(pl.col("mapping_status")).alias("mapping_status"))
            frame.write_parquet(folder / "players.parquet", compression="zstd")
            frame.filter(pl.col("position").is_in(settings["data"]["positions"]) |
                         (pl.col("injury_status") != "not_listed")).write_csv(folder / "roster_review.csv")
            (folder / "raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
            report = {"season": season, "started_at_utc": started.isoformat(), "teams": 32,
                      "completed_at_utc": (now or datetime.now(timezone.utc)).isoformat(),
                      "rows": frame.height, "requests": client.calls,
                      "unmatched_rows": frame.filter(pl.col("mapping_status") != "matched").height,
                      "note": "Current reports only. Not listed is not confirmed active. No dedicated inactive feed."}
            (folder / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            folder.rename(output)
            # Publish only after every team succeeded. Failed attempts leave latest untouched.
            pointer = parent / f"latest_{output.name}.tmp"
            pointer.write_text(json.dumps({"snapshot": output.name}), encoding="utf-8")
            pointer.replace(parent / "latest.json")
    finally:
        if owns_client:
            client.close()
    print(f"Saved availability snapshot: {output}")
    return output


def main():
    try:
        run()
    except (ValueError, FileNotFoundError) as error:
        print(f"Availability refresh stopped: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
