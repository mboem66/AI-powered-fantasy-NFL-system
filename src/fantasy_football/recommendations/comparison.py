"""Reusable comparison facts and saved-input loading for the future web app."""

import math

import numpy as np
import polars as pl

from fantasy_football.recommendations import uncertainty
from fantasy_football.recommendations.player_context import availability_sentence
from fantasy_football.modeling.sequences import SEQUENCE_FIELDS


def load_inputs(folder, prefix):
    rows = pl.read_parquet(folder / f"{prefix}_rows.parquet")
    if rows.unique(["player_id", "game_id"]).height != rows.height:
        raise ValueError("Duplicate saved player-game inputs")
    with np.load(folder / f"{prefix}_sequences.npz", allow_pickle=False) as arrays:
        lengths = arrays["lengths"].copy()
        context = arrays["context"].copy()
        if len(lengths) != rows.height or len(context) != rows.height:
            raise ValueError("Saved input dimensions differ")
        sequences = [arrays["values"][i, :length].copy() for i, length in enumerate(lengths)]
    if rows["sequence_length"].to_list() != lengths.tolist():
        raise ValueError("Saved sequence lengths differ")
    return {"rows": rows, "sequences": sequences, "context": context, "lengths": lengths,
            "history_ids": rows["history_game_ids"].to_list()}


def take(data, index):
    return {"rows": data["rows"].slice(index, 1), "sequences": [data["sequences"][index]],
            "context": data["context"][index:index+1], "lengths": data["lengths"][index:index+1],
            "history_ids": [data["history_ids"][index]]}


def resolve(rows, name):
    matches = [i for i, row in enumerate(rows) if name.casefold() in
               {str(row["player_id"]).casefold(), row["player_display_name"].casefold()}]
    if len(matches) != 1:
        raise ValueError(f"Expected one exact player name or ID for '{name}', found {len(matches)}")
    return matches[0]


def check_pair(a, b):
    if a["player_id"] == b["player_id"]:
        raise ValueError("Choose two different players")
    if (a["season"], a["week"]) != (b["season"], b["week"]):
        raise ValueError("Players must be in the same NFL season and week")
    positions = {a["position"], b["position"]}
    if not (len(positions) == 1 or positions <= {"RB", "WR", "TE"}):
        raise ValueError("Compare the same position or RB/WR/TE for a flex slot")
    for row in (a, b):
        if row["position"] not in {"QB", "RB", "WR", "TE"} or row.get("lstm_prediction") is None or not math.isfinite(row["lstm_prediction"]):
            raise ValueError("Both players need a finite LSTM prediction")


def evidence_for(side, row, sequence, historical=False, now=None, max_age_hours=24):
    name, position = row["player_display_name"], row["position"]
    evidence = {f"{side}.projection": f"{name} is projected for {row['lstm_prediction']:.1f} PPR points."}
    for key, label in (("ppr_last_game", "last-game PPR"), ("ppr_recent_avg", "recent-average PPR"),
                       ("ppr_season_avg", "season-to-date average PPR")):
        value = row.get(key)
        if value is not None and math.isfinite(value):
            evidence[f"{side}.{key}"] = f"{name}'s {label} is {value:.1f}."
    fields = {"QB": ["attempts", "rushing_yards"], "RB": ["carries", "targets"],
              "WR": ["targets", "receptions"], "TE": ["targets", "receptions"]}[position]
    recent = sequence[-3:]
    for field in fields:
        values = recent[:, SEQUENCE_FIELDS.index(field)]
        if np.isfinite(values).all():
            evidence[f"{side}.{field}"] = f"{name} averaged {values.mean():.1f} {field.replace('_', ' ')} over the last {len(recent)} appearances."
    for field in ("targets_trend", "carries_trend", "snap_share_trend"):
        value = row.get(field)
        if value is not None and math.isfinite(value):
            evidence[f"{side}.{field}"] = (f"{name}'s {field.replace('_trend', '').replace('_', ' ')} changed by {value:+.1f} "
                                          "in the last appearance versus the preceding two appearances.")
    allowed = row.get("opponent_ppr_allowed_recent_avg")
    count = row.get("opponent_history_games", 0)
    if allowed is not None and math.isfinite(allowed) and count >= 3:
        evidence[f"{side}.matchup"] = (f"{row['opponent_team']} allowed {allowed:.1f} PPR per game to opposing {position} groups "
                                      "over its last three games; this is a matchup indicator, not a guarantee.")
    else:
        evidence[f"{side}.matchup"] = f"{name} has insufficient recent opponent history for a matchup comparison."
    availability = ("Historical injury snapshots are unavailable for this comparison." if historical else
                    availability_sentence(row, now, max_age_hours))
    if availability:
        evidence[f"{side}.availability"] = name + ": " + availability
    return evidence


def prepare_pair(a, b, sequence_a, sequence_b, calibration, historical=False, now=None, max_age_hours=24):
    check_pair(a, b)
    if not len(sequence_a) or not len(sequence_b):
        raise ValueError("Both players need current-season history")
    preference = "A" if a["lstm_prediction"] > b["lstm_prediction"] else "B" if b["lstm_prediction"] > a["lstm_prediction"] else "tie"
    probabilities = uncertainty.probability(calibration,
        {"prediction": a["lstm_prediction"], "position": a["position"]},
        {"prediction": b["lstm_prediction"], "position": b["position"]})
    evidence = evidence_for("A", a, sequence_a, historical, now, max_age_hours)
    evidence.update(evidence_for("B", b, sequence_b, historical, now, max_age_hours))
    return {"players": {"A": a["player_display_name"], "B": b["player_display_name"]},
            "season": a["season"], "week": a["week"], "model_preference": preference,
            "probabilities": probabilities, "uncertainty_note": uncertainty.NOTE,
            "ranges": {side: uncertainty.interval(calibration, row["position"], row["lstm_prediction"])
                       for side, row in (("A", a), ("B", b))}, "evidence": evidence}
