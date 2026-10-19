"""Build season-to-date sequences without inserting games or future outcomes."""

from bisect import bisect_left

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fantasy_football.data.cleaning import require_unique
from fantasy_football.features.recent import make_index
from fantasy_football.modeling.prepare import POSITIONS, TARGET

GAME_FIELDS = [
    "calculated_ppr_points", "targets", "receptions", "carries", "attempts",
    "passing_yards", "rushing_yards", "receiving_yards", "passing_tds",
    "rushing_tds", "receiving_tds", "offense_pct",
]
SEQUENCE_FIELDS = GAME_FIELDS + ["days_since_previous_appearance"]
CONTEXT_FIELDS = ["opponent_ppr_allowed_recent_avg", "is_home", "is_neutral_site",
                  "opponent_history_games", "days_since_last_appearance", "sequence_length"]


def make_sequences(rows, games, cutoff=None):
    """One sequence per target row; January games retain their NFL season."""
    require_unique(games, ["player_id", "game_id"], "Sequence history")
    require_unique(rows, ["player_id", "game_id"], "Sequence targets")
    missing = set(GAME_FIELDS) - set(games.columns)
    if missing:
        raise ValueError(f"Missing sequence fields: {sorted(missing)}")
    index = make_index(games.to_dicts(), lambda r: (r["player_id"], r["season"]))
    sequences, context, history_ids = [], [], []
    for row in rows.to_dicts():
        history, dates = index.get((row["player_id"], row["season"]), ([], []))
        end_date = min(row["game_date"], cutoff) if cutoff else row["game_date"]
        prior = history[:bisect_left(dates, end_date)]
        values = []
        for i, game in enumerate(prior):
            gap = (game["game_date"] - prior[i-1]["game_date"]).days if i else 0
            values.append([game[field] for field in GAME_FIELDS] + [gap])
        sequences.append(np.asarray(values, dtype=float).reshape(-1, len(SEQUENCE_FIELDS)))
        days_since_last = (row["game_date"] - prior[-1]["game_date"]).days if prior else None
        context.append([row[field] for field in CONTEXT_FIELDS[:4]] + [days_since_last, len(prior)])
        history_ids.append([game["game_id"] for game in prior])
    return {"rows": rows, "sequences": sequences,
            "context": np.asarray(context, dtype=float).reshape(-1, len(CONTEXT_FIELDS)),
            "lengths": np.asarray([len(s) for s in sequences], dtype=np.int64),
            "history_ids": history_ids}


def clean(values):
    return np.where(np.isfinite(values), values, np.nan)


def fit_preprocessing(data):
    """Learn scaling/imputation only from training examples with a sequence."""
    usable = data["lengths"] > 0
    if not usable.any():
        raise ValueError("No training examples have current-season history")
    def scaler():
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    sequence = scaler().fit(clean(np.concatenate([s for s in data["sequences"] if len(s)])))
    context = scaler().fit(clean(data["context"][usable]))
    targets = data["rows"][TARGET].to_numpy()[usable]
    return {"sequence": sequence, "context": context,
            "target_mean": float(targets.mean()), "target_scale": float(targets.std()) or 1.0}


def transform(data, preprocessing):
    # Padding is added after scaling and never included in fitted statistics.
    # Transform all real timesteps together to avoid thousands of tiny sklearn calls.
    if data["lengths"].sum():
        values = preprocessing["sequence"].transform(clean(np.concatenate(data["sequences"]))).astype(np.float32)
        sequences = np.split(values, np.cumsum(data["lengths"])[:-1])
    else:
        sequences = [np.empty((0, len(SEQUENCE_FIELDS)), dtype=np.float32) for _ in data["sequences"]]
    context = preprocessing["context"].transform(clean(data["context"]))
    positions = np.eye(len(POSITIONS))[[POSITIONS.index(p) for p in data["rows"]["position"]]]
    return sequences, np.column_stack([context, positions]).astype(np.float32)


def save_sequences(data, folder, name):
    """Save inspectable row IDs and raw arrays; lengths identify real timesteps."""
    longest = max(data["lengths"], default=0)
    padded = np.full((len(data["sequences"]), longest, len(SEQUENCE_FIELDS)), np.nan, dtype=np.float32)
    for i, sequence in enumerate(data["sequences"]):
        padded[i, :len(sequence)] = sequence
    np.savez_compressed(folder / f"{name}_sequences.npz", values=padded,
                        lengths=data["lengths"], context=data["context"])
    data["rows"].with_columns(
        pl.Series("sequence_length", data["lengths"]),
        pl.Series("history_game_ids", data["history_ids"], dtype=pl.List(pl.String)),
    ).write_parquet(folder / f"{name}_rows.parquet", compression="zstd")
