"""Empirical scoring ranges from development residuals, checked on a later season."""

import argparse
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import polars as pl

from fantasy_football.config import PROJECT_ROOT

NOTE = ("Experimental historical-error estimate, conditional on the eligible player population. "
        "Not a probability of playing. Pair probabilities assume independent errors; teammates "
        "and opposing players can have correlated outcomes. 2024 was used for model selection; "
        "2025 is retrospective, not a new untouched test. Position/projection-level residual groups "
        "approximate changing scoring variability; coverage is not guaranteed for an individual player.")


def band(prediction):
    return "under5" if prediction < 5 else "5to10" if prediction < 10 else "10plus"


def residual_pool(calibration, position, prediction):
    key = position + ":" + band(prediction)
    return np.asarray(calibration["pools"].get(key, calibration["pools"][position]))


def usable(frame):
    return frame.filter(pl.col("lstm_prediction").is_finite() &
                        pl.col("target_ppr_points").is_finite())


def fit(frame):
    """Keep position-specific residual pools; never fit on evaluation outcomes."""
    frame = usable(frame)
    pools = {}
    for position in ("QB", "RB", "WR", "TE"):
        rows = frame.filter(pl.col("position") == position)
        if rows.height < 100:
            raise ValueError(f"Insufficient development residuals for {position}")
        pools[position] = (rows["target_ppr_points"] - rows["lstm_prediction"]).to_list()
        for label, low, high in (("under5", -np.inf, 5), ("5to10", 5, 10), ("10plus", 10, np.inf)):
            candidates = [r for r in rows.to_dicts() if band(r["lstm_prediction"]) == label]
            # Sparse groups borrow the closest projected scores, not all low-usage players.
            if len(candidates) < 100:
                candidates = sorted(rows.to_dicts(), key=lambda r: max(low - r["lstm_prediction"],
                                                                      r["lstm_prediction"] - high, 0))[:100]
            pools[position + ":" + label] = [r["target_ppr_points"] - r["lstm_prediction"] for r in candidates]
    return {"method": "position_projection_residuals_v2", "coverage": 0.8, "pools": pools, "note": NOTE}


def interval(calibration, position, prediction):
    residuals = residual_pool(calibration, position, prediction)
    low, high = np.quantile(residuals, [0.1, 0.9])
    # Negative PPR is possible, so do not clip the interval at zero.
    return {"lower": float(prediction + low), "upper": float(prediction + high),
            "nominal_coverage": 0.8, "residual_count": len(residuals)}


def probability(calibration, a, b):
    """Enumerate the empirical error pairs, rounding scores to PPR hundredths."""
    left = np.round(a["prediction"] + residual_pool(calibration, a["position"], a["prediction"]), 2)
    right = np.sort(np.round(b["prediction"] + residual_pool(calibration, b["position"], b["prediction"]), 2))
    below = np.searchsorted(right, left, side="left").sum()
    through = np.searchsorted(right, left, side="right").sum()
    total = len(left) * len(right)
    return {"a_outscores_b": float(below / total), "tie": float((through - below) / total),
            "b_outscores_a": float((total - through) / total)}


def evaluate(calibration, frame):
    frame = usable(frame)
    coverage, by_projection = [], []
    for position in ("QB", "RB", "WR", "TE"):
        rows = frame.filter(pl.col("position") == position)
        bounds = [interval(calibration, position, p) for p in rows["lstm_prediction"]]
        covered = np.asarray([b["lower"] <= actual <= b["upper"] for b, actual in zip(bounds, rows["target_ppr_points"])])
        coverage.append({"position": position, "records": rows.height,
                         "coverage": float(covered.mean()),
                         "mean_width": float(np.mean([b["upper"] - b["lower"] for b in bounds]))})
        # Overall coverage can hide worse intervals for high-scoring players.
        for lower, upper in ((float("-inf"), 5), (5, 10), (10, 15), (15, float("inf"))):
            selected = (rows["lstm_prediction"].to_numpy() >= lower) & (rows["lstm_prediction"].to_numpy() < upper)
            if selected.any():
                by_projection.append({"position": position, "projection_band": f"{lower} to {upper}",
                    "records": int(selected.sum()), "coverage": float(covered[selected].mean())})
    # One deterministic, disjoint pairing per position/week avoids counting every combination.
    pairs = []
    for group in frame.sort("player_id").partition_by(["season", "week", "position"]):
        rows = group.to_dicts()
        for a, b in zip(rows[::2], rows[1::2]):
            probs = probability(calibration,
                                {"prediction": a["lstm_prediction"], "position": a["position"]},
                                {"prediction": b["lstm_prediction"], "position": b["position"]})
            actual = a["target_ppr_points"] > b["target_ppr_points"]
            pairs.append({"probability": probs["a_outscores_b"], "actual": float(actual)})
    bins = []
    for lower in np.arange(0, 1, 0.2):
        rows = [r for r in pairs if lower <= r["probability"] < lower + 0.2 + (1e-9 if lower > .79 else 0)]
        if rows:
            bins.append({"lower": float(lower), "count": len(rows),
                         "mean_probability": float(np.mean([r["probability"] for r in rows])),
                         "observed_win_rate": float(np.mean([r["actual"] for r in rows]))})
    return {"intervals": coverage, "coverage_by_projection_band": by_projection,
            "pairs": len(pairs), "pairing": "disjoint same-position/week, player ID order",
            "brier_a_wins": float(np.mean([(r["probability"] - r["actual"]) ** 2 for r in pairs])),
            "calibration_bins": bins, "note": NOTE}


def build(root=PROJECT_ROOT, run=None):
    base = root / "artifacts/task3/runs"
    folders = [base / run] if run else sorted(base.glob("[0-9]*"), reverse=True)
    source = next((p for p in folders if (p / "retrospective_2025_predictions.parquet").exists()), None)
    if source is None:
        raise ValueError("No saved Task 3 development/retrospective run found")
    files = [source / f"{label}_predictions.parquet" for label in ("validation_2024", "retrospective_2025")]
    development, testing = [pl.read_parquet(p) for p in files]
    if set(development["season"]) != {2024} or set(testing["season"]) != {2025}:
        raise ValueError("Expected development 2024 and separate retrospective 2025")
    calibration = fit(development)
    calibration.update(source_run=source.name, source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    report = evaluate(calibration, testing)
    output = root / "artifacts/task4/uncertainty" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    for name, value in (("calibration", calibration), ("evaluation", report)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    args = parser.parse_args()
    build(run=args.run)
