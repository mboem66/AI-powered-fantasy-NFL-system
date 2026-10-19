"""Small packed-sequence LSTM, training loop, and saved-model helpers."""

from copy import deepcopy
import json

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from fantasy_football.modeling.prepare import POSITIONS, TARGET
from fantasy_football.modeling.sequences import (
    CONTEXT_FIELDS, SEQUENCE_FIELDS, fit_preprocessing, transform,
)


class SeasonLSTM(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.lstm = nn.LSTM(len(SEQUENCE_FIELDS), options["hidden_size"], batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(options["hidden_size"] + len(CONTEXT_FIELDS) + len(POSITIONS), options["dense_size"]),
            nn.ReLU(), nn.Linear(options["dense_size"], 1),
        )

    def forward(self, sequences, lengths, context):
        # Packing excludes padded slots, even when their values are nonzero.
        packed = pack_padded_sequence(sequences, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.output(torch.cat([hidden[-1], context], dim=1)).squeeze(1)


def tensors(data, preprocessing):
    sequences, context = transform(data, preprocessing)
    indices = np.flatnonzero(data["lengths"] > 0)
    if not len(indices):
        return indices, None
    return indices, (
        pad_sequence([torch.from_numpy(sequences[i]) for i in indices], batch_first=True),
        torch.tensor(data["lengths"][indices]), torch.from_numpy(context[indices]),
    )


def infer(model, inputs, batch_size):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(*(x[start:start+batch_size] for x in inputs))
                          for start in range(0, len(inputs[0]), batch_size)]).numpy()


def fit(data, options, validation=None, epochs=None):
    """Early-stop on 2024 MAE, or refit for a previously selected epoch count."""
    torch.set_num_threads(1)
    torch.manual_seed(options["seed"])
    np.random.seed(options["seed"])
    torch.use_deterministic_algorithms(True)
    preprocessing = fit_preprocessing(data)
    indices, inputs = tensors(data, preprocessing)
    mean, scale = preprocessing["target_mean"], preprocessing["target_scale"]
    targets = torch.tensor((data["rows"][TARGET].to_numpy()[indices] - mean) / scale, dtype=torch.float32)
    valid_indices, valid_inputs = tensors(validation, preprocessing) if validation is not None else (None, None)
    if validation is not None and valid_inputs is None:
        raise ValueError("No validation examples have current-season history")
    model = SeasonLSTM(options)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    best_mae, best_epoch, stale, curve = float("inf"), 0, 0, []
    best_weights = None
    for epoch in range(1, (epochs or options["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(indices))
        total = 0.0
        for batch in order.split(options["batch_size"]):
            optimizer.zero_grad()
            predicted = model(*(x[batch] for x in inputs))
            loss = nn.functional.mse_loss(predicted, targets[batch])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite LSTM loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        row = {"epoch": epoch, "training_mse_scaled": total / len(indices)}
        if validation is not None:
            predicted = infer(model, valid_inputs, options["batch_size"]) * scale + mean
            actual = validation["rows"][TARGET].to_numpy()[valid_indices]
            mae = float(np.abs(predicted - actual).mean())
            row.update(validation_mae=mae, validation_rmse=float(np.sqrt(np.mean((predicted-actual)**2))))
            if mae < best_mae - options["min_delta"]:
                best_mae, best_epoch, stale = mae, epoch, 0
                best_weights = deepcopy(model.state_dict())
            else:
                stale += 1
        else:
            best_epoch, best_weights = epoch, deepcopy(model.state_dict())
        curve.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch}: {row}", flush=True)
        if validation is not None and stale >= options["patience"]:
            break
    return {"weights": best_weights, "preprocessing": preprocessing,
            "options": options, "epochs": best_epoch}, curve


def predict(bundle, data):
    torch.set_num_threads(1)
    output = np.full(data["rows"].height, np.nan)
    if not len(output):
        return output
    indices, inputs = tensors(data, bundle["preprocessing"])
    if inputs is not None:
        model = SeasonLSTM(bundle["options"])
        model.load_state_dict(bundle["weights"])
        p = bundle["preprocessing"]
        output[indices] = infer(model, inputs, bundle["options"]["batch_size"]) * p["target_scale"] + p["target_mean"]
        if not np.isfinite(output[indices]).all():
            raise ValueError("Nonfinite LSTM predictions")
    return output


def save(bundle, folder):
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(bundle["weights"], folder / "weights.pt")
    joblib.dump(bundle["preprocessing"], folder / "preprocessing.joblib")
    (folder / "architecture.json").write_text(json.dumps({
        "options": bundle["options"], "epochs": bundle["epochs"],
        "sequence_fields": SEQUENCE_FIELDS, "context_fields": CONTEXT_FIELDS, "positions": POSITIONS,
    }, indent=2), encoding="utf-8")


def load(folder):
    metadata = json.loads((folder / "architecture.json").read_text(encoding="utf-8"))
    if metadata["sequence_fields"] != SEQUENCE_FIELDS or metadata["context_fields"] != CONTEXT_FIELDS or metadata["positions"] != POSITIONS:
        raise ValueError("Saved LSTM inputs do not match the current code")
    return {"weights": torch.load(folder / "weights.pt", map_location="cpu", weights_only=True),
            "preprocessing": joblib.load(folder / "preprocessing.joblib"),
            "options": metadata["options"], "epochs": metadata["epochs"]}
