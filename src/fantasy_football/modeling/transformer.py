"""Small attention model for experiments on existing season sequences."""

from copy import deepcopy
import json
import math

import numpy as np
import torch
from torch import nn

from fantasy_football.modeling import lstm
from fantasy_football.modeling.prepare import POSITIONS, TARGET
from fantasy_football.modeling.sequences import CONTEXT_FIELDS, SEQUENCE_FIELDS, fit_preprocessing


class SeasonTransformer(nn.Module):
    def __init__(self, options):
        super().__init__()
        width = options["hidden_size"]
        self.project = nn.Linear(len(SEQUENCE_FIELDS), width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=options["attention_heads"],
            dim_feedforward=options["feedforward_size"], dropout=options["dropout"],
            batch_first=True,
        )
        # Older saved experiments default to the original single encoder block.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=options.get("num_layers", 1), enable_nested_tensor=False,
        )
        self.output = nn.Sequential(
            nn.Linear(width + len(CONTEXT_FIELDS) + len(POSITIONS), options["dense_size"]),
            nn.ReLU(), nn.Linear(options["dense_size"], 1),
        )

    def forward(self, sequences, lengths, context):
        if torch.any(lengths <= 0) or torch.any(lengths > sequences.shape[1]):
            raise ValueError("Transformer requires nonempty, valid sequence lengths")
        steps = torch.arange(sequences.shape[1], device=sequences.device)
        padding = steps[None, :] >= lengths[:, None]
        # Mask before projection too, so even NaN padding cannot affect attention.
        values = self.project(sequences.masked_fill(padding[:, :, None], 0.0))
        width = values.shape[-1]
        frequency = torch.exp(torch.arange(0, width, 2, device=values.device) * (-math.log(10000) / width))
        angles = steps[:, None] * frequency
        positions = torch.zeros_like(values[0])
        positions[:, 0::2] = angles.sin()
        positions[:, 1::2] = angles[:, :width // 2].cos()
        # All real timesteps are earlier games; no target-game token is present.
        encoded = self.encoder(values + positions, src_key_padding_mask=padding)
        pooled = encoded.masked_fill(padding[:, :, None], 0.0).sum(1) / lengths[:, None]
        return self.output(torch.cat([pooled, context], dim=1)).squeeze(1)


def fit(data, options, validation):
    """Train on past seasons and choose the checkpoint using 2024 MAE only."""
    torch.set_num_threads(1)
    torch.manual_seed(options["seed"])
    np.random.seed(options["seed"])
    torch.use_deterministic_algorithms(True)
    preprocessing = fit_preprocessing(data)
    indices, inputs = lstm.tensors(data, preprocessing)
    valid_indices, valid_inputs = lstm.tensors(validation, preprocessing)
    if valid_inputs is None:
        raise ValueError("No validation sequences")
    mean, scale = preprocessing["target_mean"], preprocessing["target_scale"]
    targets = torch.tensor((data["rows"][TARGET].to_numpy()[indices] - mean) / scale, dtype=torch.float32)
    actual = validation["rows"][TARGET].to_numpy()[valid_indices]
    model = SeasonTransformer(options)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    best_mae, best_epoch, stale, weights, curve = float("inf"), 0, 0, None, []
    for epoch in range(1, options["max_epochs"] + 1):
        model.train()
        total = 0.0
        for batch in torch.randperm(len(indices)).split(options["batch_size"]):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(*(x[batch] for x in inputs)), targets[batch])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite Transformer loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        predicted = lstm.infer(model, valid_inputs, options["batch_size"]) * scale + mean
        if not np.isfinite(predicted).all():
            raise ValueError("Nonfinite validation predictions")
        mae = float(np.abs(predicted - actual).mean())
        curve.append({"epoch": epoch, "training_mse_scaled": total / len(indices),
                      "validation_mae": mae, "validation_rmse": float(np.sqrt(np.mean((predicted - actual)**2)))})
        if mae < best_mae - options["min_delta"]:
            best_mae, best_epoch, stale, weights = mae, epoch, 0, deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch}: {curve[-1]}", flush=True)
        if stale >= options["patience"]:
            break
    return {"weights": weights, "preprocessing": preprocessing, "options": options, "epochs": best_epoch}, curve


def predict(bundle, data):
    torch.set_num_threads(1)
    output = np.full(data["rows"].height, np.nan)
    if not len(output):
        return output
    indices, inputs = lstm.tensors(data, bundle["preprocessing"])
    if inputs is not None:
        model = SeasonTransformer(bundle["options"])
        model.load_state_dict(bundle["weights"])
        p = bundle["preprocessing"]
        output[indices] = lstm.infer(model, inputs, bundle["options"]["batch_size"]) * p["target_scale"] + p["target_mean"]
        if not np.isfinite(output[indices]).all():
            raise ValueError("Nonfinite Transformer predictions")
    return output


def save(bundle, folder):
    lstm.save(bundle, folder)
    metadata = folder / "architecture.json"
    metadata.write_text(json.dumps(json.loads(metadata.read_text()) | {"model_type": "transformer"}, indent=2), encoding="utf-8")


def load(folder):
    if json.loads((folder / "architecture.json").read_text())["model_type"] != "transformer":
        raise ValueError("Expected a saved Transformer")
    return lstm.load(folder)
