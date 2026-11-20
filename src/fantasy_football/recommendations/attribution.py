"""Integrated gradients for the saved LSTM, in PPR points relative to a reference.

Method: https://proceedings.mlr.press/v70/sundararajan17a.html
"""

import numpy as np
import torch

from fantasy_football.modeling import lstm
from fantasy_football.modeling.sequences import CONTEXT_FIELDS, SEQUENCE_FIELDS


def explain(bundle, data, expected_prediction=None):
    """Explain one row; keep position and history length fixed along the path."""
    if data["rows"].height != 1 or data["lengths"][0] < 1:
        raise ValueError("Attribution requires one player with prior appearances")
    torch.set_num_threads(1)
    _, (sequence, lengths, context) = lstm.tensors(data, bundle["preprocessing"])
    model = lstm.SeasonLSTM(bundle["options"])
    model.load_state_dict(bundle["weights"])
    model.eval()
    base_sequence = torch.zeros_like(sequence)
    base_context = torch.zeros_like(context)
    base_context[:, len(CONTEXT_FIELDS):] = context[:, len(CONTEXT_FIELDS):]
    base_context[:, CONTEXT_FIELDS.index("sequence_length")] = context[:, CONTEXT_FIELDS.index("sequence_length")]
    p = bundle["preprocessing"]
    scale, mean = p["target_scale"], p["target_mean"]
    with torch.no_grad():
        predicted = float(model(sequence, lengths, context)[0]) * scale + mean
        reference = float(model(base_sequence, lengths, base_context)[0]) * scale + mean
    if expected_prediction is not None and not np.isclose(predicted, expected_prediction, atol=0.001):
        raise ValueError("Saved model and forecast disagree; attribution withheld")
    # Trapezoid integration; refine once if the completeness check needs it.
    for steps in (64, 256):
        sequence_grad, context_grad = torch.zeros_like(sequence), torch.zeros_like(context)
        for i in range(steps + 1):
            alpha = i / steps
            s = (base_sequence + alpha * (sequence - base_sequence)).requires_grad_(True)
            c = (base_context + alpha * (context - base_context)).requires_grad_(True)
            gs, gc = torch.autograd.grad(model(s, lengths, c).sum(), (s, c))
            weight = 0.5 if i in (0, steps) else 1.0
            sequence_grad += gs.detach() * weight / steps
            context_grad += gc.detach() * weight / steps
        seq = ((sequence - base_sequence) * sequence_grad * scale).sum(dim=(0, 1)).numpy()
        ctx = ((context - base_context) * context_grad * scale).sum(dim=0).numpy()
        residual = predicted - reference - float(seq.sum() + ctx.sum())
        if abs(residual) <= 0.05:
            break
    if abs(residual) > 0.05:
        raise ValueError(f"Attribution completeness error {residual:.3f} PPR exceeds tolerance")
    contributions = {name: float(value) for name, value in zip(SEQUENCE_FIELDS, seq)}
    contributions.update({"context_" + name: float(ctx[i]) for i, name in enumerate(CONTEXT_FIELDS)})
    return {"method": "integrated_gradients", "prediction": predicted, "reference_prediction": reference,
            "contributions_ppr": contributions, "completeness_error_ppr": residual, "integration_steps": steps,
            "note": "Relative to training-mean scaled inputs, holding position and sequence length fixed. "
                    "Prior-game contributions are summed across appearances. Reference is synthetic, not an "
                    "average player forecast. Contributions describe model behavior, not causal effects."}
