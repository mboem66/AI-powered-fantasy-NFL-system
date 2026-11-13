"""Additional sequence architectures using the existing training/preprocessing."""

import json

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from fantasy_football.modeling import lstm
from fantasy_football.modeling.transformer import SeasonTransformer
from fantasy_football.modeling.sequences import SEQUENCE_FIELDS, CONTEXT_FIELDS
from fantasy_football.modeling.prepare import POSITIONS


class DeepLSTM(lstm.SeasonLSTM):
    def __init__(self, options):
        super().__init__(options)
        self.lstm = nn.LSTM(len(SEQUENCE_FIELDS), options["hidden_size"],
                            num_layers=options["num_layers"], dropout=options["dropout"], batch_first=True)


class TransformerLSTM(SeasonTransformer):
    def __init__(self, options):
        super().__init__(options)
        hidden = options["lstm_hidden_size"]
        self.temporal = nn.LSTM(options["hidden_size"], hidden, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden + len(CONTEXT_FIELDS) + len(POSITIONS), options["dense_size"]),
            nn.ReLU(), nn.Linear(options["dense_size"], 1))

    def forward(self, sequences, lengths, context):
        encoded, padding = self.encode(sequences, lengths)
        # Attention sees only prior-game tokens. Packing keeps padded steps out
        # of the LSTM, while retaining chronological order among real games.
        packed = pack_padded_sequence(encoded.masked_fill(padding[:, :, None], 0),
                                      lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.temporal(packed)
        return self.output(torch.cat([hidden[-1], context], dim=1)).squeeze(1)


FACTORIES = {"deep_lstm": DeepLSTM, "deep_transformer": SeasonTransformer,
             "transformer_lstm": TransformerLSTM}


def fit(name, data, options, validation=None, epochs=None):
    bundle, curve = lstm.fit(data, options, validation, epochs, model_factory=FACTORIES[name])
    return bundle | {"model_type": name}, curve


def predict(bundle, data):
    return lstm.predict(bundle, data, model_factory=FACTORIES[bundle["model_type"]])


def save(bundle, folder):
    lstm.save(bundle, folder)
    path = folder / "architecture.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | {"model_type": bundle["model_type"]}, indent=2))


def load(folder):
    name = json.loads((folder / "architecture.json").read_text())["model_type"]
    if name not in FACTORIES:
        raise ValueError("Unknown experimental model")
    return lstm.load(folder) | {"model_type": name}
