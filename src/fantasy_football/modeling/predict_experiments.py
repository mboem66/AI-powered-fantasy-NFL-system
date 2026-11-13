"""Predict with three experimental architectures using a saved LSTM input snapshot."""

import argparse
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib

import joblib
import numpy as np
import polars as pl

from fantasy_football.config import PROJECT_ROOT
from fantasy_football.features.build_features import EASTERN
from fantasy_football.modeling import experimental_models as models
from fantasy_football.modeling.prepare import select_records, model_inputs, IDENTIFIERS, TARGET
from fantasy_football.modeling.sequences import make_sequences, save_sequences
from fantasy_football.modeling.train_lstm import load_history
from fantasy_football.modeling.weekly_report import kickoff


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def select_epochs(root, games, features, settings):
    training, _ = select_records(features, [2021, 2022, 2023])
    validation, _ = select_records(features, [2024])
    historical = games.filter(pl.col('season').is_between(2021, 2024))
    identity = {'settings': settings, 'architecture_version': 1,
                'features_sha256': sha256(pl.concat([training, validation]).write_csv().encode()).hexdigest(),
                'games_sha256': sha256(historical.write_csv().encode()).hexdigest()}
    version = fingerprint(identity)
    destination = root / 'artifacts/model_experiments/selections' / version
    if (destination / 'summary.json').exists():
        return destination, json.loads((destination / 'summary.json').read_text())
    train_data, valid_data = make_sequences(training, historical), make_sequences(validation, historical)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix='pending_') as temp:
        folder = Path(temp) / 'result'
        folder.mkdir()
        report = {'version': version, 'identity': identity, 'models': {},
                  'note': 'Epoch selection on 2024 only. No 2025 or current-season outcomes used for selection.'}
        validation_predictions = validation.select(IDENTIFIERS + [TARGET])
        for name, architecture in settings['models'].items():
            options = settings['training'] | architecture
            print(f'Selecting {name} epochs on 2024 validation...', flush=True)
            bundle, curve = models.fit(name, train_data, options, validation=valid_data)
            folder_model = folder / name
            models.save(bundle, folder_model)
            predicted = models.predict(models.load(folder_model), valid_data)
            np.testing.assert_allclose(predicted, models.predict(bundle, valid_data), equal_nan=True)
            valid = valid_data['lengths'] > 0
            if not np.isfinite(predicted[valid]).all():
                raise ValueError('Invalid validation predictions')
            errors = predicted[valid] - validation[TARGET].to_numpy()[valid]
            report['models'][name] = {'options': options, 'epochs': bundle['epochs'],
                                     'records': int(valid.sum()), 'mae': float(np.abs(errors).mean()),
                                     'rmse': float(np.sqrt((errors**2).mean()))}
            validation_predictions = validation_predictions.with_columns(pl.Series(name, predicted).fill_nan(None))
            pl.DataFrame(curve).write_csv(folder_model / 'learning_curve.csv')
        validation_predictions.write_parquet(folder / 'validation_predictions.parquet')
        (folder / 'summary.json').write_text(json.dumps(report, indent=2))
        folder.rename(destination)
    return destination, report


def run(root=PROJECT_ROOT, source_run=None, week=None):
    now = datetime.now(timezone.utc)
    settings = tomllib.loads((root / 'config/experiments.toml').read_text())
    weekly = root / 'artifacts/task3/weekly'
    folders = sorted(p.parent for p in weekly.glob('*/summary.json'))
    source = weekly / source_run if source_run else folders[-1] if folders else None
    if source is None:
        raise ValueError('Generate a fresh predict_lstm run first')
    info = json.loads((source / 'summary.json').read_text())
    cutoff = date.fromisoformat(info['feature_cutoff_exclusive'])
    if cutoff != now.astimezone(EASTERN).date():
        raise ValueError('LSTM input snapshot is not from today. Run predict_lstm first.')
    max_age = tomllib.loads((root / 'config/api_nfl.toml').read_text())['max_snapshot_age_hours']
    created = datetime.fromisoformat(info['created_at_utc'])
    availability = info['availability']
    snapshot_age = availability.get('age_hours', 0) + (now - created).total_seconds()/3600
    if availability.get('enabled') and not 0 <= snapshot_age <= max_age:
        raise ValueError('Source availability snapshot is stale. Refresh availability and run predict_lstm.')
    rows = pl.read_parquet(source / 'upcoming_rows.parquet')
    predictions = pl.read_parquet(source / 'predictions.parquet')
    if rows.select('player_id','game_id').to_dicts() != predictions.select('player_id','game_id').to_dicts():
        raise ValueError('Source prediction/sequence row order mismatch')
    games, features = load_history(root, cutoff=cutoff)
    future = make_sequences(rows, games, cutoff=cutoff)
    if future['history_ids'] != rows['history_game_ids'].to_list():
        raise ValueError('Player history changed since the LSTM snapshot. Run predict_lstm again.')
    with np.load(source / 'upcoming_sequences.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['lengths'], future['lengths'])
        np.testing.assert_allclose(saved['context'], future['context'], equal_nan=True)
        for i, sequence in enumerate(future['sequences']):
            np.testing.assert_allclose(sequence, saved['values'][i, :len(sequence)], rtol=1e-5, atol=1e-5, equal_nan=True)
    schedules = pl.read_parquet(root / f"data/raw/schedules/{rows['season'][0]}.parquet")
    schedule_map = {r['game_id']:r for r in schedules.to_dicts()}
    def eligible(frame, at):
        return pl.Series([bool(kickoff(schedule_map[r['game_id']]) and kickoff(schedule_map[r['game_id']]) > at
                               and schedule_map[r['game_id']]['home_score'] is None
                               and schedule_map[r['game_id']]['away_score'] is None
                               and r['sequence_length'] > 0 and (week is None or r['week'] == week))
                          for r in frame.to_dicts()])
    if not eligible(rows, now).any():
        print('No eligible unstarted games in the saved snapshot.')
        return None
    selection_path, selection = select_epochs(root, games, features, settings)
    train_cutoff = date.fromisoformat(info['training_cutoff_exclusive'])
    if train_cutoff > cutoff:
        raise ValueError('Training cutoff exceeds feature cutoff')
    history = features.filter((pl.col('game_date') < train_cutoff) & (pl.col('season') >= 2021))
    training, _ = select_records(history, sorted(history['season'].unique().to_list()))
    train_data = make_sequences(training, games, cutoff=train_cutoff)
    # References share the experimental cutoff and snapshot. The Linear reference
    # is the model saved alongside this LSTM, not the separate Task 2 live model.
    linear = joblib.load(source / 'linear.joblib').predict(model_inputs(rows))
    values = {'linear_control': linear, 'lstm_control': predictions['lstm_prediction'].to_numpy()}
    version = fingerprint({'selection': selection['version'], 'training_cutoff': str(train_cutoff),
                           'source_lstm_options': info['options'], 'source_lstm_epochs': info['epochs']})
    parent = root / 'artifacts/model_experiments/weekly'
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=parent, prefix='pending_') as temp:
        folder = Path(temp) / 'result'
        folder.mkdir()
        for name, selected in selection['models'].items():
            print(f"Refitting {name} for {selected['epochs']} epochs; training before {train_cutoff}", flush=True)
            bundle, curve = models.fit(name, train_data, selected['options'], epochs=selected['epochs'])
            models.save(bundle, folder / name / 'model')
            values[name] = models.predict(models.load(folder / name / 'model'), future)
            pl.DataFrame(curve).write_csv(folder / name / 'training_curve.csv')
        finished = datetime.now(timezone.utc)
        keep = eligible(rows, finished)
        if not keep.any():
            print('Games started during training; no forecasts published.')
            return None
        if availability.get('enabled') and availability.get('age_hours',0)+(finished-created).total_seconds()/3600 > max_age:
            raise ValueError('Availability became stale during training; refresh and retry')
        saved_rows = rows.filter(keep)
        shared = predictions.drop([c for c in ['predicted_ppr_points','prediction_method','predicted_at_utc'] if c in predictions.columns])
        for name, raw in values.items():
            if not np.isfinite(raw[keep.to_numpy()]).all():
                raise ValueError(f'Nonfinite forecasts for {name}')
            frame = shared.with_columns(pl.Series('predicted_ppr_points',raw),pl.lit(name).alias('prediction_method'),
                        pl.lit(finished.isoformat()).alias('predicted_at_utc')).filter(keep)
            destination = folder / name
            destination.mkdir(exist_ok=True)
            frame.write_parquet(destination / 'predictions.parquet')
            frame.write_csv(destination / 'predictions.csv')
        save_sequences(make_sequences(saved_rows,games,cutoff=cutoff),folder,'upcoming')
        metadata = {'created_at_utc':finished.isoformat(),'experiment_version':version,
                    'source_lstm_run':str(source),'epoch_selection':str(selection_path),'models':list(values),
                    'feature_cutoff_exclusive':str(cutoff),'training_cutoff_exclusive':str(train_cutoff),
                    'availability':availability,'counts_by_week':saved_rows.group_by('season','week').len().sort('season','week').to_dicts(),
                    'excluded_no_sequence_or_started_or_other_week':rows.height-saved_rows.height,
                    'training_features_sha256':sha256(training.write_csv().encode()).hexdigest(),
                    'source_hashes':{p.name:sha256(p.read_bytes()).hexdigest() for p in [source/'upcoming_rows.parquet',source/'upcoming_sequences.npz',source/'predictions.parquet']},
                    'note':'Same snapshot, training cutoff and scored population for all five models. CPU. No fallback. Frozen training cutoff inherited from LSTM. Forecast timestamps reflect completion, not training start.'}
        (folder/'summary.json').write_text(json.dumps(metadata,indent=2))
        output=parent/finished.strftime('%Y%m%dT%H%M%S%fZ')
        folder.rename(output)
    print('Saved five-model forecasts:',output,flush=True)
    print('Experiment version:',version,'| Counts:',metadata['counts_by_week'],flush=True)
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',help='Source LSTM weekly folder name; defaults to latest')
    parser.add_argument('--week',type=int,help='Restrict to an NFL week; otherwise all unstarted source games')
    args=parser.parse_args()
    if args.run and Path(args.run).name != args.run:
        parser.error('--run must be a folder name')
    run(source_run=args.run,week=args.week)
