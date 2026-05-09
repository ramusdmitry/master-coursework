from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from pipeline.src.config import load_config
from pipeline.src.data.io import load_raw_series
from pipeline.src.data.split import train_test_split_ts
from pipeline.src.evaluate.metrics import compute_metrics
from pipeline.src.evaluate.report import save_markdown_summary, save_run_report
from pipeline.src.features.build import build_basic_features
from pipeline.src.models.autots_model import AutoTSModel
from pipeline.src.models.baseline import NaiveLastValueModel
from pipeline.src.tracking.wandb_tracker import WandbTracker


def _build_model(config: dict, model_name: str):
    if model_name == 'autots':
        autots_cfg = config['models']['autots']
        return AutoTSModel(
            forecast_length=autots_cfg['forecast_length'],
            frequency=autots_cfg.get('frequency', 'infer'),
            ensemble=autots_cfg.get('ensemble', 'simple'),
            model_list=autots_cfg.get('model_list', 'superfast'),
            max_generations=int(autots_cfg.get('max_generations', 3)),
            num_validations=int(autots_cfg.get('num_validations', 2)),
        )
    if model_name == 'naive_last_value':
        return NaiveLastValueModel()
    raise ValueError(f'Неизвестная модель: {model_name}')


def run_train(config_path: str, model_name: str | None = None, run_name: str | None = None) -> dict:
    config = load_config(config_path)
    data_cfg = config['data']
    feat_cfg = config['features']
    out_cfg = config['artifacts']
    tracking_cfg = config['project']['tracking']

    selected_model = model_name or config['models']['default_model']
    run_name = run_name or f'{selected_model}-baseline'

    series_map = load_raw_series(data_cfg['input_glob'], data_cfg['timestamp_col'], data_cfg['target_col'])
    # MVP: работаем на самой длинной серии
    symbol, df = max(series_map.items(), key=lambda kv: len(kv[1]))
    feat_df = build_basic_features(
        df,
        target_col=data_cfg['target_col'],
        ma_windows=[int(x) for x in feat_cfg.get('ma_windows', [5, 10, 20])],
        returns_window=int(feat_cfg.get('returns_window', 1)),
    )
    split = train_test_split_ts(feat_df, float(data_cfg['test_size']))

    model = _build_model(config, selected_model)
    horizon = len(split.test)

    tracker = WandbTracker(
        project=tracking_cfg['wandb_project'],
        entity=tracking_cfg.get('wandb_entity'),
        mode=tracking_cfg.get('mode', 'online'),
    )
    tracker.start_run(
        run_name=run_name,
        config={**config, 'selected_symbol': symbol, 'selected_model': selected_model},
        tags=['pipeline', selected_model],
    )

    t0 = perf_counter()
    model.fit(split.train, target_col=data_cfg['target_col'])
    train_time = perf_counter() - t0

    t1 = perf_counter()
    preds = model.predict(horizon=horizon)
    infer_time = perf_counter() - t1

    y_true = split.test[data_cfg['target_col']].to_numpy(dtype=float)
    y_pred = np.asarray(preds, dtype=float)[: len(y_true)]
    metrics = compute_metrics(y_true, y_pred)
    metrics['train_time_sec'] = float(train_time)
    metrics['infer_time_sec'] = float(infer_time)

    report_payload = {
        'model': selected_model,
        'symbol': symbol,
        'metrics': metrics,
        'metadata': model.metadata(),
        'horizon': horizon,
    }

    report_path = save_run_report(out_cfg['report_dir'], f'{selected_model}_run', report_payload)
    tracker.log_metrics(metrics)
    tracker.log_artifact_file(report_path, artifact_name=f'{selected_model}_report', artifact_type='report')

    if selected_model == 'autots':
        md_path = save_markdown_summary(
            out_cfg['report_dir'],
            'autots_baseline.md',
            [
                '# AutoTS Baseline',
                f'- Symbol: {symbol}',
                f"- RMSE: {metrics['rmse']:.6f}",
                f"- MAE: {metrics['mae']:.6f}",
                f"- MAPE: {metrics['mape']:.4f}",
                f"- SMAPE: {metrics['smape']:.4f}",
                f"- Train time (sec): {metrics['train_time_sec']:.3f}",
                f"- Inference time (sec): {metrics['infer_time_sec']:.3f}",
                '',
                'Baseline выполнен до остальных этапов по плану.',
            ],
        )
        tracker.log_artifact_file(md_path, artifact_name='autots_baseline_md', artifact_type='report')

    tracker.finish()
    return report_payload


def run_evaluate(config_path: str) -> Path:
    config = load_config(config_path)
    report_dir = Path(config['artifacts']['report_dir'])
    rows = []
    for path in sorted(report_dir.glob('*_run_*.json')):
        import json

        payload = json.loads(path.read_text(encoding='utf-8'))
        rows.append((payload['model'], payload['symbol'], payload['metrics']['rmse'], payload['metrics']['mae']))

    lines = ['# Model comparison', '', '| model | symbol | rmse | mae |', '|---|---:|---:|---:|']
    for model, symbol, rmse, mae in sorted(rows, key=lambda x: x[2]):
        lines.append(f'| {model} | {symbol} | {rmse:.6f} | {mae:.6f} |')

    return save_markdown_summary(report_dir, 'comparison.md', lines)
