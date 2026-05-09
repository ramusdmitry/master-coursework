from __future__ import annotations

import argparse
import os
from pathlib import Path
import json

from pipeline.src.config import load_config


def cmd_prepare(config_path: str) -> None:
    cfg = load_config(config_path)
    Path(cfg['artifacts']['output_dir']).mkdir(parents=True, exist_ok=True)
    Path(cfg['artifacts']['report_dir']).mkdir(parents=True, exist_ok=True)
    print('Prepared directories')


def main() -> None:
    # На некоторых macOS/NumPy сборках проверка Accelerate может падать сегфолтом.
    os.environ.setdefault('NPY_DISABLE_MACOS_ACCELERATE_CHECK', '1')

    parser = argparse.ArgumentParser(description='Pipeline CLI')
    sub = parser.add_subparsers(dest='command', required=True)

    for name in ('prepare', 'train', 'evaluate', 'hpo'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)

    compare = sub.add_parser('compare')
    compare.add_argument('--config', default='pipeline/configs/base.yaml')

    export_backend = sub.add_parser('export-backend')
    export_backend.add_argument('--source-model', required=True)
    export_backend.add_argument('--source-scaler', required=True)
    export_backend.add_argument('--backend-models-dir', default='models')

    train = sub.choices['train']
    train.add_argument('--model', default=None)
    train.add_argument('--run-name', default=None)

    args = parser.parse_args()

    if args.command == 'prepare':
        cmd_prepare(args.config)
    elif args.command == 'train':
        from pipeline.src.train.runner import run_train

        result = run_train(args.config, model_name=args.model, run_name=args.run_name)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == 'evaluate':
        from pipeline.src.train.runner import run_evaluate

        result = run_evaluate(args.config)
        print(f'Report written to {result}')
    elif args.command == 'hpo':
        from pipeline.src.hpo.optuna_search import run_hpo

        result = run_hpo(args.config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == 'compare':
        from pipeline.src.train.runner import run_evaluate

        result = run_evaluate(args.config)
        print(f'Report written to {result}')
    elif args.command == 'export-backend':
        from pipeline.src.compat.export_backend import export_backend_artifacts

        result = export_backend_artifacts(
            source_model_path=args.source_model,
            source_scaler_path=args.source_scaler,
            backend_models_dir=args.backend_models_dir,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
