from __future__ import annotations

import optuna

from pipeline.src.train.runner import run_train
from pipeline.src.config import load_config
from pipeline.src.tracking.wandb_tracker import WandbTracker


def run_hpo(config_path: str) -> dict:
    config = load_config(config_path)
    hpo_cfg = config['hpo']
    tracking_cfg = config['project']['tracking']

    tracker = WandbTracker(
        project=tracking_cfg['wandb_project'],
        entity=tracking_cfg.get('wandb_entity'),
        mode=tracking_cfg.get('mode', 'online'),
    )
    tracker.start_run('optuna-autots', config=config, tags=['pipeline', 'hpo', 'autots'])

    def objective(trial: optuna.trial.Trial) -> float:
        low_g, high_g = hpo_cfg['search_space']['autots_max_generations']
        low_v, high_v = hpo_cfg['search_space']['autots_num_validations']

        config['models']['autots']['max_generations'] = trial.suggest_int('autots_max_generations', int(low_g), int(high_g))
        config['models']['autots']['num_validations'] = trial.suggest_int('autots_num_validations', int(low_v), int(high_v))

        import tempfile
        import yaml
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / 'trial.yaml'
            cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            result = run_train(str(cfg_path), model_name='autots', run_name=f'autots-trial-{trial.number}')
        rmse = float(result['metrics']['rmse'])
        tracker.log_metrics({'trial/rmse': rmse, 'trial/number': trial.number})
        return rmse

    study = optuna.create_study(direction=hpo_cfg.get('direction', 'minimize'))
    study.optimize(objective, n_trials=int(hpo_cfg.get('n_trials', 10)))

    best = {
        'best_value': float(study.best_value),
        'best_params': study.best_params,
        'n_trials': len(study.trials),
    }
    tracker.log_metrics({'hpo/best_rmse': best['best_value'], 'hpo/n_trials': best['n_trials']})
    tracker.finish()
    return best
