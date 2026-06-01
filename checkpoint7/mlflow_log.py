"""
mlflow_log.py — собрать MLflow-трекинг чекпоинта из наших результатов.

Логирует наши прогоны (CSV) как MLflow runs по экспериментам + продакшн-модель как
артефакт. Локальный sqlite-стор. После запуска поднять UI:

    ~/.envs/ds/bin/python mlflow_log.py
    ~/.envs/ds/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""
from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
mlflow.set_tracking_uri(f"sqlite:///{HERE / 'mlflow.db'}")

# карта «колонка CSV -> имя метрики» (ASCII-safe для mlflow)
MET = {"ROC-AUC": "roc_auc", "Sh maker": "sharpe_maker", "BH": "sharpe_bh"}


def log_csv(exp: str, csv: Path, param_cols, name_cols, note: str = ""):
    if not csv.exists():
        print(f"[skip] нет {csv.name}"); return
    df = pd.read_csv(csv)
    mlflow.set_experiment(exp)
    for _, row in df.iterrows():
        rn = "_".join(str(row[c]) for c in name_cols)
        with mlflow.start_run(run_name=rn):
            for c in param_cols:
                if c in row:
                    mlflow.log_param(c, row[c])
            for c, mk in MET.items():
                if c in row and pd.notna(row[c]):
                    mlflow.log_metric(mk, float(row[c]))
            if note:
                mlflow.set_tag("note", note)
    print(f"[ok] {exp}: {len(df)} runs <- {csv.name}")


def log_model():
    p = HERE.parent / "models" / "model.pth"
    if not p.exists():
        print("[skip] нет models/model.pth"); return
    ck = torch.load(p, map_location="cpu")
    mlflow.set_experiment("production_model")
    with mlflow.start_run(run_name=ck.get("strategy", "SimpleLSTM_BTC")):
        for k in ("input_size", "hidden_size", "num_layers", "dropout",
                  "window_size", "asset", "horizon_minutes"):
            if k in ck:
                mlflow.log_param(k, ck[k])
        if "val_roc_auc" in ck:
            mlflow.log_metric("val_roc_auc", float(ck["val_roc_auc"]))
        mlflow.log_artifact(str(p))
        mlflow.set_tag("note", "продакшн-модель демо; val ROC 0.516 ≈ монетка")
    print("[ok] production_model: модель залогирована")


def main():
    log_csv("subminute_2025", HERE / "sub_polars_results.csv",
            ["Актив", "TF", "train_bars", "test"], ["Актив", "TF"],
            "полное обучение 70% 2025, без капа")
    log_csv("subminute_2023_2025", HERE / "sub_2023_2025_results.csv",
            ["Актив", "TF", "train_bars", "test"], ["Актив", "TF"],
            "2.75 года; 20/20 не торгуемо")
    log_csv("spectrum_2021_2025", HERE / "full_spectrum_results.csv",
            ["Актив", "TF", "test"], ["Актив", "TF"], "статический спектр tf x активы")
    log_csv("subminute_capped", HERE / "sub_assets_results.csv",
            ["Актив", "TF", "test"], ["Актив", "TF"], "кап 18 дней (артефакты)")
    log_model()
    print(f"\nГотово. UI: ~/.envs/ds/bin/mlflow ui --backend-store-uri "
          f"sqlite:///{HERE / 'mlflow.db'} --port 5000")


if __name__ == "__main__":
    main()
