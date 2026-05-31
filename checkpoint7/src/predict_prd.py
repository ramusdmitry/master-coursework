"""
src/predict_prd.py — скрипт загрузки PRD-модели и тестового предикта.

Загружает модель Meta-Labeling с тегом/алиасом PRD из MLflow Model Registry,
генерирует несколько тестовых примеров (реальных или синтетических),
делает предикт и выводит результат на русском языке.

Запуск из корня checkpoint7/:
    python -m src.predict_prd
    python -m src.predict_prd mlflow.tracking_uri=http://localhost:5000

Алгоритм загрузки модели:
1. Пробуем загрузить по URI вида models:/<name>@PRD (новый API alias).
2. Если не вышло — ищем версию с тегом stage=PRD через MlflowClient.
3. Загружаем как mlflow.pyfunc.load_model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from src.common import (
    ACTIVE_ASSET,
    SEED,
    make_synthetic_splits,
    set_seed,
)


def _load_prd_model(
    registered_model_name: str,
    prd_tag: str = "PRD",
    tracking_uri: str = "http://localhost:5000",
):
    """Загружает PRD-модель из MLflow Model Registry.

    Стратегия загрузки:
    1. URI models:/<name>@<alias> (MLflow >= 2.x с aliases).
    2. Поиск версии с тегом stage=<prd_tag> через MlflowClient.
    3. Если ничего не найдено — ошибка с подробным сообщением.

    Аргументы:
        registered_model_name -- имя модели в Registry.
        prd_tag               -- значение тега/алиаса PRD.
        tracking_uri          -- URI сервера MLflow.

    Возвращает:
        mlflow.pyfunc.PyFuncModel — загруженная модель.
    """
    # Стратегия 1: загрузка по alias
    alias_uri = f"models:/{registered_model_name}@{prd_tag}"
    try:
        print(f"Попытка загрузки по alias: {alias_uri}")
        model = mlflow.pyfunc.load_model(alias_uri)
        print(f"Модель загружена по alias @{prd_tag}.")
        return model
    except Exception as e1:
        print(f"Загрузка по alias не удалась: {e1}")

    # Стратегия 2: поиск версии с тегом stage=PRD
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        versions = client.search_model_versions(
            f"name='{registered_model_name}'"
        )
        prd_versions = [
            v for v in versions
            if v.tags.get("stage") == prd_tag
        ]
        if prd_versions:
            # Берём самую новую версию с тегом PRD
            best = max(prd_versions, key=lambda v: int(v.version))
            version_uri = f"models:/{registered_model_name}/{best.version}"
            print(f"Загрузка версии {best.version} с тегом stage={prd_tag}: {version_uri}")
            model = mlflow.pyfunc.load_model(version_uri)
            print(f"Модель версии {best.version} загружена.")
            return model
        else:
            raise RuntimeError(
                f"Версий с тегом stage={prd_tag} не найдено для модели '{registered_model_name}'. "
                "Убедитесь, что src/train.py был успешно выполнен."
            )
    except RuntimeError:
        raise
    except Exception as e2:
        raise RuntimeError(
            f"Не удалось загрузить PRD-модель из Registry: {e2}\n"
            "Проверьте, что MLflow-сервер запущен и модель зарегистрирована."
        ) from e2


def _build_test_inputs(
    cfg: DictConfig,
    n_samples: int = 5,
) -> pd.DataFrame:
    """Строит тестовые входы для инференса.

    Если реальные данные недоступны — генерирует синтетику.
    Возвращает DataFrame с n_features колонками признаков актива.

    Аргументы:
        cfg       -- Hydra-конфиг.
        n_samples -- число примеров для предикта.

    Возвращает:
        pd.DataFrame (n_samples + window, n_features).
    """
    asset = cfg.data.get("active_asset", ACTIVE_ASSET)
    data_dir = Path(cfg.data.data_dir)
    window = cfg.model.window

    use_synthetic = cfg.data.get("synthetic", True)
    if not use_synthetic:
        x_test_path = data_dir / cfg.data.splits.X_test
        if not x_test_path.exists():
            use_synthetic = True

    if use_synthetic:
        print("Синтетические входные данные для предикта...")
        _, _, _, _, X_test, _ = make_synthetic_splits(
            n_train=200, n_val=50, n_test=window + n_samples + 10,
            n_features=20, asset=asset, seed=cfg.seed,
        )
        prefix = f"{asset}__"
        feat_cols = [c for c in X_test.columns if str(c).startswith(prefix)]
        if not feat_cols:
            feat_cols = list(X_test.columns)
        return X_test[feat_cols]
    else:
        X_test = pd.read_parquet(data_dir / cfg.data.splits.X_test)
        prefix = f"{asset}__"
        feat_cols = [c for c in X_test.columns if str(c).startswith(prefix)]
        # Берём достаточно строк для window + n_samples
        return X_test[feat_cols].iloc[: window + n_samples + 10]


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Загружает PRD-модель из MLflow и делает тестовый предикт.

    Выводит таблицу с позициями и вероятностями для нескольких примеров.
    """
    set_seed(cfg.seed)

    tracking_uri = cfg.mlflow.get("tracking_uri", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    registered_model_name = cfg.mlflow.registered_model_name
    prd_tag = cfg.mlflow.get("prd_tag", "PRD")

    print("=" * 60)
    print(f"Загрузка PRD-модели: {registered_model_name}")
    print(f"MLflow: {tracking_uri}")
    print("=" * 60)

    # Загрузка модели
    model = _load_prd_model(
        registered_model_name=registered_model_name,
        prd_tag=prd_tag,
        tracking_uri=tracking_uri,
    )

    # Подготовка тестовых входов
    print("\nПодготовка тестовых входных данных...")
    n_samples = 5
    try:
        X_input = _build_test_inputs(cfg, n_samples=n_samples)
    except Exception as e:
        print(f"Ошибка подготовки данных: {e}")
        sys.exit(1)

    print(f"Форма входных данных: {X_input.shape}")

    # Инференс
    print("\nВыполняю предикт...")
    try:
        result_df = model.predict(X_input)
    except Exception as e:
        print(f"Ошибка инференса: {e}")
        raise

    # Вывод результатов
    print("\n" + "=" * 60)
    print("Результаты предикта (Meta-Labeling PRD-модель):")
    print("=" * 60)
    print(result_df.to_string(index=True))
    print()

    # Краткая интерпретация
    if "position" in result_df.columns:
        n_long = (result_df["position"] > 0).sum()
        n_flat = (result_df["position"] == 0).sum()
        mean_pos = result_df["position"].mean()
        print(f"Итого примеров: {len(result_df)}")
        print(f"  long (позиция > 0): {n_long}")
        print(f"  flat (позиция = 0): {n_flat}")
        print(f"  средний размер позиции: {mean_pos:.4f}")

    if "primary_prob" in result_df.columns:
        print(f"  средняя уверенность primary: {result_df['primary_prob'].mean():.4f}")

    print("\nПредикт успешно выполнен.")


if __name__ == "__main__":
    main()
