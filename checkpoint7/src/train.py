"""
src/train.py — скрипт обучения Meta-Labeling модели с логированием в MLflow.

Запуск из корня checkpoint7/:
    python -m src.train
    python -m src.train data.synthetic=true
    python -m src.train model.hidden_size=128 model.epochs=50

Логика:
1. Инициализация seed.
2. Настройка MLflow (tracking_uri, experiment).
3. Загрузка данных (реальные parquet или синтетика).
4. Подготовка признаков и forward-return таргетов.
5. Обучение MetaLabelingModel.
6. Логирование: параметры, метрики, артефакты (confusion_matrix.png,
   learning_curve.png, sample_predictions.csv).
7. Регистрация модели в MLflow Model Registry + тег PRD.
"""

from __future__ import annotations

import io
import os
import pickle
import sys
import tempfile
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")  # без GUI-бэкенда
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.common import (
    ACTIVE_ASSET,
    COST_BPS,
    COST_PER_TURN,
    HORIZON,
    MIN_HOLDING,
    PER_YEAR_1M,
    PER_YEAR_FWD,
    PT_MULT,
    SEED,
    SL_MULT,
    VERTICAL_TB,
    VOL_WINDOW_TB,
    WindowDataset,
    apply_min_holding,
    evaluate_on_forward,
    forward_return,
    load_splits,
    make_synthetic_splits,
    meta_labels,
    resolve_tracking_uri,
    resolve_y_column,
    set_seed,
    triple_barrier_labels,
    window_aggregates,
    _predict_labels_probs,
)
from src.model import MetaLabelingModel, MetaLabelingPyfunc


def _prepare_data(cfg: DictConfig):
    """Загружает или генерирует сплиты и готовит признаки / таргеты.

    Если cfg.data.synthetic=true или реальные parquet недоступны —
    переходит в синтетический режим (используется make_synthetic_splits).

    Возвращает:
        dict с ключами:
            Xt_s, Xv_s, Xte_s     -- numpy-массивы признаков (нормализованы);
            yt_tb, yv_tb, yte_tb   -- метки triple-barrier;
            fwd_tr, fwd_va, fwd_te -- pd.Series forward-return;
            idx_tr_w, idx_va_w, idx_te_w -- индексы окон;
            n_features             -- число признаков;
            scaler                 -- подогнанный StandardScaler;
            synthetic              -- True если синтетика.
    """
    asset = cfg.data.get("active_asset", ACTIVE_ASSET)
    use_synthetic = cfg.data.get("synthetic", False)
    data_dir = Path(cfg.data.data_dir)

    # Проверяем наличие реальных данных
    if not use_synthetic:
        x_train_path = data_dir / cfg.data.splits.X_train
        if not x_train_path.exists():
            print(
                f"Предупреждение: {x_train_path} не найден. "
                "Переходим в синтетический режим."
            )
            use_synthetic = True

    if use_synthetic:
        print("Режим: синтетические данные (fallback).")
        X_train, y_train, X_val, y_val, X_test, y_test = make_synthetic_splits(
            n_train=5000,
            n_val=1000,
            n_test=1000,
            n_features=20,
            asset=asset,
            seed=cfg.seed,
        )
    else:
        print(f"Загрузка реальных данных из {data_dir}...")
        X_train, y_train, X_val, y_val, X_test, y_test = load_splits(data_dir)

    # Признаки с префиксом актива
    prefix = f"{asset}__"
    feat_cols = [c for c in X_train.columns if str(c).startswith(prefix)]
    if not feat_cols:
        # Если нет префикса — берём все числовые признаки (синтетика)
        feat_cols = list(X_train.columns)
    n_features = len(feat_cols)
    print(f"Признаков: {n_features}")

    # Таргет y_reg для triple-barrier
    try:
        col_reg_tr = resolve_y_column(y_train, asset, "y_reg")
        s_tr = y_train[col_reg_tr].dropna()
        col_reg_va = resolve_y_column(y_val, asset, "y_reg")
        s_va = y_val[col_reg_va].dropna()
        col_reg_te = resolve_y_column(y_test, asset, "y_reg")
        s_te = y_test[col_reg_te].dropna()
    except KeyError:
        # Синтетика: используем y_reg напрямую из MultiIndex
        col_reg_tr = resolve_y_column(y_train, asset, "y_reg")
        s_tr = y_train[col_reg_tr].dropna()
        col_reg_va = resolve_y_column(y_val, asset, "y_reg")
        s_va = y_val[col_reg_va].dropna()
        col_reg_te = resolve_y_column(y_test, asset, "y_reg")
        s_te = y_test[col_reg_te].dropna()

    # Triple-barrier метки
    window_cfg = cfg.model.window
    horizon_cfg = cfg.model.horizon
    tb_tr = triple_barrier_labels(s_tr, VOL_WINDOW_TB, PT_MULT, SL_MULT, horizon_cfg)
    tb_va = triple_barrier_labels(s_va, VOL_WINDOW_TB, PT_MULT, SL_MULT, horizon_cfg)
    tb_te = triple_barrier_labels(s_te, VOL_WINDOW_TB, PT_MULT, SL_MULT, horizon_cfg)

    ix_tr = tb_tr.dropna().index.intersection(X_train[feat_cols].index)
    ix_va = tb_va.dropna().index.intersection(X_val[feat_cols].index)
    ix_te = tb_te.dropna().index.intersection(X_test[feat_cols].index)

    Xt = X_train[feat_cols].loc[ix_tr]
    Xv = X_val[feat_cols].loc[ix_va]
    Xte = X_test[feat_cols].loc[ix_te]
    yt_tb = tb_tr.loc[ix_tr].astype(np.int64).values
    yv_tb = tb_va.loc[ix_va].astype(np.int64).values
    yte_tb = tb_te.loc[ix_te].astype(np.int64).values

    # Нормализация ТОЛЬКО на train (без утечки)
    scaler = StandardScaler()
    Xt_s = scaler.fit_transform(Xt)
    Xv_s = scaler.transform(Xv)
    Xte_s = scaler.transform(Xte)

    # Forward-return
    fwd_tr_full = forward_return(s_tr, horizon_cfg)
    fwd_va_full = forward_return(s_va, horizon_cfg)
    fwd_te_full = forward_return(s_te, horizon_cfg)

    # Индексы окон
    idx_tr_w = Xt.index[window_cfg: window_cfg + max(0, len(Xt) - window_cfg)]
    idx_va_w = Xv.index[window_cfg: window_cfg + max(0, len(Xv) - window_cfg)]
    idx_te_w = Xte.index[window_cfg: window_cfg + max(0, len(Xte) - window_cfg)]

    fwd_tr = fwd_tr_full.reindex(idx_tr_w).fillna(0.0)
    fwd_va = fwd_va_full.reindex(idx_va_w).fillna(0.0)
    fwd_te = fwd_te_full.reindex(idx_te_w).fillna(0.0)

    # 1-барная forward-доходность (БЕЗ перекрытия) — для корректной торговой оценки.
    # forward_return(s, 1)[t] = s[t+1]; оценка pos*r1 не двойно-считает доходности
    # при удержании позиции (исправление бага overlapping forward-return).
    r1_tr = forward_return(s_tr, 1).reindex(idx_tr_w).fillna(0.0)
    r1_va = forward_return(s_va, 1).reindex(idx_va_w).fillna(0.0)
    r1_te = forward_return(s_te, 1).reindex(idx_te_w).fillna(0.0)

    return {
        "Xt_s": Xt_s, "Xv_s": Xv_s, "Xte_s": Xte_s,
        "yt_tb": yt_tb, "yv_tb": yv_tb, "yte_tb": yte_tb,
        "fwd_tr": fwd_tr, "fwd_va": fwd_va, "fwd_te": fwd_te,
        "r1_tr": r1_tr, "r1_va": r1_va, "r1_te": r1_te,
        "idx_tr_w": idx_tr_w, "idx_va_w": idx_va_w, "idx_te_w": idx_te_w,
        "n_features": n_features,
        "scaler": scaler,
        "synthetic": use_synthetic,
        "feat_cols": feat_cols,
        "asset": asset,
        "Xt": Xt, "Xv": Xv, "Xte": Xte,
        "yt_raw": yt_tb, "yv_raw": yv_tb, "yte_raw": yte_tb,
    }


def _log_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, artifact_dir: Path) -> Path:
    """Сохраняет матрицу ошибок в файл и возвращает путь."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["flat (0)", "long (1)"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Матрица ошибок (primary model, test)")
    plt.tight_layout()
    path = artifact_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _log_learning_curve(history: list, artifact_dir: Path) -> Path:
    """Строит кривую обучения primary-модели по эпохам и сохраняет в файл.

    history — список словарей {epoch, train_loss, val_net_sharpe}
    (собирается fit_classifier_sharpe через history_out). Рисуем две оси:
    train loss (левая) и val net-Sharpe (правая).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    if history:
        epochs = [h["epoch"] for h in history]
        losses = [h["train_loss"] for h in history]
        sharpes = [h["val_net_sharpe"] for h in history]

        ax.plot(epochs, losses, color="#A84B2F", linewidth=2, label="train loss")
        ax.set_ylabel("Train loss", color="#A84B2F")
        ax.tick_params(axis="y", labelcolor="#A84B2F")

        ax2 = ax.twinx()
        ax2.plot(epochs, sharpes, color="#20808D", linewidth=2, label="val net-Sharpe")
        ax2.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
        ax2.set_ylabel("Val net-Sharpe", color="#20808D")
        ax2.tick_params(axis="y", labelcolor="#20808D")

        # отмечаем лучшую эпоху (по которой выбран early stopping)
        best_i = int(np.argmax(sharpes))
        ax2.scatter([epochs[best_i]], [sharpes[best_i]], color="#20808D", zorder=5)
        ax2.annotate(
            f"best ep={epochs[best_i]}",
            (epochs[best_i], sharpes[best_i]),
            textcoords="offset points", xytext=(5, 5), fontsize=8,
        )
    else:
        ax.text(0.5, 0.5, "Нет истории обучения", ha="center", va="center")
    ax.set_xlabel("Эпоха")
    ax.set_title("Кривая обучения primary-модели (early stopping по val net-Sharpe)")
    plt.tight_layout()
    path = artifact_dir / "learning_curve.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_sample_predictions(
    model: MetaLabelingModel,
    Xte_s: np.ndarray,
    yte_tb: np.ndarray,
    fwd_te: pd.Series,
    idx_te_w,
    artifact_dir: Path,
    n_samples: int = 30,
) -> Path:
    """Сохраняет CSV с примерами предсказаний на тест-сете."""
    from torch.utils.data import DataLoader as _DataLoader
    from src.common import WindowDataset as _WD

    ds = _WD(Xte_s, yte_tb, model.window)
    loader = _DataLoader(ds, batch_size=256, shuffle=False)
    _, y_true, p1 = _predict_labels_probs(model.primary, loader, model.device)
    agg = window_aggregates(ds)
    meta_X = np.column_stack([agg, p1])
    meta_prob = model.secondary.predict_proba(meta_X)[:, 1]
    size = np.where(p1 >= 0.5, meta_prob, 0.0)
    size = np.where(meta_prob >= model.meta_threshold, size, 0.0)

    n = min(n_samples, len(y_true), len(idx_te_w))
    fwd_vals = fwd_te.values[:n] if len(fwd_te) >= n else np.zeros(n)

    df = pd.DataFrame({
        "индекс": list(idx_te_w[:n]),
        "метка (факт)": y_true[:n],
        "primary_prob": np.round(p1[:n], 4),
        "meta_prob": np.round(meta_prob[:n], 4),
        "позиция": np.round(size[:n], 4),
        "forward_return": np.round(fwd_vals, 6),
    })
    path = artifact_dir / "sample_predictions.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Главная функция: обучение + логирование в MLflow.

    Запускается через @hydra.main; параметры читаются из conf/*.yaml.
    Поддерживает override через командную строку.
    """
    # Инициализация seed
    set_seed(cfg.seed)
    print("=== CP7: Обучение Meta-Labeling ===")
    print(OmegaConf.to_yaml(cfg))

    # Настройка MLflow (с graceful-fallback на локальный sqlite, если сервер не поднят)
    tracking_uri = resolve_tracking_uri(
        cfg.mlflow.get("tracking_uri", "http://localhost:5000")
    )
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = cfg.mlflow.get("experiment_name", "crypto_meta_labeling_cp7")
    mlflow.set_experiment(experiment_name)

    # MinIO/S3 окружение для артефактов
    s3_endpoint = cfg.mlflow.get("s3_endpoint_url", None)
    if s3_endpoint:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", s3_endpoint)

    # Загрузка и подготовка данных
    data = _prepare_data(cfg)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")

        # Логирование параметров
        mlflow.log_params({
            "seed": cfg.seed,
            "hidden_size": cfg.model.hidden_size,
            "num_layers": cfg.model.num_layers,
            "dropout": cfg.model.dropout,
            "lr": cfg.model.lr,
            "batch_size": cfg.model.batch_size,
            "epochs": cfg.model.epochs,
            "patience": cfg.model.patience,
            "window": cfg.model.window,
            "horizon": cfg.model.horizon,
            "min_holding": cfg.model.min_holding,
            "cost_bps": cfg.model.cost_bps,
            "gbdt_n_estimators": cfg.model.gbdt.n_estimators,
            "gbdt_max_depth": cfg.model.gbdt.max_depth,
            "gbdt_learning_rate": cfg.model.gbdt.learning_rate,
            "asset": data["asset"],
            "n_features": data["n_features"],
            "synthetic": data["synthetic"],
            "n_train": len(data["Xt_s"]),
            "n_val": len(data["Xv_s"]),
            "n_test": len(data["Xte_s"]),
            "vol_window_tb": VOL_WINDOW_TB,
            "pt_mult": PT_MULT,
            "sl_mult": SL_MULT,
        })
        mlflow.set_tag("model_type", "meta_labeling")
        mlflow.set_tag("asset", data["asset"])

        # Создание модели
        gbdt_params = {
            "n_estimators": cfg.model.gbdt.n_estimators,
            "max_depth": cfg.model.gbdt.max_depth,
            "learning_rate": cfg.model.gbdt.learning_rate,
            "random_state": cfg.seed,
        }
        model = MetaLabelingModel(
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            lr=cfg.model.lr,
            batch_size=cfg.model.batch_size,
            epochs=cfg.model.epochs,
            patience=cfg.model.patience,
            window=cfg.model.window,
            horizon=cfg.model.horizon,
            min_holding=cfg.model.min_holding,
            cost_bps=cfg.model.cost_bps,
            gbdt_params=gbdt_params,
        )

        # Обучение
        metrics = model.fit(
            X_tr=data["Xt_s"],
            y_tr=data["yt_tb"],
            X_va=data["Xv_s"],
            y_va=data["yv_tb"],
            fwd_va=data["fwd_va"],
            idx_va_w=data["idx_va_w"],
            X_te=data["Xte_s"],
            y_te=data["yte_tb"],
            fwd_te=data["fwd_te"],
            idx_te_w=data["idx_te_w"],
            seed=cfg.seed,
        )

        # Логирование метрик
        mlflow.log_metrics({k: v for k, v in metrics.items() if not isinstance(v, float) or not np.isnan(v)})
        mlflow.log_metric("meta_threshold", model.meta_threshold)
        mlflow.log_metric("best_min_hold", float(model.best_min_hold))
        print("Метрики:", metrics)

        # Тег PRD на run
        mlflow.set_tag(cfg.mlflow.prd_tag, "true")

        # Артефакты
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Матрица ошибок (primary на test)
            try:
                from torch.utils.data import DataLoader as _DL
                ds_te = WindowDataset(data["Xte_s"], data["yte_tb"], model.window)
                ld_te = _DL(ds_te, batch_size=256, shuffle=False)
                pred_te, y_te_true, p1_te = _predict_labels_probs(
                    model.primary, ld_te, model.device
                )
                y_pred_bin = (p1_te >= 0.5).astype(int)
                cm_path = _log_confusion_matrix(y_te_true, y_pred_bin, tmp_path)
                mlflow.log_artifact(str(cm_path))
            except Exception as e:
                print(f"Предупреждение: не удалось создать confusion_matrix: {e}")

            # 2. Кривая обучения (реальная история по эпохам primary-модели)
            lc_path = _log_learning_curve(model.train_history, tmp_path)
            mlflow.log_artifact(str(lc_path))
            # Per-epoch метрики как step-серии (видно в MLflow UI)
            for h in model.train_history:
                mlflow.log_metric("epoch_train_loss", h["train_loss"], step=h["epoch"])
                mlflow.log_metric("epoch_val_net_sharpe", h["val_net_sharpe"], step=h["epoch"])

            # 3. Примеры предсказаний
            try:
                sp_path = _make_sample_predictions(
                    model,
                    data["Xte_s"],
                    data["yte_tb"],
                    data["fwd_te"],
                    data["idx_te_w"],
                    tmp_path,
                )
                mlflow.log_artifact(str(sp_path))
            except Exception as e:
                print(f"Предупреждение: не удалось создать sample_predictions: {e}")

            # 4. Сохранение компонентов модели для pyfunc
            model_save_dir = tmp_path / "model_artifacts"
            model.save(model_save_dir)

            # Artef paths для pyfunc
            artifacts = {
                "primary": str(model_save_dir / "primary.pt"),
                "secondary": str(model_save_dir / "secondary.pkl"),
                "meta": str(model_save_dir / "meta.pkl"),
            }

            # 5. Логирование pyfunc-модели в MLflow
            registered_model_name = cfg.mlflow.registered_model_name
            pyfunc_wrapper = MetaLabelingPyfunc()
            mlflow.pyfunc.log_model(
                artifact_path="meta_labeling_model",
                python_model=pyfunc_wrapper,
                artifacts=artifacts,
                registered_model_name=registered_model_name,
                pip_requirements=[
                    "mlflow==3.13.0",
                    "torch==2.12.0",
                    "scikit-learn==1.9.0",
                    "pandas==2.3.3",
                    "numpy==2.4.6",
                ],
            )
            print(f"Модель зарегистрирована: {registered_model_name}")

        # Выставляем тег PRD через MlflowClient на последнюю версию модели
        client = MlflowClient()
        prd_tag_value = cfg.mlflow.prd_tag

        try:
            # Получаем список версий модели
            versions = client.search_model_versions(f"name='{registered_model_name}'")
            if versions:
                # Берём самую новую версию
                latest_version = max(versions, key=lambda v: int(v.version))
                version_number = latest_version.version

                # Выставляем тег stage=PRD на версию модели
                client.set_model_version_tag(
                    name=registered_model_name,
                    version=version_number,
                    key="stage",
                    value=prd_tag_value,
                )

                # Создаём alias @PRD (MLflow >= 2.x)
                try:
                    client.set_registered_model_alias(
                        name=registered_model_name,
                        alias=prd_tag_value,
                        version=version_number,
                    )
                    print(f"Alias @{prd_tag_value} выставлен на версию {version_number}")
                except Exception as alias_err:
                    print(
                        f"Предупреждение: alias не поддерживается данной версией MLflow: {alias_err}"
                    )

                print(
                    f"Тег stage={prd_tag_value} выставлен на версию {version_number} "
                    f"модели '{registered_model_name}'"
                )
        except Exception as e:
            print(f"Предупреждение: не удалось выставить тег PRD: {e}")

    print("=== Обучение завершено ===")
    print(f"Experiment: {experiment_name}, run_id: {run_id}")


if __name__ == "__main__":
    main()
