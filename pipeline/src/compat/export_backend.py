from __future__ import annotations

from pathlib import Path
import shutil
import json
from datetime import datetime, timezone


def export_backend_artifacts(source_model_path: str, source_scaler_path: str, backend_models_dir: str = 'models') -> dict:
    out_dir = Path(backend_models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_src = Path(source_model_path)
    scaler_src = Path(source_scaler_path)

    if not model_src.exists():
        raise FileNotFoundError(f'Не найден файл модели: {model_src}')
    if not scaler_src.exists():
        raise FileNotFoundError(f'Не найден файл scaler: {scaler_src}')

    model_dst = out_dir / 'model.pth'
    scaler_dst = out_dir / 'scaler.pkl'

    shutil.copy2(model_src, model_dst)
    shutil.copy2(scaler_src, scaler_dst)

    manifest = {
        'exported_at_utc': datetime.now(timezone.utc).isoformat(),
        'model': str(model_dst),
        'scaler': str(scaler_dst),
        'note': 'Формат совместим с app/ml_model.py (ожидаются models/model.pth и models/scaler.pkl).',
    }
    manifest_path = out_dir / 'pipeline_export_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest
