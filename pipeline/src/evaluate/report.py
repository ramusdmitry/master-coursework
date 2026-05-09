from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone


def save_run_report(report_dir: str | Path, name: str, payload: dict) -> Path:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    path = out_dir / f'{name}_{ts}.json'
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return path


def save_markdown_summary(report_dir: str | Path, filename: str, lines: list[str]) -> Path:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text('
'.join(lines) + '
', encoding='utf-8')
    return path
