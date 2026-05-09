from __future__ import annotations

from pathlib import Path
from typing import Any

import wandb


class WandbTracker:
    def __init__(self, project: str, entity: str | None, mode: str = 'online') -> None:
        self.project = project
        self.entity = entity
        self.mode = mode
        self.run = None

    def start_run(self, run_name: str, config: dict[str, Any], tags: list[str] | None = None) -> None:
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            config=config,
            name=run_name,
            tags=tags or [],
            mode=self.mode,
            reinit=True,
        )

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        wandb.log(metrics, step=step)

    def log_artifact_file(self, path: str | Path, artifact_name: str, artifact_type: str = 'dataset') -> None:
        art = wandb.Artifact(artifact_name, type=artifact_type)
        art.add_file(str(path))
        wandb.log_artifact(art)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
