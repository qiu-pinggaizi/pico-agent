"""Training Knowledge Base — structured training memory for Pico Agent.

Inspired by OpenHuman's Memory Trees: each training run is canonicalized
into structured records with hyperparameters, metrics, dataset info,
and file paths. Stored in SQLite for efficient querying.

This lets the agent answer questions like:
  - "What was my best mAP so far?"
  - "Which dataset performed better?"
  - "What hyperparameters worked best for person detection?"
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pico.config import CONFIG_DIR

logger = logging.getLogger(__name__)

KB_DB = CONFIG_DIR / "training_kb.db"


@dataclass
class TrainingRun:
    """A single training run record."""
    id: int = 0
    timestamp: float = 0.0
    task: str = ""              # e.g. "person_detection", "pet_detection"
    dataset_path: str = ""
    dataset_name: str = ""
    model_arch: str = ""        # e.g. "yolov8n", "yolov8s"
    epochs: int = 0
    imgsz: int = 640
    batch_size: int = 16
    lr0: float = 0.01
    map50: float = 0.0
    map50_95: float = 0.0
    precision_: float = 0.0
    recall_: float = 0.0
    best_weights: str = ""
    results_dir: str = ""
    notes: str = ""
    extra: str = ""             # JSON blob for additional data


class TrainingKB:
    """SQLite-backed training knowledge base.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else KB_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create the training_runs table if it doesn't exist."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    task TEXT DEFAULT '',
                    dataset_path TEXT DEFAULT '',
                    dataset_name TEXT DEFAULT '',
                    model_arch TEXT DEFAULT '',
                    epochs INTEGER DEFAULT 0,
                    imgsz INTEGER DEFAULT 640,
                    batch_size INTEGER DEFAULT 16,
                    lr0 REAL DEFAULT 0.01,
                    map50 REAL DEFAULT 0.0,
                    map50_95 REAL DEFAULT 0.0,
                    precision_ REAL DEFAULT 0.0,
                    recall_ REAL DEFAULT 0.0,
                    best_weights TEXT DEFAULT '',
                    results_dir TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    extra TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task ON training_runs(task)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_map50_95 ON training_runs(map50_95)
            """)
            conn.commit()

    def record_run(self, run: TrainingRun) -> int:
        """Record a training run. Returns the row ID."""
        if run.timestamp == 0.0:
            run.timestamp = time.time()
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                """INSERT INTO training_runs
                   (timestamp, task, dataset_path, dataset_name, model_arch,
                    epochs, imgsz, batch_size, lr0,
                    map50, map50_95, precision_, recall_,
                    best_weights, results_dir, notes, extra)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run.timestamp, run.task, run.dataset_path, run.dataset_name,
                 run.model_arch, run.epochs, run.imgsz, run.batch_size, run.lr0,
                 run.map50, run.map50_95, run.precision_, run.recall_,
                 run.best_weights, run.results_dir, run.notes, run.extra),
            )
            conn.commit()
            row_id = cursor.lastrowid
            logger.info("Recorded training run #%d: %s %s mAP50=%.3f",
                        row_id, run.task, run.model_arch, run.map50)
            return row_id or 0

    def get_best_run(self, task: str = "", metric: str = "map50_95") -> TrainingRun | None:
        """Get the best training run by metric, optionally filtered by task."""
        valid_metrics = {"map50", "map50_95", "precision_", "recall_"}
        if metric not in valid_metrics:
            metric = "map50_95"
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if task:
                row = conn.execute(
                    f"SELECT * FROM training_runs WHERE task = ? ORDER BY {metric} DESC LIMIT 1",
                    (task,),
                ).fetchone()
            else:
                row = conn.execute(
                    f"SELECT * FROM training_runs ORDER BY {metric} DESC LIMIT 1"
                ).fetchone()
            if row:
                return self._row_to_run(row)
        return None

    def list_runs(self, task: str = "", limit: int = 20) -> list[TrainingRun]:
        """List recent training runs, optionally filtered by task."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if task:
                rows = conn.execute(
                    "SELECT * FROM training_runs WHERE task = ? ORDER BY timestamp DESC LIMIT ?",
                    (task, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM training_runs ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._row_to_run(row) for row in rows]

    def get_task_summary(self) -> list[dict[str, Any]]:
        """Get a summary of all tasks with run counts and best metrics."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT task,
                       COUNT(*) as run_count,
                       MAX(map50_95) as best_map50_95,
                       MAX(map50) as best_map50,
                       AVG(epochs) as avg_epochs
                FROM training_runs
                GROUP BY task
                ORDER BY best_map50_95 DESC
            """).fetchall()
            return [dict(row) for row in rows]

    def _row_to_run(self, row: sqlite3.Row) -> TrainingRun:
        """Convert a database row to a TrainingRun dataclass."""
        return TrainingRun(
            id=row["id"],
            timestamp=row["timestamp"],
            task=row["task"],
            dataset_path=row["dataset_path"],
            dataset_name=row["dataset_name"],
            model_arch=row["model_arch"],
            epochs=row["epochs"],
            imgsz=row["imgsz"],
            batch_size=row["batch_size"],
            lr0=row["lr0"],
            map50=row["map50"],
            map50_95=row["map50_95"],
            precision_=row["precision_"],
            recall_=row["recall_"],
            best_weights=row["best_weights"],
            results_dir=row["results_dir"],
            notes=row["notes"],
            extra=row["extra"],
        )

    def summary_text(self) -> str:
        """Return a human-readable summary of all training runs."""
        tasks = self.get_task_summary()
        if not tasks:
            return "No training runs recorded yet."

        lines = ["# Training Knowledge Base\n"]
        for t in tasks:
            lines.append(
                f"**{t['task'] or 'untitled'}**: {t['run_count']} runs | "
                f"Best mAP50-95: {t['best_map50_95']:.3f} | "
                f"Best mAP50: {t['best_map50']:.3f}"
            )

        # Recent runs
        recent = self.list_runs(limit=5)
        if recent:
            lines.append("\n## Recent Runs\n")
            for r in recent:
                lines.append(
                    f"- **{r.task or 'untitled'}** ({r.model_arch}) — "
                    f"mAP50={r.map50:.3f}, mAP50-95={r.map50_95:.3f} | "
                    f"{r.epochs} epochs, imgsz={r.imgsz}"
                )

        return "\n".join(lines)
