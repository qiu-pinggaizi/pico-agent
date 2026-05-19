"""Tests for Training Knowledge Base."""

import tempfile
import time
from pathlib import Path

import pytest

from pico.training_kb import TrainingKB, TrainingRun


@pytest.fixture
def kb(tmp_path):
    """Create a temporary TrainingKB for testing."""
    db_path = tmp_path / "test_kb.db"
    return TrainingKB(db_path=db_path)


class TestTrainingRun:
    def test_default_values(self):
        run = TrainingRun()
        assert run.id == 0
        assert run.task == ""
        assert run.map50 == 0.0
        assert run.epochs == 0


class TestTrainingKB:
    def test_init_creates_db(self, kb):
        assert kb.db_path.exists()

    def test_record_and_retrieve(self, kb):
        run = TrainingRun(
            task="person_detection",
            dataset_name="coco128",
            model_arch="yolov8n",
            epochs=10,
            map50=0.85,
            map50_95=0.65,
            precision_=0.88,
            recall_=0.80,
            best_weights="/tmp/weights/best.pt",
        )
        row_id = kb.record_run(run)
        assert row_id > 0

        best = kb.get_best_run(task="person_detection")
        assert best is not None
        assert best.task == "person_detection"
        assert best.map50 == pytest.approx(0.85)
        assert best.map50_95 == pytest.approx(0.65)

    def test_get_best_run_across_tasks(self, kb):
        kb.record_run(TrainingRun(task="task_a", map50=0.7, map50_95=0.5))
        kb.record_run(TrainingRun(task="task_b", map50=0.9, map50_95=0.7))

        best = kb.get_best_run()
        assert best is not None
        assert best.task == "task_b"
        assert best.map50_95 == pytest.approx(0.7)

    def test_get_best_run_empty(self, kb):
        assert kb.get_best_run() is None

    def test_list_runs(self, kb):
        for i in range(5):
            kb.record_run(TrainingRun(task=f"task_{i}", map50=0.5 + i * 0.1))

        runs = kb.list_runs(limit=3)
        assert len(runs) == 3
        # Most recent first
        assert runs[0].task == "task_4"

    def test_list_runs_by_task(self, kb):
        kb.record_run(TrainingRun(task="a", map50=0.8))
        kb.record_run(TrainingRun(task="b", map50=0.9))
        kb.record_run(TrainingRun(task="a", map50=0.85))

        runs = kb.list_runs(task="a")
        assert len(runs) == 2
        assert all(r.task == "a" for r in runs)

    def test_get_task_summary(self, kb):
        kb.record_run(TrainingRun(task="det", map50=0.8, map50_95=0.6))
        kb.record_run(TrainingRun(task="det", map50=0.85, map50_95=0.65))
        kb.record_run(TrainingRun(task="seg", map50=0.7, map50_95=0.5))

        summary = kb.get_task_summary()
        assert len(summary) == 2
        # det should have higher best_map50_95
        assert summary[0]["task"] == "det"
        assert summary[0]["run_count"] == 2
        assert summary[0]["best_map50_95"] == pytest.approx(0.65)

    def test_summary_text_empty(self, kb):
        text = kb.summary_text()
        assert "No training runs" in text

    def test_summary_text_with_runs(self, kb):
        kb.record_run(TrainingRun(
            task="person_detection",
            model_arch="yolov8n",
            epochs=10,
            imgsz=640,
            map50=0.85,
            map50_95=0.65,
        ))
        text = kb.summary_text()
        assert "person_detection" in text
        assert "yolov8n" in text
        assert "0.850" in text
        assert "0.650" in text

    def test_custom_metric_best(self, kb):
        kb.record_run(TrainingRun(task="t", map50=0.9, map50_95=0.5, precision_=0.95))
        kb.record_run(TrainingRun(task="t", map50=0.7, map50_95=0.8, precision_=0.80))

        # Best by map50
        best_map50 = kb.get_best_run(task="t", metric="map50")
        assert best_map50.map50 == pytest.approx(0.9)

        # Best by precision
        best_prec = kb.get_best_run(task="t", metric="precision_")
        assert best_prec.precision_ == pytest.approx(0.95)

    def test_auto_timestamp(self, kb):
        run = TrainingRun(task="test")
        row_id = kb.record_run(run)
        runs = kb.list_runs(limit=1)
        assert runs[0].timestamp > 0
