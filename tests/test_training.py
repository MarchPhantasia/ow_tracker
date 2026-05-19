from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.head_tracker.training import recommend_epoch
from train.train import build_train_kwargs


def test_recommend_epoch_picks_highest_metric():
    root = Path("tmp") / "test-artifacts" / uuid4().hex
    root.mkdir(parents=True)
    results = root / "results.csv"
    results.write_text(
        "epoch, metrics/mAP50-95(B)\n"
        "1,0.10\n"
        "2,0.30\n"
        "3,0.20\n",
        encoding="utf-8",
    )

    rec = recommend_epoch(results)

    assert rec.epoch == 2
    assert rec.value == 0.30


def test_build_train_kwargs_can_disable_amp():
    class Args:
        imgsz = 1280
        epochs = 200
        batch = 32
        device = "0"
        workers = 8
        project = "runs/train"
        name = "ally_enemy_s"
        cache = "False"
        patience = 50
        use_all_data = False
        exist_ok = False
        amp = "false"

    kwargs = build_train_kwargs(Args(), "runs/datasets/ally_enemy_s/data.yaml")

    assert kwargs["amp"] is False
    assert kwargs["patience"] == 50
