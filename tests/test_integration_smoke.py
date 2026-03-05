from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION", "0") != "1", reason="set RUN_INTEGRATION=1 to run"
)
def test_train_smoke_cli_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    cmd = [
        "python",
        "-m",
        "videocap.train",
        "--config",
        "configs/smoke_128.yaml",
        "--run-name",
        "pytest_smoke",
    ]
    subprocess.run(cmd, cwd=root, check=True)
