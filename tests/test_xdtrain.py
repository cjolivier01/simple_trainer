from __future__ import annotations

from pathlib import Path

import pytest

from tools import xdtrain


def test_xdtrain_launches_xtrain_ranks_with_cuda_device_maps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launches: list[tuple[list[str], dict[str, str]]] = []

    def fake_check_output(cmd: list[str], *, text: bool) -> str:
        assert text is True
        id_args = [item for item in cmd if item.startswith("--id=")]
        if not id_args:
            return "GPU-0\nGPU-1\nGPU-6\nGPU-7\n"
        gpu_id = id_args[0].split("=", 1)[1]
        return f"GPU-{gpu_id}\n"

    class FakeProcess:
        def __init__(self, cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
            assert cwd == xdtrain.REPO_ROOT
            launches.append((cmd, env))

        def wait(self) -> int:
            return 0

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            raise AssertionError("terminate should not be called for completed process")

    monkeypatch.setattr(xdtrain.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(xdtrain.subprocess, "Popen", FakeProcess)

    rc = xdtrain.main(
        [
            "--runtime-dir",
            str(tmp_path / "runtime"),
            "--nproc-per-node",
            "2",
            "--snapshot-cuda-visible-devices",
            "0",
            "--cuda-visible-devices",
            "6,7",
            "--",
            "--epochs",
            "1",
        ]
    )

    assert rc == 0
    assert len(launches) == 2
    assert launches[0][0][-2:] == ["--epochs", "1"]
    assert launches[0][1]["RANK"] == "0"
    assert launches[1][1]["RANK"] == "1"
    assert launches[0][1]["LOCAL_RANK"] == "0"
    assert launches[1][1]["LOCAL_RANK"] == "0"
    assert launches[0][1]["CUDA_VISIBLE_DEVICES"] == "6"
    assert launches[1][1]["CUDA_VISIBLE_DEVICES"] == "7"
    assert "NCCL_HOSTID" not in launches[0][1]
    assert "NCCL_HOSTID" not in launches[1][1]
    assert (
        launches[0][1]["XTRAIN_CUDA_DEVICE_MAP"]
        == "GPU-0=GPU-6,GPU-1=GPU-1,GPU-6=GPU-0,GPU-7=GPU-7"
    )
    assert (
        launches[1][1]["XTRAIN_CUDA_DEVICE_MAP"]
        == "GPU-0=GPU-7,GPU-1=GPU-1,GPU-6=GPU-6,GPU-7=GPU-0"
    )
    assert launches[0][1]["XTRAIN_RUNTIME_DIR"] == str((tmp_path / "runtime").resolve())
