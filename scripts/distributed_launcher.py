"""
This is a launcher script to set CUDA_VISIBLE_DEVICES on distributed scripts.
usage:
    python scripts/distributed_launcher.py <real_script> [args]

Mirrors /home/colivier/src/ai/scripts/distributed_launcher.py. Two minor
deltas from the ai-repo original: GPUS_PER_NODE is defined locally rather
than imported from infra.constants, and the ai-repo specific
``infra.core.init()`` + ``config_distributed_logger()`` setup calls are
omitted (those modules are not available here). Cluster-specific branches
(s11, TESLA_MULTIPLE_GPUS_PER_PROCESS) are preserved verbatim — they
simply don't fire unless the matching env vars are set.
"""

import ctypes
import logging
import os
import signal
import subprocess
import sys

# Local default for the per-node GPU count, used only when SLURM_NTASKS_PER_NODE
# is unset. Mirrors the role infra.constants.GPUS_PER_NODE plays in the ai repo.
GPUS_PER_NODE = 8

logger = logging.getLogger(__name__)


def initialize_visible_devices():
    torch_launcher_vars = ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    if any(v in os.environ for v in torch_launcher_vars):
        assert all(v in os.environ for v in torch_launcher_vars)
        local_rank = int(os.environ.get("LOCAL_RANK"))
    else:
        # slurm or single rank
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        tasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", GPUS_PER_NODE))
        allow_multiple_gpus_per_process = int(
            os.environ.get("TESLA_MULTIPLE_GPUS_PER_PROCESS", "0")
        )

        if len(cuda_visible_devices) > 1:
            if (
                tasks_per_node < len(cuda_visible_devices)
                and allow_multiple_gpus_per_process
            ):
                assert tasks_per_node == 4, (
                    "Only 2 GPUs per process are supported now. Ask ~ml-infra for support."
                )
                if os.environ.get("CLUSTER_ID") == "s11":
                    # We want to ensure that for every process first GPU device has local NIC.
                    # On s11 devices with local NIC are 0, 3, 4, 7 (instead of expected 0, 2, 4, 6), so
                    # we need to manually reshuffle visible devices.
                    # Chanding order in CUDA_VISIBLE_DEVICES will affect what physical devices are mapped to torch device 0 or 1.
                    if local_rank % 2 == 0:
                        os.environ["CUDA_VISIBLE_DEVICES"] = (
                            f"{2 * local_rank},{2 * local_rank + 1}"
                        )
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = (
                            f"{2 * local_rank + 1},{2 * local_rank}"
                        )
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = (
                        f"{2 * local_rank},{2 * local_rank + 1}"
                    )
            else:
                # more than 1 means --gpu-bind=none, manually bind them to tasks
                # CUDA_VISIBLE_DEVICES has to start from 0, which is guaranteed with slurm.conf ConstrainDevices=yes
                os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)


def _set_die_if_parent_dies():
    # PR_SET_PDEATHSIG: get SIGKILL when parent dies
    prctl = ctypes.CDLL("libc.so.6").prctl
    prctl(1, signal.SIGKILL)  # 1 = PR_SET_PDEATHSIG


if __name__ == "__main__":
    initialize_visible_devices()
    # use execv to replace our current process
    if sys.argv[1] == "debugpy":
        os.execv(sys.executable, ["python", "-m", "tools.vsdebugpy"] + sys.argv[2:])
    elif sys.argv[1] == "--vscode":
        _set_die_if_parent_dies()
        # Launching this way makes the subprocess debuggable with vscode
        sys.exit(subprocess.call([sys.executable] + sys.argv[2:]))
    else:
        os.execv(
            sys.executable,
            (["python"] if sys.argv[1] != "python" else []) + sys.argv[1:],
        )
