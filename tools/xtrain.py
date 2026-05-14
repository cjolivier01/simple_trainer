#!/usr/bin/env python
"""@file xtrain.py
@brief Wrapper that hydrates a snapshot (skipping startup imports) and runs train.py.

Restore mode is selected by --xtrain-restore (or the env-var default):

  * Default (no --xtrain-restore on the command line): try to hydrate
    $XTRAIN_DEFAULT_RESTORE_REF (defaulting to ``ai/train_lenet``) from the
    local image cache, then from the OCI registry. If the snapshot isn't
    found in either, print a warning and run train.py directly with no
    autosnapshot machinery and no stable_modules save — i.e. xtrain becomes
    a no-op.
  * ``--xtrain-restore=auto``: run the legacy autosnapshot flow — generate
    a bootstrap from this checkout's imports, run it, save_stable_modules
    afterwards, and optionally tag/push the resulting snapshot via
    --xtrain-snapshot-tag / --xtrain-snapshot-push (those flags imply
    auto mode when --xtrain-restore is not also passed).
  * ``--xtrain-restore=<ref>``: hydrate that specific ref; SystemExit if
    hydrate fails. Use this when the caller really requires the snapshot.

``--xtrain-fast=0`` disables xtrain entirely and runs train.py raw.

DDP Support: When running under torchrun/SLURM with DDP env vars, uses manual
restore mode to preserve RANK/LOCAL_RANK/WORLD_SIZE across snapshot restore.

CONVENTION — xtrain-consumed args use the ``--xtrain-*`` prefix (or the
  shorthand alias ``--xt-*``): every flag this wrapper interprets (and strips
  before forwarding to train.py) is namespaced this way to avoid collisions
  with train's own arg set. Flags can appear in any position. The full set
  of recognized flags is declared in ``XTRAIN_FLAGS`` below; add new ones
  there.
"""

import json
import os
import runpy
import shutil
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BOOTSTRAP = os.path.join(REPO_ROOT, "bootstrap_train.py")
DEFAULT_TRAIN = os.path.join(REPO_ROOT, "tools", "train.py")
BOOTSTRAP = os.environ.get("XTRAIN_BOOTSTRAP", DEFAULT_BOOTSTRAP)
TRAIN = os.environ.get("XTRAIN_SCRIPT", DEFAULT_TRAIN)
XTRAIN_RUNTIME_DIR_ENV = "XTRAIN_RUNTIME_DIR"
XTRAIN_CUDA_DEVICE_MAP_ENV = "XTRAIN_CUDA_DEVICE_MAP"

# Default-mode restore (used when --xtrain-restore is not passed and no
# build/tag/push is requested): the snapshot ref to try hydrating before
# falling back to running train.py directly. Override per-shell via
# DEFAULT_RESTORE_REF_ENV_VAR. The literal string "auto" (in either the env
# var or via --xtrain-restore=auto) opts into the legacy autosnapshot flow.
DEFAULT_RESTORE_REF = "ai/train_lenet"
DEFAULT_RESTORE_REF_ENV_VAR = "XTRAIN_DEFAULT_RESTORE_REF"

# Flags this wrapper consumes. Both "--xtrain-<name>" and "--xt-<name>" forms
# are accepted (the --xt- form is a shorthand alias). Each flag may appear
# anywhere in argv; recognized flags are stripped before forwarding the rest
# to the inner script. To add a flag, add a row here and read its value out
# of the dict returned by _parse_xtrain_flags.
#
# Spec: name -> (kind, default).
#   "bool": presence sets True (no "=value" form).
#   "int":  takes a value via "--flag=N" or "--flag N"; a bare "--flag"
#           (no value, or followed by something that doesn't parse as int)
#           is treated as "--flag 1" (i.e. enabled).
#   "str":  takes a required value via "--flag=VALUE" or "--flag VALUE".
XTRAIN_FLAGS: dict[str, tuple[str, object]] = {
    "fast": ("int", 1),
    "external-only": ("int", 1),
    "clean": ("bool", False),
    "profile": ("int", 1),
    # Snapshot ref to hydrate before running train.py. Three modes:
    #   * unset (default): try $XTRAIN_DEFAULT_RESTORE_REF (or
    #     "ai/train_lenet"); warn + fall back to running train.py raw if the
    #     snapshot isn't found locally or in the OCI registry. No
    #     stable_modules saving.
    #   * "auto": legacy autosnapshot flow — generate a bootstrap from this
    #     checkout's imports, run it, save_stable_modules afterwards, and
    #     (with --xtrain-snapshot-tag / --xtrain-snapshot-push) tag/push the
    #     resulting snapshot.
    #   * any other value: hydrate that ref and SystemExit on failure.
    "restore": ("str", ""),
    # When enabled, after the inner script returns and save_stable_modules
    # has written the latest module list, run the bootstrap freshness checks
    # and build the autosnapshot if needed. The just-built (or already-
    # current) snapshot id is then captured so this run can tag/push it
    # without waiting for a "snapshot mode" pass on the next run.
    "build-snapshot": ("int", 0),
    # Optional OCI ref to tag the resulting snapshot id with (e.g.
    # ``ai/train_lenet:cuda-x86``). Implies the build-snapshot path.
    "snapshot-tag": ("str", ""),
    # When set, push the tagged snapshot to the OCI repo via
    # ``snapshot push <tag>``. Ignored when --xtrain-snapshot-tag is unset.
    "snapshot-push": ("int", 0),
}
# Order matters: longer prefix first so "--xtrain-foo" doesn't get matched as
# "--xt-" + "train-foo".
#
# Footgun: the "--xt-" shorthand is generic enough that future user-defined
# inner-script flags like "--xt-axis" would currently pass through (good), but
# would be silently captured by this wrapper if someone later added a matching
# entry to XTRAIN_FLAGS (e.g. an "axis" key). Keep XTRAIN_FLAGS keys distinctive
# enough to avoid plausible inner-script collisions.
_XTRAIN_PREFIXES = ("--xtrain-", "--xt-")


def _parse_xtrain_flags(argv):
    """Pull recognized --xtrain-*/--xt-* flags out of argv (in any position).

    Returns (values, forwarded). values has every key from XTRAIN_FLAGS;
    forwarded preserves the order of unrecognized args.
    """
    values = {n: d for n, (_, d) in XTRAIN_FLAGS.items()}
    forwarded = []
    i = 0
    while i < len(argv):
        a = argv[i]
        suffix = next((a[len(p) :] for p in _XTRAIN_PREFIXES if a.startswith(p)), None)
        if suffix is None:
            forwarded.append(a)
            i += 1
            continue
        name, eq, eq_value = suffix.partition("=")
        if name not in XTRAIN_FLAGS:
            # Recognized prefix but unknown name — pass through; inner script
            # (or the user) can decide how to handle it.
            forwarded.append(a)
            i += 1
            continue
        kind, _ = XTRAIN_FLAGS[name]
        if kind == "bool":
            values[name] = True
            i += 1
            continue
        if kind == "str":
            if eq:
                if not eq_value:
                    raise SystemExit(f"[xtrain] error: {a} expects a non-empty value")
                values[name] = eq_value
                i += 1
                continue
            if i + 1 >= len(argv):
                raise SystemExit(f"[xtrain] error: {a} expects a value")
            next_value = argv[i + 1]
            if not next_value or next_value.startswith("-"):
                raise SystemExit(f"[xtrain] error: {a} expects a restore reference")
            values[name] = next_value
            i += 2
            continue
        # int kind
        if eq:
            try:
                values[name] = int(eq_value)
            except ValueError:
                raise SystemExit(
                    f"[xtrain] error: {a} expects an int, got {eq_value!r}"
                )
            i += 1
            continue
        if i + 1 < len(argv):
            try:
                values[name] = int(argv[i + 1])
                i += 2
                continue
            except ValueError:
                pass
        # Bare flag: treat as enabled (=1).
        values[name] = 1
        i += 1
    return values, forwarded


def is_ddp_context():
    """@brief Detect whether the current process is running under DDP.

    @details
    Treats either of these env-var sets as a positive DDP signal:
      - torchrun:   LOCAL_RANK, RANK, WORLD_SIZE             (all three set)
      - SLURM srun: SLURM_PROCID, SLURM_LOCALID, SLURM_NTASKS (all three set)

    @return True iff either tuple is fully present in os.environ.
    """
    # torchrun sets these
    torch_vars = ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    # SLURM sets these
    slurm_vars = ("SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS")
    return all(v in os.environ for v in torch_vars) or all(
        v in os.environ for v in slurm_vars
    )


def should_use_manual_restore():
    """@brief Decide whether to use snapshot manual-restore (env-injecting) mode.

    @details
    Manual restore is required when all of the following hold:
      1. bootstrap_train.py exists on disk (snapshot is usable at all).
      2. We are in a DDP context (rank-specific env must survive the restore).
      3. The snapshot images directory exists under REPO_ROOT (something to
         restore from). Checked via snapshot.runtime.IMAGES_DIR.

    Falls back to False if the snapshot.runtime module cannot be imported.

    @return True iff all three conditions hold.
    """
    if not (os.path.exists(BOOTSTRAP) and is_ddp_context()):
        return False

    # Check if snapshot images exist using snapshot runtime API
    try:
        from snapshot.runtime import IMAGES_DIR

        runtime_dir = _manual_restore_runtime_dir()
        images_path = runtime_dir / IMAGES_DIR
        return images_path.exists()
    except ImportError:
        # Fallback if snapshot module not available
        return False


def _manual_restore_runtime_dir() -> Path:
    runtime_dir = os.environ.get(XTRAIN_RUNTIME_DIR_ENV, "").strip()
    if runtime_dir:
        return Path(runtime_dir).expanduser().resolve()
    return Path(REPO_ROOT)


def _ddp_local_rank() -> str | None:
    """Return LOCAL_RANK (or SLURM_LOCALID) as a string in DDP context, else None.

    Used to keep per-rank artifacts (autosnapshot runtime_dir, snapshot tag
    suffix) keyed off the same identity that ``scripts/distributed_launcher.py``
    uses to bind GPUs. Outside DDP, returns None so the caller can keep
    legacy single-snapshot behavior unchanged.
    """
    if not is_ddp_context():
        return None
    return os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID") or "0"


def _per_rank_autosnapshot_runtime_dir() -> Path | None:
    """Return a per-LOCAL_RANK runtime_dir for autosnapshot, or None outside DDP.

    Each DDP rank captures distinct in-memory state (its own model weights,
    NCCL handles, GPU bindings), so a single shared snapshot is wrong for
    DDP restore — the autosnapshot lease serializes the ranks and they all
    end up reusing rank-0's captured state. Giving each rank its own
    runtime_dir gives each its own lease, generation, and snapshot id.

    Falls back to None when the snapshot package isn't importable so the
    caller can pass `runtime_dir=None` and let build_autosnapshot_now use
    the manifest default.
    """
    local_rank = _ddp_local_rank()
    if local_rank is None:
        return None
    try:
        from snapshot.cache import default_runtime_dir
    except ImportError:
        return None
    base = default_runtime_dir(Path(REPO_ROOT))
    return base.with_name(f"{base.name}-rank-{local_rank}")


@contextmanager
def _ddp_autosnapshot_update_lock():
    """Serialize shared stable-module/bootstrap updates across DDP ranks."""
    if not is_ddp_context():
        yield
        return
    import fcntl

    try:
        from snapshot.cache import repo_cache_dir

        lock_dir = repo_cache_dir(Path(REPO_ROOT))
    except ImportError:
        lock_dir = Path(REPO_ROOT)
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "xtrain-autosnapshot-update.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _apply_rank_suffix_to_tag(reference: str) -> str:
    """Append ``-rank-<LOCAL_RANK>`` to the tag part of a snapshot ref under DDP.

    No-op when:
      * not in a DDP context (single-rank run);
      * the ref has no ``:tag`` part (bare repo, e.g. the default-restore
        ``ai/train_lenet`` — those resolve via oci_ref_with_default_tag,
        and per-rank doesn't make sense without an explicit tag to extend);
      * the existing tag already contains ``-rank-`` (caller is being
        explicit, don't double-suffix).

    Symmetric across build (``--xtrain-snapshot-tag``) and restore
    (``--xtrain-restore``) so ``--create`` and ``--restore`` round-trip
    in DDP.
    """
    local_rank = _ddp_local_rank()
    if local_rank is None:
        return reference
    repo, sep, tag = reference.rpartition(":")
    if not sep or not repo or "/" in tag:
        # No tag part (or the ":" was inside a path, e.g. "host:5000/repo").
        return reference
    if "-rank-" in tag:
        return reference
    return f"{repo}:{tag}-rank-{local_rank}"


def _rank_aware_restore_reference(reference: str) -> str:
    """Return the effective restore ref for this process/rank.

    DDP create publishes ``<ref>:<tag>-rank-N``. When default restore gets a
    bare repository ref, apply ``SNAPSHOT_OCI_TAG`` first so each rank looks up
    the matching tag it created.
    """
    if _ddp_local_rank() is None or _looks_like_snapshot_id(reference):
        return _apply_rank_suffix_to_tag(reference)
    effective_ref = _apply_rank_suffix_to_tag(reference)
    if effective_ref != reference:
        return effective_ref
    try:
        from snapshot.oci_repository import oci_ref_with_default_tag

        tagged = oci_ref_with_default_tag(reference)
    except Exception:
        return reference
    return _apply_rank_suffix_to_tag(tagged)


def _snapshot_can_generate_bootstrap(snapshot_module) -> bool:
    try:
        return bool(snapshot_module.can_generate_bootstrap(REPO_ROOT))
    except Exception:
        return False


def _current_autosnapshot_generation_root() -> Path | None:
    """Return the repo's current autosnapshot generation, if one is attached."""
    try:
        from snapshot.autosnapshot_state import (
            autosnapshot_current_generation_root,
            autosnapshot_paths,
        )
        from snapshot.cache import default_runtime_dir

        runtime_dir = default_runtime_dir(Path(REPO_ROOT))
        generation_root = autosnapshot_current_generation_root(
            autosnapshot_paths(runtime_dir)
        )
    except Exception:
        return None
    if generation_root is None or not generation_root.exists():
        return None
    return generation_root


def _query_local_index_to_uuid() -> dict[int, str] | None:
    """Return host {GPU index -> UUID} via nvidia-smi, or None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
    ):
        return None
    mapping: dict[int, str] = {}
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        if parts[1]:
            mapping[idx] = parts[1]
    return mapping or None


def _resolve_cuda_visible_to_uuids(
    cuda_visible_devices: str,
    index_to_uuid: dict[int, str],
) -> list[str] | None:
    """Resolve CUDA_VISIBLE_DEVICES tokens to physical UUIDs, in order.

    Tokens are either numeric indices (looked up in *index_to_uuid*) or
    literal "GPU-…" / "MIG-…" UUIDs (passed through). Empty / "-1" mean
    "no GPUs" → []. Returns None if any token is unrecognized; the caller
    should then skip auto-deriving rather than build a partial map.
    """
    raw = cuda_visible_devices.strip()
    if raw in ("", "-1"):
        return []
    uuids: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.startswith(("GPU-", "MIG-")):
            uuids.append(token)
            continue
        try:
            idx = int(token)
        except ValueError:
            return None
        uuid_value = index_to_uuid.get(idx)
        if not uuid_value:
            return None
        uuids.append(uuid_value)
    return uuids


def _compute_cuda_device_map(runtime_dir: Path | str | None) -> str:
    """Build the cuda-checkpoint --device-map for restoring on this rank.

    Returns ``"oldUuid=newUuid[,oldUuid=newUuid...]"`` when the snapshot's
    ``cuda-bound-uuids.json`` sidecar exists and the local CUDA_VISIBLE_DEVICES
    resolves to the same number of physical UUIDs. Returns ``""`` when:
      * ``XTRAIN_CUDA_DEVICE_MAP`` is set to one of {"none","off","0"};
      * the sidecar is missing or marks ``cuda_initialized=false``;
      * nvidia-smi is unavailable;
      * counts mismatch between the snapshot's UUIDs and the rank's visible
        UUIDs (cuda-checkpoint requires "all checkpointed devices" mapped,
        so a partial map would be rejected anyway).

    ``XTRAIN_CUDA_DEVICE_MAP=<pairs>`` (any other value) is honored as an
    explicit override and returned verbatim.
    """
    override = os.environ.get(XTRAIN_CUDA_DEVICE_MAP_ENV, "").strip()
    if override.lower() in ("none", "off", "0"):
        return ""
    if override:
        return override
    if runtime_dir is None:
        return ""
    sidecar = Path(runtime_dir) / "cuda-bound-uuids.json"
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return ""
    if not payload.get("cuda_initialized"):
        return ""
    snapshot_uuids = payload.get("uuids") or []
    if not snapshot_uuids:
        return ""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is None:
        return ""
    index_to_uuid = _query_local_index_to_uuid()
    if index_to_uuid is None:
        return ""
    new_uuids = _resolve_cuda_visible_to_uuids(cuda_visible, index_to_uuid)
    if new_uuids is None or len(new_uuids) != len(snapshot_uuids):
        if new_uuids is not None:
            print(
                f"[xtrain] cuda_device_map auto-derive skipped: snapshot has "
                f"{len(snapshot_uuids)} captured GPU(s) but CUDA_VISIBLE_DEVICES "
                f"resolves to {len(new_uuids)}; pass XTRAIN_CUDA_DEVICE_MAP=<pairs> "
                f"to override",
                file=sys.stderr,
            )
        return ""
    pairs = [
        f"{old}={new}" for old, new in zip(snapshot_uuids, new_uuids) if old != new
    ]
    return ",".join(pairs)


def _compute_restore_name() -> str:
    """Build a unique-but-rank-prefixed per-restore state name.

    Pattern: ``rank-<R>[-<SLURM_JOB_ID>]-<12hex>``. The rank prefix lets a
    human eyeball which directory belongs to which rank; SLURM_JOB_ID, when
    set, groups names from the same Slurm allocation; the 12-hex suffix
    makes every restore globally unique so concurrent runs (and re-runs in
    the same job) never collide on the shared restore-state dir.
    """
    rank = os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "0"
    parts = [f"rank-{rank}"]
    jobid = os.environ.get("SLURM_JOB_ID")
    if jobid:
        parts.append(jobid)
    parts.append(uuid.uuid4().hex[:12])
    return "-".join(parts)


_SNAPSHOT_RESTORE_FAILURE_TYPES: tuple[type[BaseException], ...] = (
    RuntimeError,  # SnapshotdError + generic snapshot/CRIU errors
    subprocess.CalledProcessError,  # CRIU / snapshotd CLI invocations
    OSError,  # missing binaries, broken sockets, etc.
    ImportError,  # snapshot package / extension version mismatch
)


def _is_snapshot_restore_failure(exc: BaseException) -> bool:
    """True if *exc* looks like a CRIU/snapshotd-level restore failure.

    Used to decide whether to fall back to running train.py directly
    (snapshot-machinery problem — degrade gracefully) versus re-raising
    (exception came from user code that ran post-restore — propagate).

    The categorization is heuristic: we can't perfectly distinguish a
    restore-time RuntimeError from a user-code RuntimeError. The bias is
    toward graceful fallback so version skew between the conda env's
    snapshot/snapshotd/CRIU and the snapshot artifact never crashes a
    training run that could otherwise have proceeded without restore.
    """
    return isinstance(exc, _SNAPSHOT_RESTORE_FAILURE_TYPES)


def _run_train_directly(args: list[str]) -> None:
    """Run train.py via runpy with no snapshot machinery."""
    sys.argv = [TRAIN, *args]
    runpy.run_path(TRAIN, run_name="__main__")


def _spawn_restore_state_cleanup(
    runtime_dir: Path | str | None,
    restore_name: str,
) -> None:
    """Background-rmtree the per-restore state dir after a successful run.

    Unique restore names accumulate one-per-launch under
    ``<runtime_dir>/restores/<name>/`` and the snapshot package only auto-
    clears them on a same-name reuse (which never happens with our UUID
    suffix). We delete on success in a non-daemon thread so xtrain returns
    immediately but Python still waits for the rmtree at interpreter exit.
    """
    if not restore_name:
        return
    if runtime_dir is None:
        runtime_dir = _current_autosnapshot_generation_root()
    if runtime_dir is None:
        print(
            f"[xtrain] cleanup skipped: no runtime_dir resolved for "
            f"restore_name={restore_name}; <unknown>/restores/{restore_name}/ "
            f"may persist on disk",
            file=sys.stderr,
        )
        return
    state_dir = Path(runtime_dir) / "restores" / restore_name

    def _cleanup() -> None:
        try:
            shutil.rmtree(state_dir, ignore_errors=True)
        except Exception as exc:
            print(
                f"[xtrain] restore-state cleanup warning ({state_dir}): {exc}",
                file=sys.stderr,
            )

    threading.Thread(
        target=_cleanup,
        name="xtrain-restore-cleanup",
        daemon=False,
    ).start()


def _restored_worker_exit_code(
    runtime_dir: Path | str, restore_name: str
) -> int | None:
    status_path = (
        Path(runtime_dir) / "restores" / restore_name / "logs" / "worker.status.json"
    )
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        return None
    exit_code = payload.get("exit_code")
    return exit_code if isinstance(exit_code, int) else None


def _ddp_restore_env_pairs() -> list[str]:
    """Collect restore-time distributed env vars that must override the snapshot."""
    restore_env = {}
    individual_vars = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "ROLE_RANK",
        "LOCAL_WORLD_SIZE",
        "ROLE_WORLD_SIZE",
        "PYTHON_EXEC",
        "DDP_BACKEND",
    ]
    for var in individual_vars:
        if var in os.environ:
            restore_env[var] = os.environ[var]

    env_prefixes = [
        "SLURM_",
        "TORCHELASTIC_",
        "NCCL_",
        "GLOO_",
        "UCX_",
        "TORCH_NCCL_",
        "TORCH_DISTRIBUTED_",
        "TORCH_CUDNN_",
        "PYTORCH_CUDA_",
        "CUDA_",
        "OMP_",
        "MKL_",
    ]
    for var, value in os.environ.items():
        if any(var.startswith(prefix) for prefix in env_prefixes):
            restore_env[var] = value
    return [f"{key}={value}" for key, value in restore_env.items()]


def _run_manual_ddp_restore(script_args: list[str], runtime_dir: Path | str) -> bool:
    """Restore a hydrated DDP snapshot with current torchrun env injection."""
    if not is_ddp_context():
        return False
    try:
        from snapshot.runtime import restore_runtime
    except ImportError:
        return False

    runtime_path = Path(runtime_dir)
    restore_name = _compute_restore_name()
    restore_env_pairs = _ddp_restore_env_pairs()
    cuda_device_map = _compute_cuda_device_map(runtime_path)
    print(
        f"[xtrain] DDP restore: restore_name={restore_name} with "
        f"{len(restore_env_pairs)} env overrides"
        + (f", cuda_device_map={cuda_device_map}" if cuda_device_map else ""),
        file=sys.stderr,
        flush=True,
    )
    restore_runtime(
        runtime_dir=runtime_path,
        restore_name=restore_name,
        restore_env=restore_env_pairs,
        script_args=script_args,
        cuda_device_map=cuda_device_map,
    )
    exit_code = _restored_worker_exit_code(runtime_path, restore_name)
    if exit_code == 0:
        _spawn_restore_state_cleanup(runtime_path, restore_name)
        return True
    state_dir = runtime_path / "restores" / restore_name
    if exit_code is None:
        print(
            f"[xtrain] restored worker exit status is unknown; "
            f"leaving restore logs in {state_dir}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(
        f"[xtrain] restored worker exited with status {exit_code}; "
        f"leaving restore logs in {state_dir}",
        file=sys.stderr,
    )
    raise SystemExit(exit_code)


def _generate_bootstrap(
    *,
    external_only: int,
    allow_missing_stable_modules: bool,
) -> bool:
    generate_argv = [
        sys.executable,
        "-m",
        "snapshot",
        "generate",
        "--repo-root",
        REPO_ROOT,
        "--script",
        TRAIN,
        "--output-script",
        BOOTSTRAP,
        "--snapshotd",
    ]
    if external_only:
        generate_argv.append("--external-only")
    if allow_missing_stable_modules:
        generate_argv.append("--allow-missing-stable-modules")
    try:
        subprocess.check_call(generate_argv)
    except subprocess.CalledProcessError as e:
        print(
            f"[xtrain] Warning: Bootstrap generation failed ({e}), "
            "running without snapshot optimization",
            file=sys.stderr,
        )
        return False
    return True


def _ensure_bootstrap_for_restore(*, external_only: int) -> None:
    print(
        "[xtrain] Generating restore bootstrap without requiring stable_modules data",
        file=sys.stderr,
    )
    if not _generate_bootstrap(
        external_only=external_only,
        allow_missing_stable_modules=True,
    ):
        raise SystemExit("[xtrain] error: unable to generate bootstrap for restore")


def _run_bootstrap(
    script_args: list[str],
    *,
    runtime_dir: Path | None = None,
    restore_name: str = "",
    cuda_device_map: str = "",
) -> None:
    # The bootstrap parser accepts --restore-name / --cuda-device-map on both
    # `run` (default) and `restore` subcommands. Inject before script_args so
    # they bind to the bootstrap parser rather than getting forwarded to the
    # inner script.
    bootstrap_args: list[str] = []
    if restore_name:
        bootstrap_args += ["--restore-name", restore_name]
    if cuda_device_map:
        bootstrap_args += ["--cuda-device-map", cuda_device_map]
    # Capture the cleanup target *before* the bootstrap runs. In `run` mode the
    # bootstrap may publish a new autosnapshot selection marker mid-run, so a
    # post-bootstrap query of _current_autosnapshot_generation_root() could
    # resolve a different generation than the one we restored into and leak
    # the real per-restore state dir.
    cleanup_runtime_dir = runtime_dir or _current_autosnapshot_generation_root()
    if runtime_dir is None:
        sys.argv = [BOOTSTRAP, *bootstrap_args, *script_args]
    else:
        sys.argv = [
            BOOTSTRAP,
            "restore",
            "--runtime-dir",
            str(runtime_dir),
            *bootstrap_args,
            *script_args,
        ]
    runpy.run_path(BOOTSTRAP, run_name="__main__")
    # Successful return: rmtree the per-restore state dir in the background.
    # On exception we leave it behind so restore_log/ + worker_log are
    # available for post-mortem.
    _spawn_restore_state_cleanup(cleanup_runtime_dir, restore_name)


def _looks_like_snapshot_id(value: str) -> bool:
    text = value.strip().lower()
    return len(text) >= 8 and all(ch in "0123456789abcdef" for ch in text)


def _restore_reference_candidates(reference: str) -> list[str]:
    candidates = [reference]
    if not _looks_like_snapshot_id(reference):
        try:
            from snapshot.oci_repository import oci_ref_with_default_tag

            tagged = oci_ref_with_default_tag(reference)
        except Exception:
            tagged = reference
        if tagged not in candidates:
            candidates.insert(0, tagged)
    return candidates


def _hydrate_restore_reference(reference: str):
    from snapshot.oci_repository import (
        hydrate_local_snapshot,
        unhydrate_repo_autosnapshot,
    )

    def hydrate_or_switch(ref: str):
        try:
            return hydrate_local_snapshot(ref, repo_root=REPO_ROOT)
        except RuntimeError as exc:
            if "already points at complete generation" not in str(exc):
                raise
            unhydrate_repo_autosnapshot(repo_root=REPO_ROOT)
            return hydrate_local_snapshot(ref, repo_root=REPO_ROOT)

    last_error: Exception | None = None
    for candidate in _restore_reference_candidates(reference):
        try:
            return hydrate_or_switch(candidate)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            last_error = exc

    print(f"[xtrain] pulling snapshot {reference}", file=sys.stderr)
    subprocess.check_call(
        [sys.executable, "-m", "snapshot", "pull", reference], cwd=REPO_ROOT
    )

    for candidate in _restore_reference_candidates(reference):
        try:
            return hydrate_or_switch(candidate)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(reference)


def _remove_all_local_snapshots() -> None:
    """Remove every local OCI snapshot image (and its runtime backing dir).

    ``snapshot image ls --json`` enumerates both kinds of local images:
      - "oci" entries: artifacts under ~/.cache/snapshots/oci/
      - "runtime-cache" entries: standalone runtime checkpoints not backed
        by an OCI image
    Each entry is removed via ``snapshot image rm``, which deletes the
    backing OCI files (when present) plus the runtime dir and unlinks any
    local tags pointing at it. Snapshot ids are deduped so we don't try to
    remove the same image twice when multiple tags share an id.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "snapshot", "image", "ls", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"[xtrain] warning: failed to list local snapshots ({exc}); "
            f"skipping local image cleanup",
            file=sys.stderr,
        )
        return
    try:
        entries = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        print(
            f"[xtrain] warning: could not parse `snapshot image ls --json` output: {exc}",
            file=sys.stderr,
        )
        return
    refs: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        snapshot_id = str(entry.get("snapshot_id", "")).strip()
        if snapshot_id:
            ref = snapshot_id
        else:
            # stale-tag entry without a resolvable snapshot id — fall back to tag
            repo = str(entry.get("repository", "")).strip()
            tag = str(entry.get("tag", "")).strip()
            if not repo or repo == "<none>" or not tag or tag == "<none>":
                continue
            ref = f"{repo}:{tag}"
        if ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
    if not refs:
        print("[xtrain] no local snapshots to remove", file=sys.stderr)
        return
    print(
        f"[xtrain] removing {len(refs)} local snapshot image(s): "
        f"{' '.join(r[:12] if len(r) > 24 else r for r in refs)}",
        file=sys.stderr,
    )
    rc = subprocess.call([sys.executable, "-m", "snapshot", "image", "rm", *refs])
    if rc != 0:
        print(
            f"[xtrain] warning: `snapshot image rm` exited {rc}",
            file=sys.stderr,
        )


def _restore_requested_snapshot(
    reference: str,
    script_args: list[str],
    *,
    external_only: int,
) -> None:
    effective_ref = _rank_aware_restore_reference(reference)
    if effective_ref != reference:
        print(
            f"[xtrain] DDP: restoring per-rank snapshot {effective_ref!r} "
            f"(suffixed from {reference!r})",
            file=sys.stderr,
        )
    try:
        result = _hydrate_restore_reference(effective_ref)
    except Exception as exc:
        raise SystemExit(
            f"[xtrain] error: failed to hydrate snapshot {effective_ref!r}: {exc}"
        ) from exc
    with _ddp_autosnapshot_update_lock():
        _ensure_bootstrap_for_restore(external_only=external_only)
    print(
        f"[xtrain] restoring snapshot {result.snapshot_id[:12]} "
        f"from {result.generation_root}",
        file=sys.stderr,
    )
    try:
        if _run_manual_ddp_restore(script_args, result.generation_root):
            return
    except Exception as exc:
        if not _is_snapshot_restore_failure(exc):
            raise
        print(
            f"[xtrain] warning: manual DDP restore of {reference!r} failed "
            f"({type(exc).__name__}: {exc}); falling back to running train.py "
            f"directly without snapshot machinery",
            file=sys.stderr,
        )
        _run_train_directly(script_args)
        return
    try:
        _run_bootstrap(
            script_args,
            runtime_dir=result.generation_root,
            restore_name=_compute_restore_name(),
            cuda_device_map=_compute_cuda_device_map(result.generation_root),
        )
    except Exception as exc:
        if not _is_snapshot_restore_failure(exc):
            raise
        print(
            f"[xtrain] warning: snapshot restore of {reference!r} failed "
            f"({type(exc).__name__}: {exc}); falling back to running train.py "
            f"directly without snapshot machinery (likely a snapshot/snapshotd/"
            f"CRIU version mismatch with this conda env)",
            file=sys.stderr,
        )
        _run_train_directly(script_args)


def _default_restore_ref() -> str:
    """Return the snapshot ref used in default mode (no --xtrain-restore).

    Honors $XTRAIN_DEFAULT_RESTORE_REF as an override; falls back to
    DEFAULT_RESTORE_REF. Whitespace is stripped.
    """
    return (
        os.environ.get(DEFAULT_RESTORE_REF_ENV_VAR, "").strip() or DEFAULT_RESTORE_REF
    )


def _try_default_restore(
    reference: str,
    script_args: list[str],
    *,
    external_only: int,
) -> bool:
    """Default-mode hydrate-then-restore that NEVER calls SystemExit.

    Returns True if hydrate + bootstrap-restore took over (typically the
    process is then replaced by CRIU restore and we never return; if the
    bootstrap path runs the inner script directly, it returns normally
    after the script completes). Returns False after printing a warning
    when the snapshot can't be hydrated locally or pulled from the OCI
    registry — the caller should then run train.py directly with no
    snapshot machinery.
    """
    effective_ref = _rank_aware_restore_reference(reference)
    print(
        f"[xtrain] default mode: trying snapshot {effective_ref!r}",
        file=sys.stderr,
    )
    try:
        result = _hydrate_restore_reference(effective_ref)
    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(
            f"[xtrain] warning: snapshot {reference!r} not found locally or in "
            f"OCI registry ({exc}); running train.py directly. Pass "
            f"--xtrain-restore={reference} to make this an error, "
            f"--xtrain-restore=auto to build a local autosnapshot from "
            f"imports, or --xtrain-fast=0 to silence this attempt.",
            file=sys.stderr,
        )
        return False
    try:
        with _ddp_autosnapshot_update_lock():
            _ensure_bootstrap_for_restore(external_only=external_only)
    except SystemExit as exc:
        print(
            f"[xtrain] warning: failed to prepare bootstrap for snapshot "
            f"{reference!r} ({exc}); running train.py directly",
            file=sys.stderr,
        )
        return False
    print(
        f"[xtrain] restoring snapshot {result.snapshot_id[:12]} "
        f"from {result.generation_root}",
        file=sys.stderr,
    )
    try:
        if _run_manual_ddp_restore(script_args, result.generation_root):
            return True
    except Exception as exc:
        if not _is_snapshot_restore_failure(exc):
            raise
        print(
            f"[xtrain] warning: manual DDP restore of {reference!r} failed "
            f"({type(exc).__name__}: {exc}); running train.py directly",
            file=sys.stderr,
        )
        return False
    try:
        _run_bootstrap(
            script_args,
            runtime_dir=result.generation_root,
            restore_name=_compute_restore_name(),
            cuda_device_map=_compute_cuda_device_map(result.generation_root),
        )
    except Exception as exc:
        if not _is_snapshot_restore_failure(exc):
            raise
        print(
            f"[xtrain] warning: snapshot restore of {reference!r} failed "
            f"({type(exc).__name__}: {exc}); falling back to running train.py "
            f"directly (likely a snapshot/snapshotd/CRIU version mismatch with "
            f"this conda env)",
            file=sys.stderr,
        )
        return False
    return True


def _maybe_build_and_publish_snapshot(*, tag_ref: str, push: bool) -> None:
    """Build the autosnapshot (if needed) and optionally tag/push it.

    Triggered by --xtrain-build-snapshot (or implicitly by
    --xtrain-snapshot-tag / --xtrain-snapshot-push). Runs after
    save_stable_modules so the next run's freshness check would pass
    without rebuilding — but instead of waiting for that next run, we
    materialize the snapshot here.

    The build result includes the exact autosnapshot generation root, so DDP
    per-rank runtimes can be tagged without relying on global cache discovery.
    Push still runs as a subprocess so the snapshot CLI handles output and
    exit codes uniformly.
    """
    try:
        from snapshot import build_autosnapshot_now_result
    except ImportError as exc:
        print(
            f"[xtrain] --xtrain-build-snapshot needs snapshot.build_autosnapshot_now_result; "
            f"upgrade the `snapshot` package (got: {exc})",
            file=sys.stderr,
        )
        return
    if not os.path.isfile(BOOTSTRAP):
        print(
            f"[xtrain] --xtrain-build-snapshot: bootstrap script not present at "
            f"{BOOTSTRAP}; nothing to build (re-run with --xtrain-fast first)",
            file=sys.stderr,
        )
        return
    runtime_dir_override = _per_rank_autosnapshot_runtime_dir()
    if runtime_dir_override is not None:
        print(
            f"[xtrain] DDP: building per-rank autosnapshot in {runtime_dir_override}",
            file=sys.stderr,
        )
    print(
        f"[xtrain] building autosnapshot (or reusing current) via {BOOTSTRAP}",
        file=sys.stderr,
    )
    build_result = build_autosnapshot_now_result(
        bootstrap_script=BOOTSTRAP,
        runtime_dir=runtime_dir_override,
    )
    snapshot_id = build_result.snapshot_id
    print(f"[xtrain] autosnapshot ready: snapshot_id={snapshot_id}", file=sys.stderr)
    if not tag_ref:
        if push:
            print(
                "[xtrain] --xtrain-snapshot-push ignored: no --xtrain-snapshot-tag given",
                file=sys.stderr,
            )
        return
    effective_tag = _apply_rank_suffix_to_tag(tag_ref)
    if effective_tag != tag_ref:
        print(
            f"[xtrain] DDP: tagging as {effective_tag!r} (suffixed from {tag_ref!r})",
            file=sys.stderr,
        )
    try:
        from snapshot.oci_repository import tag_runtime_cache_snapshot
    except ImportError as exc:
        raise SystemExit(
            "[xtrain] error: --xtrain-snapshot-tag needs "
            "snapshot.oci_repository.tag_runtime_cache_snapshot; upgrade the "
            f"`snapshot` package (got: {exc})"
        ) from exc
    print(
        f"[xtrain] tagging {snapshot_id[:12]} from {build_result.generation_root} "
        f"as {effective_tag}",
        file=sys.stderr,
    )
    tag_runtime_cache_snapshot(
        effective_tag,
        runtime_dir=build_result.generation_root,
        snapshot_id=snapshot_id,
        ref=effective_tag,
    )
    if push:
        print(f"[xtrain] pushing {effective_tag}", file=sys.stderr)
        subprocess.check_call(
            [sys.executable, "-m", "snapshot", "push", effective_tag],
            cwd=REPO_ROOT,
        )


def main():
    """@brief Entry point: dispatch to bootstrap or train based on snapshot state.

    @details
    Steps:
      1. Parse and strip --xtrain-* flags. Everything else passes through to
         the inner script.
      2. ``--xtrain-clean`` → wipe BOOTSTRAP + bootstrap bytecode, remove all
         local OCI/runtime snapshot images, run ``snapshot cache clean
         --yes``, sys.exit.
      3. ``--xtrain-fast=0`` → run train.py raw (no snapshot machinery).
      4. Resolve the restore mode:
           a. ``--xtrain-restore=<ref>`` (non-auto) → hydrate that ref or
              SystemExit.
           b. No ``--xtrain-restore`` and no build/tag/push → default mode:
              try $XTRAIN_DEFAULT_RESTORE_REF (or "ai/train_lenet"); on
              miss, warn and run train.py directly with NO autosnapshot
              machinery and NO save_stable_modules — i.e. xtrain becomes a
              no-op.
           c. ``--xtrain-restore=auto`` (or implied by build/tag/push):
              fall through to the legacy autosnapshot flow.
      5. Auto-mode flow: try to adopt a sibling checkout's
         stable_modules.json; (re)generate BOOTSTRAP if missing. If
         BOOTSTRAP exists, run it via runpy. In a DDP context with snapshot
         images present (should_use_manual_restore()), use
         ``snapshot.runtime.restore_runtime()`` and inject preserved env vars
         (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_*, SLURM_*, TORCHELASTIC_*,
         TORCH_*, NCCL_*, GLOO_*, UCX_*, CUDA_*, OMP_*, MKL_*) — CRIU restore
         replaces the process and never returns. Otherwise run train.py
         directly via runpy, ``start_import_tracking()`` before and
         ``save_stable_modules()`` after to seed the next rebuild, and
         optionally tag/push the resulting snapshot.
    """
    # Parse xtrain-consumed flags (see XTRAIN_FLAGS at module top). Anything
    # not recognized is forwarded to the inner script unchanged.
    flags, args = _parse_xtrain_flags(sys.argv[1:])
    fast = flags["fast"]
    clean = flags["clean"]
    external_only = flags["external-only"]
    profile: int = flags["profile"]
    restore_ref = str(flags["restore"]).strip()
    restore_ref_explicit = bool(restore_ref)
    snapshot_tag_ref = str(flags["snapshot-tag"]).strip()
    snapshot_push = bool(flags["snapshot-push"])
    # --xtrain-snapshot-tag implies --xtrain-build-snapshot — the user
    # always needs the snapshot to exist locally before tagging it.
    build_snapshot = (
        bool(flags["build-snapshot"]) or bool(snapshot_tag_ref) or snapshot_push
    )

    # --xtrain-clean: wipe bootstrap artifacts, all local OCI snapshots, then
    # `snapshot cache clean --yes`, exit.
    if clean:
        bootstrap_targets = [Path(BOOTSTRAP)]
        pycache_dir = Path(REPO_ROOT) / "__pycache__"
        if pycache_dir.is_dir():
            bootstrap_targets.extend(pycache_dir.glob("bootstrap_train.*"))
        for target in bootstrap_targets:
            if target.exists():
                print(f"[xtrain] removing {target}", file=sys.stderr)
                target.unlink()
        _remove_all_local_snapshots()
        print(
            "[xtrain] running `python -m snapshot cache clean --yes`", file=sys.stderr
        )
        rc = subprocess.call(
            [sys.executable, "-m", "snapshot", "cache", "clean", "--yes"]
        )
        sys.exit(rc)

    # --xtrain-fast=0: bypass xtrain entirely.
    if not fast:
        _run_train_directly(args)
        return

    # Build/tag/push imply auto mode (the legacy flow is what produces a
    # snapshot to publish). Auto wins over the implicit default ref.
    if build_snapshot and not restore_ref_explicit:
        restore_ref = "auto"

    # Resolve the dispatch:
    #   * "auto" (explicit, env-var, or build/tag/push-implied) → fall
    #     through to the legacy autosnapshot flow below.
    #   * Explicit non-auto ref → hydrate or SystemExit. No fallback.
    #   * Otherwise (no --xtrain-restore, no build/tag/push) → default mode:
    #     try $XTRAIN_DEFAULT_RESTORE_REF (or "ai/train_lenet"); on miss,
    #     warn and run train.py directly with NO snapshot machinery and NO
    #     stable_modules save. Env var value of "auto" opts into the legacy
    #     flow without command-line flags.
    if restore_ref != "auto":
        if restore_ref_explicit:
            _restore_requested_snapshot(restore_ref, args, external_only=external_only)
            return
        default_ref = _default_restore_ref()
        if default_ref != "auto":
            if _try_default_restore(default_ref, args, external_only=external_only):
                return
            _run_train_directly(args)
            return
        restore_ref = "auto"

    # ---- Auto mode: legacy autosnapshot generate/restore/save flow.
    assert restore_ref == "auto"

    if profile:
        os.environ["SNAPSHOT_PROFILE"] = "1"

    snapshot = None
    try:
        import snapshot
    except ImportError:
        snapshot = None
    can_generate = (
        _snapshot_can_generate_bootstrap(snapshot) if snapshot is not None else False
    )
    if not can_generate and snapshot is not None:
        # Fresh checkouts have no captured stable_modules.json yet, but a
        # sibling checkout of the same repo may already have one whose git
        # HEAD + dirty .py state matches ours. Adopting it lets the very
        # first run build (and reuse) the autosnapshot without an extra
        # seed pass. getattr keeps this safe with older snapshot installs.
        materialize_sibling = getattr(
            snapshot, "materialize_sibling_stable_modules", None
        )
        if materialize_sibling is not None:
            try:
                materialized = materialize_sibling(REPO_ROOT)
            except Exception:
                materialized = None
            if materialized is not None:
                print(
                    f"[xtrain] adopted stable_modules from sibling checkout: {materialized}",
                    file=sys.stderr,
                )
                can_generate = _snapshot_can_generate_bootstrap(snapshot)
    skip_bootstrap_for_direct_train = False

    # Generate bootstrap if needed (fast=0 already short-circuited above).
    if not os.path.exists(BOOTSTRAP):
        # Check if snapshot package can generate a useful bootstrap
        if snapshot is not None:
            if not can_generate:
                print(
                    "[xtrain] No stable_modules data available; "
                    "generating bootstrap with --allow-missing-stable-modules, "
                    "then running train.py directly to seed stable_modules.",
                    file=sys.stderr,
                )
                skip_bootstrap_for_direct_train = True
            _generate_bootstrap(
                external_only=external_only,
                allow_missing_stable_modules=not can_generate,
            )
        else:
            print(
                "[xtrain] Warning: snapshot package not available, "
                "running without snapshot optimization",
                file=sys.stderr,
            )

    # Run bootstrap or train
    if os.path.exists(BOOTSTRAP):
        current_generation = _current_autosnapshot_generation_root()
        manual_restore_available = should_use_manual_restore()
        bootstrap_restore_blocked = False
        if manual_restore_available:
            # DDP mode: use manual restore with env injection
            print(
                "[xtrain] DDP mode detected, using manual restore with env preservation",
                file=sys.stderr,
                flush=True,
            )
        elif skip_bootstrap_for_direct_train and current_generation is None:
            print(
                "[xtrain] running train.py directly because no current autosnapshot "
                "is attached yet",
                file=sys.stderr,
            )
            bootstrap_restore_blocked = True
        elif not can_generate and current_generation is None:
            print(
                "[xtrain] stable_modules data is unavailable and no current "
                "autosnapshot is attached; running train.py directly",
                file=sys.stderr,
            )
            bootstrap_restore_blocked = True
        if manual_restore_available:
            try:
                from snapshot.runtime import restore_runtime
            except ImportError:
                print(
                    "[xtrain] WARNING: snapshot.runtime not available, falling back to normal bootstrap",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    _run_bootstrap(
                        args,
                        restore_name=_compute_restore_name(),
                        cuda_device_map=_compute_cuda_device_map(
                            _manual_restore_runtime_dir()
                        ),
                    )
                except Exception as exc:
                    if not _is_snapshot_restore_failure(exc):
                        raise
                    print(
                        f"[xtrain] warning: bootstrap fallback failed "
                        f"({type(exc).__name__}: {exc}); running train.py directly",
                        file=sys.stderr,
                    )
                    _run_train_directly(args)
                sys.exit(0)

            # Collect DDP env vars to preserve across restore
            # Use a dict to avoid duplicates
            restore_env = {}

            # Specific individual torchrun vars (always check these first)
            individual_vars = [
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "GROUP_RANK",
                "ROLE_RANK",
                "LOCAL_WORLD_SIZE",
                "ROLE_WORLD_SIZE",
                "PYTHON_EXEC",
                "DDP_BACKEND",
            ]
            for var in individual_vars:
                if var in os.environ:
                    restore_env[var] = os.environ[var]

            # All env vars with specific prefixes
            env_prefixes = [
                "SLURM_",  # SLURM job management
                "TORCHELASTIC_",  # torchrun/elastic agent
                "NCCL_",  # NCCL backend config (communication)
                "GLOO_",  # Gloo backend config
                "UCX_",  # UCX backend config
                "TORCH_NCCL_",  # PyTorch NCCL settings
                "TORCH_DISTRIBUTED_",  # PyTorch distributed settings
                "TORCH_CUDNN_",  # cuDNN settings
                "PYTORCH_CUDA_",  # PyTorch CUDA allocator settings
                "CUDA_",  # CUDA runtime settings (includes CUDA_VISIBLE_DEVICES)
                "OMP_",  # OpenMP threading (OMP_NUM_THREADS)
                "MKL_",  # MKL threading (MKL_NUM_THREADS)
            ]
            for var, value in os.environ.items():
                if any(var.startswith(prefix) for prefix in env_prefixes):
                    restore_env[var] = value

            # Convert to list of KEY=VALUE pairs
            restore_env_pairs = [f"{k}={v}" for k, v in restore_env.items()]

            restore_name = _compute_restore_name()
            runtime_dir = _manual_restore_runtime_dir()
            cuda_device_map = _compute_cuda_device_map(runtime_dir)

            print(
                f"[xtrain] Restoring restore_name={restore_name} with "
                f"{len(restore_env_pairs)} env overrides"
                + (f", cuda_device_map={cuda_device_map}" if cuda_device_map else ""),
                file=sys.stderr,
                flush=True,
            )

            try:
                restore_runtime(
                    runtime_dir=runtime_dir,
                    restore_name=restore_name,
                    restore_env=restore_env_pairs,
                    script_args=args,
                    cuda_device_map=cuda_device_map,
                )
            except Exception as exc:
                if not _is_snapshot_restore_failure(exc):
                    raise
                print(
                    f"[xtrain] warning: manual DDP restore failed "
                    f"({type(exc).__name__}: {exc}); falling back to running "
                    f"train.py directly (likely a snapshot/snapshotd/CRIU "
                    f"version mismatch with this conda env)",
                    file=sys.stderr,
                )
                # Fall through to the train-direct path below.
            else:
                exit_code = _restored_worker_exit_code(runtime_dir, restore_name)
                if exit_code == 0:
                    _spawn_restore_state_cleanup(runtime_dir, restore_name)
                    return
                state_dir = runtime_dir / "restores" / restore_name
                if exit_code is None:
                    print(
                        f"[xtrain] restored worker exit status is unknown; "
                        f"leaving restore logs in {state_dir}",
                        file=sys.stderr,
                    )
                    raise SystemExit(1)
                print(
                    f"[xtrain] restored worker exited with status {exit_code}; "
                    f"leaving restore logs in {state_dir}",
                    file=sys.stderr,
                )
                raise SystemExit(exit_code)
        elif not bootstrap_restore_blocked:
            # Normal mode: single GPU or first-time run
            try:
                _run_bootstrap(
                    args,
                    restore_name=_compute_restore_name(),
                    cuda_device_map=_compute_cuda_device_map(
                        _current_autosnapshot_generation_root()
                    ),
                )
                return
            except Exception as exc:
                if not _is_snapshot_restore_failure(exc):
                    raise
                print(
                    f"[xtrain] warning: bootstrap restore failed "
                    f"({type(exc).__name__}: {exc}); falling back to train.py "
                    f"directly (likely a snapshot/snapshotd/CRIU version "
                    f"mismatch with this conda env)",
                    file=sys.stderr,
                )
                # Fall through to the train-direct path below.

    # No usable bootstrap restore path yet - run train.py directly.
    sys.argv = [TRAIN, *args]

    if snapshot is not None:
        try:
            if not snapshot.process_was_restored():
                snapshot.start_import_tracking()
        except Exception:
            pass

    runpy.run_path(TRAIN, run_name="__main__")

    if snapshot is not None:
        with _ddp_autosnapshot_update_lock():
            snapshot.save_stable_modules(
                only_non_repo=external_only,
                include_non_repo=True,
                max_age_days=3,
                script_path=TRAIN,
                bootstrap_script_path=BOOTSTRAP,
                repo_root=REPO_ROOT,
            )
        # Build/tag/push runs per-rank (per-rank runtime_dir and per-rank
        # snapshot tag), so it doesn't need the shared-state lock — keeping
        # it inside serializes the slow CRIU dump + OCI push across ranks.
        if build_snapshot:
            _maybe_build_and_publish_snapshot(
                tag_ref=snapshot_tag_ref,
                push=snapshot_push,
            )


if __name__ == "__main__":
    main()
