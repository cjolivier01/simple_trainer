#!/usr/bin/env python
"""@file xtrain.py
@brief Wrapper that runs bootstrap_train.py if it exists, otherwise train.py.

When --xtrain-fast is passed and no bootstrap exists, attempts to generate one
via ``snapshot generate`` before running it. Falls back to train.py if
generation fails.

DDP Support: When running under torchrun/SLURM with DDP env vars, uses manual
restore mode to preserve RANK/LOCAL_RANK/WORLD_SIZE across snapshot restore.

CONVENTION — xtrain-consumed args use the ``--xtrain-*`` prefix (or the
  shorthand alias ``--xt-*``): every flag this wrapper interprets (and strips
  before forwarding to train.py) is namespaced this way to avoid collisions
  with train's own arg set. Flags can appear in any position. The full set
  of recognized flags is declared in ``XTRAIN_FLAGS`` below; add new ones
  there.
"""

import os
import runpy
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOTSTRAP = os.path.join(REPO_ROOT, "bootstrap_train.py")
TRAIN = os.path.join(REPO_ROOT, "tools", "train.py")

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
    "fast":          ("int",  1),
    "external-only": ("int",  1),
    "clean":         ("bool", False),
    "restore":       ("str",  ""),
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
        suffix = next((a[len(p):] for p in _XTRAIN_PREFIXES if a.startswith(p)), None)
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
                raise SystemExit(f"[xtrain] error: {a} expects an int, got {eq_value!r}")
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
    return all(v in os.environ for v in torch_vars) or all(v in os.environ for v in slurm_vars)


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
        from pathlib import Path
        images_path = Path(REPO_ROOT) / IMAGES_DIR
        return images_path.exists()
    except ImportError:
        # Fallback if snapshot module not available
        return False


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
        "--sudo",
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
) -> None:
    # The bootstrap parser accepts --restore-name on both `run` (default) and
    # `restore` subcommands. Inject before script_args so it binds to the
    # bootstrap parser rather than getting forwarded to the inner script.
    bootstrap_args: list[str] = []
    if restore_name:
        bootstrap_args += ["--restore-name", restore_name]
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
    subprocess.check_call([sys.executable, "-m", "snapshot", "pull", reference], cwd=REPO_ROOT)

    for candidate in _restore_reference_candidates(reference):
        try:
            return hydrate_or_switch(candidate)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(reference)


def _restore_requested_snapshot(
    reference: str,
    script_args: list[str],
    *,
    external_only: int,
) -> None:
    try:
        result = _hydrate_restore_reference(reference)
    except Exception as exc:
        raise SystemExit(f"[xtrain] error: failed to hydrate snapshot {reference!r}: {exc}") from exc
    _ensure_bootstrap_for_restore(external_only=external_only)
    print(
        f"[xtrain] restoring snapshot {result.snapshot_id[:12]} "
        f"from {result.generation_root}",
        file=sys.stderr,
    )
    _run_bootstrap(
        script_args,
        runtime_dir=result.generation_root,
        restore_name=_compute_restore_name(),
    )


def main():
    """@brief Entry point: dispatch to bootstrap or train based on snapshot state.

    @details
    Steps:
      1. Parse and strip --xtrain-* flags (--xtrain-fast=N, --xtrain-clean);
         everything else passes through to the inner script.
      2. If --xtrain-clean: unlink BOOTSTRAP and __pycache__/bootstrap_train.*,
         run ``python -m snapshot cache clean --yes``, then sys.exit.
      3. If --xtrain-restore/--xt-restore is set, pull/hydrate that snapshot,
         generate a bootstrap if needed, and restore that generation directly.
      4. If --xtrain-fast and BOOTSTRAP is missing, attempt to (re)generate it
         via ``python -m snapshot generate``. Missing stable_modules data is
         allowed so a hydrated autosnapshot can still restore in a fresh clone.
      5. If BOOTSTRAP exists: run it via runpy. In a DDP context with snapshot
         images present (should_use_manual_restore()), use
         ``snapshot.runtime.restore_runtime()`` and inject preserved env vars
         (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_*, SLURM_*, TORCHELASTIC_*,
         TORCH_*, NCCL_*, GLOO_*, UCX_*, CUDA_*, OMP_*, MKL_*) — CRIU restore
         replaces the process and never returns.
      6. Else run train.py directly via runpy, and if the snapshot package is
         importable, call ``start_import_tracking()`` before and
         ``save_stable_modules(max_age_days=3)`` after — this seeds the next
         --xtrain-fast rebuild.
    """
    # Parse xtrain-consumed flags (see XTRAIN_FLAGS at module top). Anything
    # not recognized is forwarded to the inner script unchanged.
    flags, args = _parse_xtrain_flags(sys.argv[1:])
    fast = flags["fast"]
    clean = flags["clean"]
    external_only = flags["external-only"]
    restore_ref = str(flags["restore"]).strip()

    # --xtrain-clean: wipe bootstrap artifacts, run `snapshot cache clean --yes`, exit
    if clean:
        bootstrap_targets = [Path(BOOTSTRAP)]
        pycache_dir = Path(REPO_ROOT) / "__pycache__"
        if pycache_dir.is_dir():
            bootstrap_targets.extend(pycache_dir.glob("bootstrap_train.*"))
        for target in bootstrap_targets:
            if target.exists():
                print(f"[xtrain] removing {target}", file=sys.stderr)
                target.unlink()
        print("[xtrain] running `python -m snapshot cache clean --yes`", file=sys.stderr)
        rc = subprocess.call([sys.executable, "-m", "snapshot", "cache", "clean", "--yes"])
        sys.exit(rc)

    if restore_ref:
        _restore_requested_snapshot(restore_ref, args, external_only=external_only)
        return

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

    # Generate bootstrap if needed
    if fast and not os.path.exists(BOOTSTRAP):
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
        if skip_bootstrap_for_direct_train and current_generation is None:
            print(
                "[xtrain] running train.py directly because no current autosnapshot "
                "is attached yet",
                file=sys.stderr,
            )
        elif not can_generate and current_generation is None:
            print(
                "[xtrain] stable_modules data is unavailable and no current "
                "autosnapshot is attached; running train.py directly",
                file=sys.stderr,
            )
        elif should_use_manual_restore():
            # DDP mode: use manual restore with env injection
            print(
                "[xtrain] DDP mode detected, using manual restore with env preservation", file=sys.stderr, flush=True
            )

            try:
                from snapshot.runtime import restore_runtime
            except ImportError:
                print(
                    "[xtrain] WARNING: snapshot.runtime not available, falling back to normal bootstrap",
                    file=sys.stderr,
                    flush=True,
                )
                _run_bootstrap(args, restore_name=_compute_restore_name())
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

            print(
                f"[xtrain] Restoring restore_name={restore_name} with "
                f"{len(restore_env_pairs)} env overrides",
                file=sys.stderr,
                flush=True,
            )

            restore_runtime(
                runtime_dir=REPO_ROOT,
                restore_name=restore_name,
                restore_env=restore_env_pairs,
            )
            _spawn_restore_state_cleanup(REPO_ROOT, restore_name)
        else:
            # Normal mode: single GPU or first-time run
            _run_bootstrap(args, restore_name=_compute_restore_name())
            return

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
        snapshot.save_stable_modules(
            only_non_repo=external_only,
            include_non_repo=True,
            max_age_days=3,
            script_path=TRAIN,
            bootstrap_script_path=BOOTSTRAP,
            repo_root=REPO_ROOT,
        )


if __name__ == "__main__":
    main()
