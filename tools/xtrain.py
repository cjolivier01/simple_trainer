#!/usr/bin/env python
"""@file xtrain.py
@brief Wrapper that runs bootstrap_train.py if it exists, otherwise train.py.

When --xtrain-fast is passed and no bootstrap exists, attempts to generate one
via ``snapshot generate`` before running it. Falls back to train.py if
generation fails.

DDP Support: When running under torchrun/SLURM with DDP env vars, uses manual
restore mode to preserve RANK/LOCAL_RANK/WORLD_SIZE across snapshot restore.

CONVENTION — xtrain-consumed args use the ``--xtrain-*`` prefix (or the
  shorthand alias ``--x-*``): every flag this wrapper interprets (and strips
  before forwarding to train.py) is namespaced this way to avoid collisions
  with train's own arg set. Flags can appear in any position. The full set
  of recognized flags is declared in ``XTRAIN_FLAGS`` below; add new ones
  there.
"""

import os
import runpy
import subprocess
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOTSTRAP = os.path.join(REPO_ROOT, "bootstrap_train.py")
TRAIN = os.path.join(REPO_ROOT, "tools", "train.py")

# Flags this wrapper consumes. Both "--xtrain-<name>" and "--x-<name>" forms
# are accepted (the --x- form is a shorthand alias). Each flag may appear
# anywhere in argv; recognized flags are stripped before forwarding the rest
# to the inner script. To add a flag, add a row here and read its value out
# of the dict returned by _parse_xtrain_flags.
#
# Spec: name -> (kind, default).
#   "bool": presence sets True (no "=value" form).
#   "int":  takes a value via "--flag=N" or "--flag N"; a bare "--flag"
#           (no value, or followed by something that doesn't parse as int)
#           is treated as "--flag 1" (i.e. enabled).
XTRAIN_FLAGS: dict[str, tuple[str, object]] = {
    "fast":          ("int",  1),
    "external-only": ("int",  0),
    "clean":         ("bool", False),
}
# Order matters: longer prefix first so "--xtrain-foo" doesn't get matched as
# "--x-" + "train-foo".
_XTRAIN_PREFIXES = ("--xtrain-", "--x-")


def _parse_xtrain_flags(argv):
    """Pull recognized --xtrain-*/--x-* flags out of argv (in any position).

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


def main():
    """@brief Entry point: dispatch to bootstrap or train based on snapshot state.

    @details
    Steps:
      1. Parse and strip --xtrain-* flags (--xtrain-fast=N, --xtrain-clean);
         everything else passes through to the inner script.
      2. If --xtrain-clean: unlink BOOTSTRAP and __pycache__/bootstrap_train.*,
         run ``python -m snapshot cache clean --yes``, then sys.exit.
      3. If --xtrain-fast and BOOTSTRAP is missing, attempt to (re)generate it
         via ``python -m snapshot generate`` — gated on
         ``snapshot.can_generate_bootstrap()`` (stable_modules data must exist).
      4. If BOOTSTRAP exists: run it via runpy. In a DDP context with snapshot
         images present (should_use_manual_restore()), use
         ``snapshot.runtime.restore_runtime()`` and inject preserved env vars
         (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_*, SLURM_*, TORCHELASTIC_*,
         TORCH_*, NCCL_*, GLOO_*, UCX_*, CUDA_*, OMP_*, MKL_*) — CRIU restore
         replaces the process and never returns.
      5. Else run train.py directly via runpy, and if the snapshot package is
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

    # Generate bootstrap if needed
    if fast and not os.path.exists(BOOTSTRAP):
        # Check if snapshot package can generate a useful bootstrap
        try:
            import snapshot
            if not snapshot.can_generate_bootstrap(REPO_ROOT):
                print(
                    "[xtrain] No stable_modules data available. ",
                    file=sys.stderr,
                )
            else:
                # Try to generate bootstrap
                generate_argv = [
                    sys.executable,
                    "-m",
                    "snapshot",
                    "generate",
                    "--script",
                    TRAIN,
                    "--output-script",
                    BOOTSTRAP,
                    "--sudo",
                ]
                if external_only:
                    generate_argv.append("--external-only")
                try:
                    subprocess.check_call(generate_argv)
                except subprocess.CalledProcessError as e:
                    print(
                        f"[xtrain] Warning: Bootstrap generation failed ({e}), "
                        "running without snapshot optimization",
                        file=sys.stderr,
                    )
        except ImportError:
            print(
                "[xtrain] Warning: snapshot package not available, "
                "running without snapshot optimization",
                file=sys.stderr,
            )

    # Run bootstrap or train
    if os.path.exists(BOOTSTRAP):
        if should_use_manual_restore():
            # DDP mode: use manual restore with env injection
            print(
                f"[xtrain] DDP mode detected, using manual restore with env preservation", file=sys.stderr, flush=True
            )

            try:
                from snapshot.runtime import restore_runtime
            except ImportError:
                print(
                    "[xtrain] WARNING: snapshot.runtime not available, falling back to normal bootstrap",
                    file=sys.stderr,
                    flush=True,
                )
                sys.argv = [BOOTSTRAP, *args]
                runpy.run_path(BOOTSTRAP, run_name="__main__")
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

            # Use rank-specific restore name for isolation
            rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))
            restore_name = f"rank-{rank}"

            print(
                f"[xtrain] Restoring rank {rank} with {len(restore_env_pairs)} env overrides",
                file=sys.stderr,
                flush=True,
            )

            # This function will restore from snapshot and never return
            # The restored process will have the injected env vars
            restore_runtime(
                runtime_dir=REPO_ROOT,
                restore_name=restore_name,
                restore_env=restore_env_pairs,
            )
            # Never reached - process is replaced by CRIU restore
        else:
            # Normal mode: single GPU or first-time run
            sys.argv = [BOOTSTRAP, *args]
            runpy.run_path(BOOTSTRAP, run_name="__main__")
    else:
        # No bootstrap - run train.py directly
        sys.argv = [TRAIN, *args]

        snapshot = None
        try:
            import snapshot
            if not snapshot.process_was_restored():
                snapshot.start_import_tracking()
        except:
            pass

        runpy.run_path(TRAIN, run_name="__main__")

        if snapshot is not None:
            snapshot.save_stable_modules(
                # only_non_repo=external_only,
                include_non_repo=True,
                max_age_days=3,
                script_path=TRAIN,
                bootstrap_script_path=BOOTSTRAP,
            )


if __name__ == "__main__":
    main()
