# AGENTS.md

Notes for AI coding agents working in this repo.

## Pre-commit checks

Before committing, every change must pass:

```bash
ruff check .
ruff format --check .
```

If `ruff format --check` reports diffs, run `ruff format .` to apply them, then
re-run `ruff check .`. Don't commit code that fails either step.

## Noisy log lines to ignore

When training runs through xtrain / cuda-checkpoint / the snapshotd shim, you
will routinely see lines like

```
[CUDA][E] No CUDA context is current to the calling thread
[CUDA][E] Returning 201 (CUDA_ERROR_INVALID_CONTEXT) from cuCtxGetDevice_v2
```

These come from a side process (cuda-checkpoint / snapshot probe) querying
CUDA context state on a thread that doesn't own one. They're benign — not
caused by anything in this repo and not a sign of a real failure. When
triaging logs, filter them out:

```bash
... 2>&1 | grep -v "CUDA_ERROR_INVALID_CONTEXT\|cuCtxGetDevice_v2\|No CUDA context"
```
