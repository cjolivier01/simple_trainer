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
