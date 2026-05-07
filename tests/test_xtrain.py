from pathlib import Path
import types
import sys

import pytest

from tools import xtrain


def test_parse_restore_flags_strip_xtrain_args() -> None:
    values, forwarded = xtrain._parse_xtrain_flags(
        ["--xtrain-restore=repo/image:tag", "--epochs", "1"]
    )

    assert values["restore"] == "repo/image:tag"
    assert forwarded == ["--epochs", "1"]


def test_parse_restore_shorthand_requires_value() -> None:
    with pytest.raises(SystemExit, match="expects a value"):
        xtrain._parse_xtrain_flags(["--xt-restore"])


def test_parse_restore_shorthand_rejects_option_as_value() -> None:
    with pytest.raises(SystemExit, match="expects a restore reference"):
        xtrain._parse_xtrain_flags(["--xt-restore", "--epochs", "1"])


def test_restore_flag_hydrates_and_restores_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []

    result = types.SimpleNamespace(
        snapshot_id="abcdef1234567890",
        generation_root=tmp_path / "generation",
    )
    monkeypatch.setattr(
        xtrain,
        "_hydrate_restore_reference",
        lambda reference: calls.append(("hydrate", reference)) or result,
    )
    monkeypatch.setattr(
        xtrain,
        "_ensure_bootstrap_for_restore",
        lambda **kwargs: calls.append(("ensure", kwargs)),
    )
    monkeypatch.setattr(
        xtrain,
        "_run_bootstrap",
        lambda args, **kwargs: calls.append(("run", (list(args), kwargs))),
    )
    monkeypatch.setattr(sys, "argv", ["xtrain.py", "--xt-restore", "repo/image", "--epochs", "1"])

    xtrain.main()

    assert calls == [
        ("hydrate", "repo/image"),
        ("ensure", {"external_only": 1}),
        ("run", (["--epochs", "1"], {"runtime_dir": result.generation_root})),
    ]


def test_restore_regenerates_existing_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bootstrap = tmp_path / "bootstrap_train.py"
    bootstrap.write_text("# stale\n", encoding="utf-8")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xtrain, "BOOTSTRAP", str(bootstrap))
    monkeypatch.setattr(
        xtrain,
        "_generate_bootstrap",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    xtrain._ensure_bootstrap_for_restore(external_only=1)

    assert calls == [{"external_only": 1, "allow_missing_stable_modules": True}]


def test_missing_stable_modules_generates_bootstrap_but_runs_train_directly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bootstrap = tmp_path / "bootstrap_train.py"
    train = tmp_path / "tools" / "train.py"
    train.parent.mkdir(parents=True)
    train.write_text("print('train')\n", encoding="utf-8")

    calls: list[tuple[str, object]] = []
    fake_snapshot = types.SimpleNamespace(
        can_generate_bootstrap=lambda _repo_root: False,
        process_was_restored=lambda: False,
        start_import_tracking=lambda: calls.append(("track", None)),
        save_stable_modules=lambda **kwargs: calls.append(("save", kwargs)),
    )

    def generate_bootstrap(**kwargs: object) -> bool:
        calls.append(("generate", kwargs))
        bootstrap.write_text("# generated\n", encoding="utf-8")
        return True

    monkeypatch.setitem(sys.modules, "snapshot", fake_snapshot)
    monkeypatch.setattr(xtrain, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(xtrain, "BOOTSTRAP", str(bootstrap))
    monkeypatch.setattr(xtrain, "TRAIN", str(train))
    monkeypatch.setattr(xtrain, "_generate_bootstrap", generate_bootstrap)
    monkeypatch.setattr(xtrain, "_current_autosnapshot_generation_root", lambda: None)
    monkeypatch.setattr(xtrain.runpy, "run_path", lambda path, run_name: calls.append(("runpy", path)))
    monkeypatch.setattr(sys, "argv", ["xtrain.py", "--max-steps", "1"])

    xtrain.main()

    assert calls[0] == (
        "generate",
        {"external_only": 1, "allow_missing_stable_modules": True},
    )
    assert ("track", None) in calls
    assert ("runpy", str(train)) in calls
    save_calls = [payload for name, payload in calls if name == "save"]
    assert len(save_calls) == 1
    assert save_calls[0]["only_non_repo"] == 1
    assert save_calls[0]["bootstrap_script_path"] == str(bootstrap)
    assert save_calls[0]["repo_root"] == str(tmp_path)


def test_missing_stable_modules_with_current_generation_uses_bootstrap_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bootstrap = tmp_path / "bootstrap_train.py"
    train = tmp_path / "tools" / "train.py"
    generation = tmp_path / "generation"
    train.parent.mkdir(parents=True)
    bootstrap.write_text("# generated\n", encoding="utf-8")
    train.write_text("print('train')\n", encoding="utf-8")
    generation.mkdir()

    calls: list[tuple[list[str], dict[str, object]]] = []
    fake_snapshot = types.SimpleNamespace(
        can_generate_bootstrap=lambda _repo_root: False,
    )

    monkeypatch.setitem(sys.modules, "snapshot", fake_snapshot)
    monkeypatch.setattr(xtrain, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(xtrain, "BOOTSTRAP", str(bootstrap))
    monkeypatch.setattr(xtrain, "TRAIN", str(train))
    monkeypatch.setattr(xtrain, "_current_autosnapshot_generation_root", lambda: generation)
    monkeypatch.setattr(xtrain, "should_use_manual_restore", lambda: False)
    monkeypatch.setattr(
        xtrain,
        "_run_bootstrap",
        lambda args, **kwargs: calls.append((list(args), kwargs)),
    )
    monkeypatch.setattr(sys, "argv", ["xtrain.py", "--epochs", "1"])

    xtrain.main()

    assert calls == [(["--epochs", "1"], {})]
