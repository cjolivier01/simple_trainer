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


def test_parse_build_snapshot_and_tag_flags() -> None:
    values, forwarded = xtrain._parse_xtrain_flags(
        [
            "--xtrain-build-snapshot=1",
            "--xtrain-snapshot-tag=ai/train_lenet:cuda-x86",
            "--xtrain-snapshot-push=1",
            "--xtrain-profile=0",
            "--epochs",
            "1",
        ]
    )

    assert values["build-snapshot"] == 1
    assert values["snapshot-tag"] == "ai/train_lenet:cuda-x86"
    assert values["snapshot-push"] == 1
    assert values["profile"] == 0
    assert forwarded == ["--epochs", "1"]


def test_fast_zero_short_circuits_to_train_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        xtrain, "_run_train_directly", lambda args: calls.append(list(args))
    )

    # Default-restore must NOT run. Make it explode if reached so the test
    # catches a regression that lets fast=0 fall through.
    def _fail_default(*args: object, **kwargs: object) -> bool:
        raise AssertionError("default-restore must not run when --xtrain-fast=0")

    monkeypatch.setattr(xtrain, "_try_default_restore", _fail_default)
    monkeypatch.setattr(sys, "argv", ["xtrain.py", "--xtrain-fast=0", "--epochs", "1"])

    xtrain.main()

    assert calls == [["--epochs", "1"]]


def test_default_restore_miss_falls_back_to_train_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No --xt-restore + no build/tag/push: try default ref, miss, run train raw.

    Critically, save_stable_modules / start_import_tracking must NOT be called
    on the miss path — that's the "xtrain becomes a no-op" guarantee.
    """
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        xtrain,
        "_try_default_restore",
        lambda ref, args, **kwargs: calls.append(("default", (ref, list(args))))
        or False,
    )
    monkeypatch.setattr(
        xtrain, "_run_train_directly", lambda args: calls.append(("direct", list(args)))
    )
    fake_snapshot = types.SimpleNamespace(
        process_was_restored=lambda: calls.append(("process_was_restored", None))
        or False,
        start_import_tracking=lambda: calls.append(("track", None)),
        save_stable_modules=lambda **kwargs: calls.append(("save", kwargs)),
    )
    monkeypatch.setitem(sys.modules, "snapshot", fake_snapshot)
    monkeypatch.delenv(xtrain.DEFAULT_RESTORE_REF_ENV_VAR, raising=False)
    monkeypatch.setattr(sys, "argv", ["xtrain.py", "--epochs", "1"])

    xtrain.main()

    assert calls == [
        ("default", (xtrain.DEFAULT_RESTORE_REF, ["--epochs", "1"])),
        ("direct", ["--epochs", "1"]),
    ]


def test_default_restore_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(xtrain.DEFAULT_RESTORE_REF_ENV_VAR, "lenet/cifar10:abc")
    assert xtrain._default_restore_ref() == "lenet/cifar10:abc"


def test_default_restore_env_unset_uses_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(xtrain.DEFAULT_RESTORE_REF_ENV_VAR, raising=False)
    assert xtrain._default_restore_ref() == xtrain.DEFAULT_RESTORE_REF


def test_default_restore_env_auto_opts_into_autosnapshot_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """XTRAIN_DEFAULT_RESTORE_REF=auto skips default-restore and runs the
    legacy autosnapshot flow without requiring --xt-restore=auto on argv."""
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
    monkeypatch.setattr(
        xtrain.runpy, "run_path", lambda path, run_name: calls.append(("runpy", path))
    )

    # _try_default_restore would fail-test if reached; auto must skip it.
    def _fail_default(*args: object, **kwargs: object) -> bool:
        raise AssertionError("default-restore must not run when env=auto")

    monkeypatch.setattr(xtrain, "_try_default_restore", _fail_default)
    monkeypatch.setenv(xtrain.DEFAULT_RESTORE_REF_ENV_VAR, "auto")
    monkeypatch.setattr(sys, "argv", ["xtrain.py", "--epochs", "1"])

    xtrain.main()

    assert calls[0][0] == "generate"
    assert ("runpy", str(train)) in calls
    assert any(name == "save" for name, _ in calls)


def test_build_snapshot_implies_auto_and_publishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """--xtrain-snapshot-tag (or --xtrain-build-snapshot) selects auto mode
    and triggers _maybe_build_and_publish_snapshot post-save."""
    bootstrap = tmp_path / "bootstrap_train.py"
    train = tmp_path / "tools" / "train.py"
    train.parent.mkdir(parents=True)
    train.write_text("print('train')\n", encoding="utf-8")
    bootstrap.write_text("# pre-existing\n", encoding="utf-8")

    publish_calls: list[dict[str, object]] = []
    save_calls: list[dict[str, object]] = []
    fake_snapshot = types.SimpleNamespace(
        can_generate_bootstrap=lambda _repo_root: True,
        process_was_restored=lambda: False,
        start_import_tracking=lambda: None,
        save_stable_modules=lambda **kwargs: save_calls.append(kwargs),
    )
    monkeypatch.setitem(sys.modules, "snapshot", fake_snapshot)
    monkeypatch.setattr(xtrain, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(xtrain, "BOOTSTRAP", str(bootstrap))
    monkeypatch.setattr(xtrain, "TRAIN", str(train))
    monkeypatch.setattr(xtrain, "_current_autosnapshot_generation_root", lambda: None)
    monkeypatch.setattr(xtrain, "should_use_manual_restore", lambda: False)
    # Skip the bootstrap-restore attempt; force fall-through to train-direct.
    monkeypatch.setattr(
        xtrain,
        "_run_bootstrap",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("no bootstrap restore")
        ),
    )
    monkeypatch.setattr(xtrain.runpy, "run_path", lambda path, run_name: None)
    monkeypatch.setattr(
        xtrain,
        "_maybe_build_and_publish_snapshot",
        lambda *, tag_ref, push: publish_calls.append(
            {"tag_ref": tag_ref, "push": push}
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xtrain.py",
            "--xtrain-snapshot-tag=ai/train_lenet:abc",
            "--xtrain-snapshot-push=1",
            "--epochs",
            "1",
        ],
    )

    xtrain.main()

    assert publish_calls == [{"tag_ref": "ai/train_lenet:abc", "push": True}]
    assert len(save_calls) == 1


def test_publish_push_without_tag_is_warn_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_maybe_build_and_publish_snapshot with tag_ref="" but push=True emits
    a warning to stderr and does not invoke `snapshot tag` / `snapshot push`."""
    bootstrap = tmp_path / "bootstrap_train.py"
    bootstrap.write_text("# stub\n", encoding="utf-8")

    fake_snapshot_module = types.ModuleType("snapshot")
    fake_snapshot_module.build_autosnapshot_now = (
        lambda *, bootstrap_script: "abc123def456"
    )
    monkeypatch.setitem(sys.modules, "snapshot", fake_snapshot_module)
    monkeypatch.setattr(xtrain, "BOOTSTRAP", str(bootstrap))

    def _explode(*args: object, **kwargs: object) -> int:
        raise AssertionError(
            "subprocess.check_call must not be invoked when tag is unset"
        )

    monkeypatch.setattr(xtrain.subprocess, "check_call", _explode)

    xtrain._maybe_build_and_publish_snapshot(tag_ref="", push=True)

    captured = capsys.readouterr()
    assert "ignored" in captured.err and "snapshot-push" in captured.err


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
    monkeypatch.setattr(xtrain, "_compute_restore_name", lambda: "fake-restore-name")
    monkeypatch.setattr(
        sys, "argv", ["xtrain.py", "--xt-restore", "repo/image", "--epochs", "1"]
    )

    xtrain.main()

    assert calls == [
        ("hydrate", "repo/image"),
        ("ensure", {"external_only": 1}),
        (
            "run",
            (
                ["--epochs", "1"],
                {
                    "runtime_dir": result.generation_root,
                    "restore_name": "fake-restore-name",
                    "cuda_device_map": "",
                },
            ),
        ),
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
    monkeypatch.setattr(
        xtrain.runpy, "run_path", lambda path, run_name: calls.append(("runpy", path))
    )
    # Opt into the legacy autosnapshot flow; default-restore mode would otherwise
    # try to hydrate the default ref and fall through to running train.py raw.
    monkeypatch.setattr(
        sys, "argv", ["xtrain.py", "--xt-restore=auto", "--max-steps", "1"]
    )

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
    monkeypatch.setattr(
        xtrain, "_current_autosnapshot_generation_root", lambda: generation
    )
    monkeypatch.setattr(xtrain, "should_use_manual_restore", lambda: False)
    monkeypatch.setattr(
        xtrain,
        "_run_bootstrap",
        lambda args, **kwargs: calls.append((list(args), kwargs)),
    )
    monkeypatch.setattr(xtrain, "_compute_restore_name", lambda: "fake-restore-name")
    # Opt into the legacy autosnapshot flow; default-restore mode would otherwise
    # try to hydrate the default ref before reaching the bootstrap-run path.
    monkeypatch.setattr(
        sys, "argv", ["xtrain.py", "--xt-restore=auto", "--epochs", "1"]
    )

    xtrain.main()

    assert calls == [
        (
            ["--epochs", "1"],
            {
                "restore_name": "fake-restore-name",
                "cuda_device_map": "",
            },
        )
    ]
