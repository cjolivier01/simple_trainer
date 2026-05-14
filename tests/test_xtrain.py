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
    fake_snapshot_module.build_autosnapshot_now_result = (
        lambda *, bootstrap_script, runtime_dir=None: types.SimpleNamespace(
            snapshot_id="abc123def456",
            generation_root=tmp_path / "generation",
        )
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


# --------------------------------------------------------------------------- #
# Per-rank snapshot build/restore (DDP)
# --------------------------------------------------------------------------- #


def _enter_torchrun_env(monkeypatch: pytest.MonkeyPatch, *, local_rank: str) -> None:
    """Set the torchrun env vars that xtrain.is_ddp_context() checks."""
    monkeypatch.setenv("LOCAL_RANK", local_rank)
    monkeypatch.setenv("RANK", local_rank)
    monkeypatch.setenv("WORLD_SIZE", "2")


def test_apply_rank_suffix_outside_ddp_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "SLURM_LOCALID"):
        monkeypatch.delenv(var, raising=False)
    assert (
        xtrain._apply_rank_suffix_to_tag("ai/train_lenet:abc") == "ai/train_lenet:abc"
    )
    assert xtrain._apply_rank_suffix_to_tag("ai/train_lenet") == "ai/train_lenet"


def test_apply_rank_suffix_appends_local_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enter_torchrun_env(monkeypatch, local_rank="3")
    assert (
        xtrain._apply_rank_suffix_to_tag("ai/train_lenet:abc")
        == "ai/train_lenet:abc-rank-3"
    )


def test_apply_rank_suffix_skips_bare_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare-repo refs (no :tag) get no suffix — they resolve via default-tag."""
    _enter_torchrun_env(monkeypatch, local_rank="1")
    assert xtrain._apply_rank_suffix_to_tag("ai/train_lenet") == "ai/train_lenet"


def test_apply_rank_suffix_skips_already_ranked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller-supplied per-rank refs aren't double-suffixed."""
    _enter_torchrun_env(monkeypatch, local_rank="0")
    assert (
        xtrain._apply_rank_suffix_to_tag("ai/train_lenet:abc-rank-7")
        == "ai/train_lenet:abc-rank-7"
    )


def test_apply_rank_suffix_skips_host_port_in_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`host:5000/repo` (port-in-host) shouldn't be confused with a :tag."""
    _enter_torchrun_env(monkeypatch, local_rank="0")
    assert (
        xtrain._apply_rank_suffix_to_tag("registry.example:5000/ai/train_lenet")
        == "registry.example:5000/ai/train_lenet"
    )


def test_per_rank_runtime_dir_outside_ddp_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "SLURM_LOCALID"):
        monkeypatch.delenv(var, raising=False)
    assert xtrain._per_rank_autosnapshot_runtime_dir() is None


def test_per_rank_runtime_dir_in_ddp_appends_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enter_torchrun_env(monkeypatch, local_rank="2")
    runtime_dir = xtrain._per_rank_autosnapshot_runtime_dir()
    assert runtime_dir is not None
    # Sibling directory of the snapshot package's default runtime dir.
    assert runtime_dir.name.endswith("-rank-2")


def test_build_snapshot_under_ddp_uses_per_rank_runtime_and_tag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """build_autosnapshot_now is called with a per-rank runtime_dir, and
    the returned generation root is tagged with the rank-suffixed tag."""
    _enter_torchrun_env(monkeypatch, local_rank="1")
    bootstrap = tmp_path / "bootstrap_train.py"
    bootstrap.write_text("# stub\n", encoding="utf-8")
    generation_root = tmp_path / "generation-rank-1"

    build_calls: list[dict[str, object]] = []
    fake_snapshot_module = types.ModuleType("snapshot")
    fake_snapshot_module.__path__ = []

    def fake_build(*, bootstrap_script: str, runtime_dir: object = None) -> object:
        build_calls.append(
            {"bootstrap_script": bootstrap_script, "runtime_dir": runtime_dir}
        )
        return types.SimpleNamespace(
            snapshot_id="deadbeef0001",
            generation_root=generation_root,
        )

    fake_snapshot_module.build_autosnapshot_now_result = fake_build
    monkeypatch.setitem(sys.modules, "snapshot", fake_snapshot_module)
    tag_calls: list[dict[str, object]] = []
    fake_oci_module = types.ModuleType("snapshot.oci_repository")

    def fake_tag_runtime_cache_snapshot(
        tag: str,
        *,
        runtime_dir: object,
        snapshot_id: str,
        ref: str,
    ) -> None:
        tag_calls.append(
            {
                "tag": tag,
                "runtime_dir": runtime_dir,
                "snapshot_id": snapshot_id,
                "ref": ref,
            }
        )

    fake_oci_module.tag_runtime_cache_snapshot = fake_tag_runtime_cache_snapshot
    monkeypatch.setitem(sys.modules, "snapshot.oci_repository", fake_oci_module)
    monkeypatch.setattr(xtrain, "BOOTSTRAP", str(bootstrap))

    cli_calls: list[list[str]] = []
    monkeypatch.setattr(
        xtrain.subprocess,
        "check_call",
        lambda cmd, **_: cli_calls.append(list(cmd)) or 0,
    )

    xtrain._maybe_build_and_publish_snapshot(tag_ref="ai/train_lenet:abc", push=True)

    assert len(build_calls) == 1
    runtime_dir_arg = build_calls[0]["runtime_dir"]
    assert runtime_dir_arg is not None and str(runtime_dir_arg).endswith("-rank-1")
    assert tag_calls == [
        {
            "tag": "ai/train_lenet:abc-rank-1",
            "runtime_dir": generation_root,
            "snapshot_id": "deadbeef0001",
            "ref": "ai/train_lenet:abc-rank-1",
        }
    ]
    push_cmd = next(c for c in cli_calls if "push" in c)
    assert push_cmd[-1] == "ai/train_lenet:abc-rank-1"


def test_restore_under_ddp_hydrates_per_rank_tag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """--xt-restore=ai/train_lenet:abc → rank N hydrates ai/train_lenet:abc-rank-N."""
    _enter_torchrun_env(monkeypatch, local_rank="0")
    hydrated: list[str] = []

    result = types.SimpleNamespace(
        snapshot_id="abcdef1234567890",
        generation_root=tmp_path / "generation",
    )
    monkeypatch.setattr(
        xtrain,
        "_hydrate_restore_reference",
        lambda ref: hydrated.append(ref) or result,
    )
    monkeypatch.setattr(xtrain, "_ensure_bootstrap_for_restore", lambda **_: None)
    manual_calls: list[tuple[list[str], object]] = []
    monkeypatch.setattr(
        xtrain,
        "_run_manual_ddp_restore",
        lambda args, runtime_dir: manual_calls.append((list(args), runtime_dir))
        or True,
    )

    def _fail_bootstrap(*args: object, **kwargs: object) -> None:
        raise AssertionError("manual DDP restore should bypass bootstrap restore")

    monkeypatch.setattr(xtrain, "_run_bootstrap", _fail_bootstrap)
    monkeypatch.setattr(
        sys, "argv", ["xtrain.py", "--xt-restore=ai/train_lenet:abc", "--epochs", "1"]
    )

    xtrain.main()

    assert hydrated == ["ai/train_lenet:abc-rank-0"]
    assert manual_calls == [(["--epochs", "1"], result.generation_root)]


def test_default_restore_under_ddp_applies_default_tag_before_rank_suffix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enter_torchrun_env(monkeypatch, local_rank="1")
    monkeypatch.setenv("SNAPSHOT_OCI_TAG", "cuda-x86")
    hydrated: list[str] = []

    result = types.SimpleNamespace(
        snapshot_id="abcdef1234567890",
        generation_root=tmp_path / "generation",
    )
    monkeypatch.setattr(
        xtrain,
        "_hydrate_restore_reference",
        lambda ref: hydrated.append(ref) or result,
    )
    monkeypatch.setattr(xtrain, "_ensure_bootstrap_for_restore", lambda **_: None)
    manual_calls: list[tuple[list[str], object]] = []
    monkeypatch.setattr(
        xtrain,
        "_run_manual_ddp_restore",
        lambda args, runtime_dir: manual_calls.append((list(args), runtime_dir))
        or True,
    )

    def _fail_bootstrap(*args: object, **kwargs: object) -> None:
        raise AssertionError("manual DDP restore should bypass bootstrap restore")

    monkeypatch.setattr(xtrain, "_run_bootstrap", _fail_bootstrap)

    restored = xtrain._try_default_restore(
        "ai/train_lenet",
        ["--epochs", "1"],
        external_only=1,
    )

    assert restored is True
    assert hydrated == ["ai/train_lenet:cuda-x86-rank-1"]
    assert manual_calls == [(["--epochs", "1"], result.generation_root)]


def test_manual_ddp_restore_passes_current_torchrun_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enter_torchrun_env(monkeypatch, local_rank="0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29617")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "abc")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")

    fake_snapshot_module = types.ModuleType("snapshot")
    fake_snapshot_module.__path__ = []
    fake_runtime_module = types.ModuleType("snapshot.runtime")
    restore_calls: list[dict[str, object]] = []

    def fake_restore_runtime(**kwargs: object) -> None:
        restore_calls.append(kwargs)

    fake_runtime_module.restore_runtime = fake_restore_runtime
    monkeypatch.setitem(sys.modules, "snapshot", fake_snapshot_module)
    monkeypatch.setitem(sys.modules, "snapshot.runtime", fake_runtime_module)
    monkeypatch.setattr(xtrain, "_compute_restore_name", lambda: "rank-0-test")
    monkeypatch.setattr(
        xtrain,
        "_restored_worker_exit_code",
        lambda runtime_dir, restore_name: 0,
    )
    cleanup_calls: list[tuple[object, str]] = []
    monkeypatch.setattr(
        xtrain,
        "_spawn_restore_state_cleanup",
        lambda runtime_dir, restore_name: cleanup_calls.append(
            (runtime_dir, restore_name)
        ),
    )

    restored = xtrain._run_manual_ddp_restore(["--epochs", "1"], tmp_path)

    assert restored is True
    assert len(restore_calls) == 1
    restore_kwargs = restore_calls[0]
    restore_env = {
        key: value
        for key, value in (pair.split("=", 1) for pair in restore_kwargs["restore_env"])
    }
    assert restore_kwargs["runtime_dir"] == tmp_path
    assert restore_kwargs["restore_name"] == "rank-0-test"
    assert restore_kwargs["script_args"] == ["--epochs", "1"]
    assert restore_env["RANK"] == "0"
    assert restore_env["LOCAL_RANK"] == "0"
    assert restore_env["WORLD_SIZE"] == "2"
    assert restore_env["MASTER_ADDR"] == "127.0.0.1"
    assert restore_env["MASTER_PORT"] == "29617"
    assert restore_env["DDP_BACKEND"] == "nccl"
    assert restore_env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert restore_env["TORCHELASTIC_RUN_ID"] == "abc"
    assert "GLOO_SOCKET_IFNAME" not in restore_env
    assert cleanup_calls == [(tmp_path, "rank-0-test")]


def test_manual_ddp_restore_rejects_gloo_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enter_torchrun_env(monkeypatch, local_rank="0")
    monkeypatch.setenv("DDP_BACKEND", "gloo")

    with pytest.raises(RuntimeError, match="DDP_BACKEND=nccl"):
        xtrain._ddp_restore_env_pairs()


def test_auto_mode_manual_restore_runtime_prefers_per_rank_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enter_torchrun_env(monkeypatch, local_rank="1")
    rank_runtime = tmp_path / "runtime-rank-1"
    generation = rank_runtime / "autosnapshot" / "generations" / "current"
    calls: list[object] = []

    monkeypatch.delenv(xtrain.XTRAIN_RUNTIME_DIR_ENV, raising=False)
    monkeypatch.setattr(
        xtrain, "_per_rank_autosnapshot_runtime_dir", lambda: rank_runtime
    )
    monkeypatch.setattr(
        xtrain,
        "_current_autosnapshot_generation_root",
        lambda runtime_dir=None: calls.append(runtime_dir) or generation,
    )

    assert xtrain._auto_mode_manual_restore_runtime_dir() == generation
    assert calls == [rank_runtime]


def test_should_use_manual_restore_checks_auto_mode_runtime_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enter_torchrun_env(monkeypatch, local_rank="0")
    bootstrap = tmp_path / "bootstrap_train.py"
    generation = tmp_path / "generation"
    (generation / "images").mkdir(parents=True)
    bootstrap.write_text("# generated\n", encoding="utf-8")

    fake_runtime_module = types.ModuleType("snapshot.runtime")
    fake_runtime_module.IMAGES_DIR = "images"
    monkeypatch.setitem(sys.modules, "snapshot.runtime", fake_runtime_module)
    monkeypatch.setattr(xtrain, "BOOTSTRAP", str(bootstrap))
    monkeypatch.setattr(
        xtrain,
        "_auto_mode_manual_restore_runtime_dir",
        lambda: generation,
    )

    assert xtrain.should_use_manual_restore() is True


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
        sys, "argv", ["xtrain.py", "--xt-restore=auto", "--max-iters", "1"]
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
