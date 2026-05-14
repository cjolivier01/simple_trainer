from __future__ import annotations

import json
import os
import signal
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import FrameType
from typing import Any

import torch

Batch = Any
LossFn = Callable[[torch.nn.Module, Batch, torch.device], torch.Tensor]


@dataclass(frozen=True)
class TrainerConfig:
    max_iters: int
    save_path: str = "./checkpoint.pt"
    checkpoint_every: int | None = None
    weights_from: str | None = None
    init_from: str | None = None
    logging_interval: int = 1
    pause_on_sigusr1: bool = True
    pause_runtime_dir: str | None = None
    pause_snapshot_name: str = "sigusr1"
    pause_barrier_timeout: float = 60.0
    pause_use_snapshotd: bool = True
    pause_criu_bin: str = "criu"
    pause_criu_ns_bin: str | None = None
    pause_criu_log_level: int | None = None
    pause_rank_subdirs: bool = True


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device_index: int
    device: torch.device
    backend: str = ""

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @classmethod
    def initialize(cls) -> DistributedContext:
        enabled = is_distributed_env()
        if not enabled:
            device_index = 0
            device = _cuda_device(device_index, purpose="Training")
            return cls(
                enabled=False,
                rank=0,
                world_size=1,
                local_rank=0,
                device_index=device_index or 0,
                device=device,
                backend="",
            )

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        requested_backend = os.environ.get("DDP_BACKEND", "").strip()
        if requested_backend and requested_backend != "nccl":
            raise RuntimeError(
                "Distributed training in this repo requires DDP_BACKEND=nccl"
            )
        backend = "nccl"

        device_index = local_device_index(local_rank)
        device = _cuda_device(device_index, purpose="Distributed training")

        import torch.distributed as dist

        init_kwargs: dict[str, object] = {
            "backend": backend,
            "rank": rank,
            "world_size": world_size,
            "timeout": timedelta(seconds=_ddp_timeout_seconds()),
        }
        init_kwargs["device_id"] = device

        dist.init_process_group(**init_kwargs)
        return cls(
            enabled=True,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device_index=device_index,
            device=device,
            backend=backend,
        )

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        model = model.to(self.device)
        if not self.enabled:
            return model

        from torch.nn.parallel import DistributedDataParallel

        return DistributedDataParallel(model, device_ids=[self.device_index])

    def all_gather_bool(self, value: bool) -> list[bool]:
        if not self.enabled:
            return [value]
        import torch.distributed as dist

        local = torch.tensor([1 if value else 0], dtype=torch.uint8, device=self.device)
        gathered = [torch.zeros_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local)
        return [bool(item.item()) for item in gathered]

    def barrier(self, *, timeout_seconds: float | None = None) -> None:
        if not self.enabled:
            return
        import torch.distributed as dist

        dist.barrier(device_ids=[self.device_index])

    def cleanup(self) -> None:
        if not self.enabled:
            return
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: LossFn,
        train_loader: Iterable[Batch],
        config: TrainerConfig,
        context: DistributedContext,
        train_sampler: Any | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.config = config
        self.context = context
        self.train_sampler = train_sampler
        self.global_step = 0
        self.start_pass_index = 0
        self.start_step_in_pass = 0
        self.pass_index = 0
        self.step_in_pass = 0
        self.signal_handler = Sigusr1PauseHandler(enabled=config.pause_on_sigusr1)

    @property
    def module(self) -> torch.nn.Module:
        return unwrap_model(self.model)

    @property
    def is_primary(self) -> bool:
        return self.context.is_primary

    def fit(self) -> None:
        self.load_initial_state()
        self.write_pause_metadata()
        self.signal_handler.install(rank=self.context.rank)
        self.model.train()
        self.pass_index = self.start_pass_index
        skip = self.start_step_in_pass
        started = False
        paused = False
        try:
            while self.global_step < self.config.max_iters:
                if started:
                    self.pass_index += 1
                    self.step_in_pass = 0
                started = True
                self._train_pass(skip=skip)
                skip = 0
        except PauseExit:
            paused = True
        finally:
            if not paused:
                self.save_checkpoint()
            self.context.cleanup()

    def _train_pass(self, *, skip: int) -> None:
        if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(self.pass_index)

        running_loss = 0.0
        last_batch_index = 0
        for batch_index, batch in enumerate(self.train_loader, start=1):
            last_batch_index = batch_index
            if batch_index <= skip:
                continue
            if self.global_step >= self.config.max_iters:
                return

            loss = self.train_step(batch)
            self.step_in_pass = batch_index
            running_loss += float(loss.detach().item())
            if self.should_log(self.global_step):
                self.log_step(running_loss=running_loss)
                running_loss = 0.0
            if self.should_checkpoint():
                self.save_checkpoint()
            self.handle_sigusr1_pause()

        if skip > 0 and last_batch_index <= skip and self.is_primary:
            print(
                f"WARN: resume skip={skip} >= batches in pass "
                f"{self.pass_index} ({last_batch_index}); advancing to next "
                f"pass without training. Loader may have shrunk since the "
                f"checkpoint was saved.",
                file=sys.stderr,
                flush=True,
            )

    def train_step(self, batch: Batch) -> torch.Tensor:
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model, batch, self.context.device)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return loss

    def should_log(self, step: int) -> bool:
        return self.is_primary and step % self.config.logging_interval == 0

    def log_step(self, *, running_loss: float) -> None:
        loss = running_loss / self.config.logging_interval
        print(f"step={self.global_step} loss={loss:.4f}")

    def should_checkpoint(self) -> bool:
        return (
            self.config.checkpoint_every is not None
            and self.global_step % self.config.checkpoint_every == 0
        )

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "pass_index": self.pass_index,
            "step_in_pass": self.step_in_pass,
        }

    def save_checkpoint(self) -> None:
        if not self.is_primary:
            return
        torch.save(self.checkpoint_payload(), self.config.save_path)
        print(f"Saved checkpoint to {self.config.save_path}")

    def load_initial_state(self) -> None:
        load_path = self.config.init_from or self.config.weights_from
        if load_path is None:
            return
        checkpoint = load_checkpoint(load_path, map_location=self.context.device)
        self.module.load_state_dict(checkpoint["model"])
        if self.config.init_from is not None:
            optimizer_state = checkpoint.get("optimizer")
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)
            self.global_step = int(checkpoint.get("global_step", 0))
            self.start_pass_index = int(checkpoint.get("pass_index", 0))
            self.start_step_in_pass = int(checkpoint.get("step_in_pass", 0))
            self.pass_index = self.start_pass_index
            self.step_in_pass = self.start_step_in_pass
            if self.is_primary:
                print(
                    f"Resumed from {load_path} at global_step={self.global_step} "
                    f"pass_index={self.start_pass_index} "
                    f"step_in_pass={self.start_step_in_pass}"
                )
        elif self.is_primary:
            print(f"Loaded weights from {load_path}")

    def write_pause_metadata(self) -> None:
        if not self.config.pause_runtime_dir:
            return
        runtime_dir = Path(self.config.pause_runtime_dir)
        pids_dir = runtime_dir / "pids"
        pids_dir.mkdir(parents=True, exist_ok=True)
        rank_runtime_dir = self.pause_rank_runtime_dir()
        rank_runtime_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "pid_start_time": _process_start_time(os.getpid()),
            "rank": self.context.rank,
            "world_size": self.context.world_size,
            "local_rank": self.context.local_rank,
            "runtime_dir": str(rank_runtime_dir),
            "snapshot_name": self.config.pause_snapshot_name,
            "save_path": self.config.save_path,
            "argv": sys.argv,
        }
        _atomic_write_json(pids_dir / f"rank-{self.context.rank}.json", payload)

    def pause_rank_runtime_dir(self) -> Path:
        base = Path(self.config.pause_runtime_dir or ".").resolve()
        if self.config.pause_rank_subdirs:
            return base / f"rank-{self.context.rank}"
        return base

    def handle_sigusr1_pause(self) -> None:
        local_received = self.signal_handler.consume()
        flags = self.context.all_gather_bool(local_received)
        if local_received:
            print(
                f"rank={self.context.rank} received SIGUSR1 at "
                f"global_step={self.global_step}",
                flush=True,
            )
        if not any(flags):
            return

        if self.is_primary:
            ranks = [str(index) for index, value in enumerate(flags) if value]
            print(
                "SIGUSR1 pause requested by rank(s) "
                f"{','.join(ranks)} at global_step={self.global_step}",
                flush=True,
            )
            self.save_checkpoint()

        self.context.barrier(timeout_seconds=self.config.pause_barrier_timeout)
        if self.config.pause_runtime_dir:
            self.snapshot_for_pause()
            self.signal_handler.clear()
            self.context.barrier(timeout_seconds=self.config.pause_barrier_timeout)
            return

        self.context.barrier(timeout_seconds=self.config.pause_barrier_timeout)
        raise PauseExit

    def snapshot_for_pause(self) -> None:
        try:
            import snapshot
        except ImportError as exc:
            raise RuntimeError(
                "SIGUSR1 pause requested a runtime snapshot, but the snapshot "
                "package is not importable"
            ) from exc

        runtime_dir = self.pause_rank_runtime_dir()
        if self.is_primary:
            print(
                f"Saving runtime snapshot '{self.config.pause_snapshot_name}' "
                f"under {Path(self.config.pause_runtime_dir or '.').resolve()}",
                flush=True,
            )
        snapshot.checkpoint(
            runtime_dir=runtime_dir,
            snapshot_name=self.config.pause_snapshot_name,
            criu_bin=self.config.pause_criu_bin,
            criu_ns_bin=self.config.pause_criu_ns_bin,
            sudo=self.config.pause_use_snapshotd,
            criu_log_level=self.config.pause_criu_log_level,
            helper_start_timeout=min(10.0, self.config.pause_barrier_timeout),
        )
        if getattr(snapshot, "process_was_restored", lambda: False)():
            print(
                f"rank={self.context.rank} restored from "
                f"'{self.config.pause_snapshot_name}' at "
                f"global_step={self.global_step}",
                flush=True,
            )
            self.model.train()
            return
        raise PauseExit


def is_distributed_env() -> bool:
    return any(v in os.environ for v in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))


def local_device_index(local_rank: int) -> int:
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    if len(visible) <= 1:
        return 0
    return local_rank


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def load_checkpoint(path: str, map_location: torch.device) -> dict[str, Any]:
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj}


def supervised_loss_fn(
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> LossFn:
    def _loss_fn(
        model: torch.nn.Module,
        batch: Batch,
        device: torch.device,
    ) -> torch.Tensor:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        return criterion(logits, labels)

    return _loss_fn


def causal_lm_loss_fn() -> LossFn:
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def _loss_fn(
        model: torch.nn.Module,
        batch: Batch,
        device: torch.device,
    ) -> torch.Tensor:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        return criterion(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
        )

    return _loss_fn


def _cuda_device(device_index: int, *, purpose: str) -> torch.device:
    require_cuda_available(purpose)
    torch.cuda.set_device(device_index)
    return torch.device(f"cuda:{device_index}")


def require_cuda_available(purpose: str = "Training") -> None:
    if not torch_cuda_available():
        raise RuntimeError(f"{purpose} requires CUDA; CPU fallback is not supported")


def torch_cuda_available() -> bool:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None and cuda_visible_devices.strip() in {"", "-1"}:
        return False
    return torch.cuda.is_available()


def _ddp_timeout_seconds() -> float:
    value = os.environ.get("DDP_TIMEOUT_SECONDS", "180").strip()
    try:
        return float(value)
    except ValueError:
        return 180.0


class PauseExit(Exception):
    pass


class Sigusr1PauseHandler:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled and hasattr(signal, "SIGUSR1")
        self._received = False
        self._installed = False
        self._previous: Any = None

    def install(self, *, rank: int) -> None:
        if not self.enabled or self._installed:
            return
        self._previous = signal.getsignal(signal.SIGUSR1)

        def _handle(signum: int, frame: FrameType | None) -> None:  # noqa: ARG001
            self._received = True

        signal.signal(signal.SIGUSR1, _handle)
        self._installed = True
        if rank == 0:
            print("Installed SIGUSR1 pause handler", flush=True)

    def consume(self) -> bool:
        received = self._received
        self._received = False
        return received

    def clear(self) -> None:
        self._received = False


def _process_start_time(pid: int) -> str:
    try:
        stat_text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return ""
    fields = stat_text.split()
    if len(fields) < 22:
        return ""
    return fields[21]


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp_path.replace(path)
