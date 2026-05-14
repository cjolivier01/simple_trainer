from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch

Batch = Any
LossFn = Callable[[torch.nn.Module, Batch, torch.device], torch.Tensor]


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int
    max_steps: int | None = None
    checkpoint_every: int | None = None
    save_path: str = "./checkpoint.pt"
    weights_from: str | None = None
    init_from: str | None = None
    logging_interval: int = 1


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device_index: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @classmethod
    def initialize(cls) -> DistributedContext:
        enabled = is_distributed_env()
        if not enabled:
            device = _default_device()
            device_index = device.index if device.type == "cuda" else 0
            return cls(
                enabled=False,
                rank=0,
                world_size=1,
                local_rank=0,
                device_index=device_index or 0,
                device=device,
            )

        import torch.distributed as dist

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        backend = os.environ.get("DDP_BACKEND", "").strip()
        if not backend:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        device_index = local_device_index(local_rank)
        init_kwargs: dict[str, object] = {
            "backend": backend,
            "rank": rank,
            "world_size": world_size,
        }
        if torch.cuda.is_available():
            torch.cuda.set_device(device_index)
            device = torch.device(f"cuda:{device_index}")
            if backend == "nccl":
                init_kwargs["device_id"] = device
        else:
            device = torch.device("cpu")

        dist.init_process_group(**init_kwargs)
        return cls(
            enabled=True,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device_index=device_index,
            device=device,
        )

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        model = model.to(self.device)
        if not self.enabled:
            return model

        from torch.nn.parallel import DistributedDataParallel

        device_ids = [self.device_index] if self.device.type == "cuda" else None
        return DistributedDataParallel(model, device_ids=device_ids)

    def barrier(self) -> None:
        if not self.enabled:
            return
        import torch.distributed as dist

        dist.barrier()

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
        self.start_epoch = 0

    @property
    def module(self) -> torch.nn.Module:
        return unwrap_model(self.model)

    @property
    def is_primary(self) -> bool:
        return self.context.is_primary

    def fit(self) -> None:
        self.load_initial_state()
        self.model.train()
        epoch = self.start_epoch
        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                self.train_epoch(epoch)
        finally:
            self.save_checkpoint(epoch)
            self.context.cleanup()

    def train_epoch(self, epoch: int) -> None:
        if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)

        running_loss = 0.0
        for step, batch in enumerate(self.train_loader, start=1):
            if self.config.max_steps is not None and step > self.config.max_steps:
                break

            loss = self.train_step(batch)
            running_loss += float(loss.detach().item())
            if self.should_log(step):
                self.log_step(epoch=epoch, step=step, running_loss=running_loss)
                running_loss = 0.0
            if self.should_checkpoint():
                self.save_checkpoint(epoch)

    def train_step(self, batch: Batch) -> torch.Tensor:
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model, batch, self.context.device)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return loss

    def should_log(self, step: int) -> bool:
        return self.is_primary and step % self.config.logging_interval == 0

    def log_step(self, *, epoch: int, step: int, running_loss: float) -> None:
        loss = running_loss / self.config.logging_interval
        print(f"epoch={epoch + 1} step={step} loss={loss:.4f}")

    def should_checkpoint(self) -> bool:
        return (
            self.config.checkpoint_every is not None
            and self.global_step % self.config.checkpoint_every == 0
        )

    def checkpoint_payload(self, epoch: int) -> dict[str, object]:
        return {
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": epoch,
        }

    def save_checkpoint(self, epoch: int) -> None:
        if not self.is_primary:
            return
        torch.save(self.checkpoint_payload(epoch), self.config.save_path)
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
            self.start_epoch = int(checkpoint.get("epoch", 0))
            if self.is_primary:
                print(
                    f"Resumed from {load_path} at epoch={self.start_epoch} "
                    f"global_step={self.global_step}"
                )
        elif self.is_primary:
            print(f"Loaded weights from {load_path}")


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


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
