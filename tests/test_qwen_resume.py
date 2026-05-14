from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.qwen import QwenConfig, QwenForCausalLM
from tools.train import SyntheticTokenDataset
from tools.train import LLaVAInstructDataset
from tools.trainer import (
    DistributedContext,
    Trainer,
    TrainerConfig,
    causal_lm_loss_fn,
)


def _cpu_context() -> DistributedContext:
    return DistributedContext(
        enabled=False,
        rank=0,
        world_size=1,
        local_rank=0,
        device_index=0,
        device=torch.device("cpu"),
    )


def _qwen_config() -> QwenConfig:
    return QwenConfig(
        vocab_size=32,
        seq_len=8,
        hidden_size=16,
        num_heads=4,
        num_key_value_heads=2,
        num_layers=1,
        intermediate_size=32,
    )


def test_qwen_forward_shape() -> None:
    model = QwenForCausalLM(_qwen_config())
    input_ids = torch.randint(0, model.config.vocab_size, (2, model.config.seq_len))

    logits = model(input_ids)

    assert logits.shape == (2, model.config.seq_len, model.config.vocab_size)


def test_llava_instruct_dataset_formats_text_and_masks_padding(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "llava_instruct_150k.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "hello"},
                        {"from": "gpt", "value": "hi"},
                    ]
                },
                {
                    "image": "ignored.jpg",
                    "conversations": [
                        {"from": "human", "value": "what is here?"},
                        {"from": "gpt", "value": "a test image"},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    dataset = LLaVAInstructDataset(data_path, seq_len=96, vocab_size=257)
    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["input_ids"].shape == (96,)
    assert sample["labels"].shape == (96,)
    assert torch.equal(sample["labels"] == -100, sample["input_ids"] == 0)
    assert sample["input_ids"].max().item() <= 256


def _run_qwen_training(
    *,
    save_path: Path,
    max_iters: int,
    init_from: Path | None = None,
) -> tuple[list[float], dict[str, torch.Tensor]]:
    torch.manual_seed(123)
    config = _qwen_config()
    model = QwenForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    generator = torch.Generator().manual_seed(999)
    loader = DataLoader(
        SyntheticTokenDataset(
            size=64,
            seq_len=config.seq_len,
            vocab_size=config.vocab_size,
            seed=456,
        ),
        batch_size=4,
        shuffle=True,
        generator=generator,
    )
    losses: list[float] = []
    base_loss_fn = causal_lm_loss_fn()

    def recording_loss_fn(
        wrapped_model: torch.nn.Module,
        batch: object,
        device: torch.device,
    ) -> torch.Tensor:
        loss = base_loss_fn(wrapped_model, batch, device)
        losses.append(float(loss.detach().item()))
        return loss

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=recording_loss_fn,
        train_loader=loader,
        config=TrainerConfig(
            max_iters=max_iters,
            save_path=str(save_path),
            init_from=str(init_from) if init_from is not None else None,
            logging_interval=999,
            pause_on_sigusr1=False,
        ),
        context=_cpu_context(),
    )
    trainer.fit()
    payload = torch.load(save_path, map_location="cpu")
    return losses, payload["model"]


def test_hard_resume_matches_uninterrupted_qwen_loss_and_weights(
    tmp_path: Path,
) -> None:
    uninterrupted_losses, uninterrupted_state = _run_qwen_training(
        save_path=tmp_path / "uninterrupted.pt",
        max_iters=4,
    )
    first_half_losses, _ = _run_qwen_training(
        save_path=tmp_path / "pause.pt",
        max_iters=2,
    )
    resumed_losses, resumed_state = _run_qwen_training(
        save_path=tmp_path / "resumed.pt",
        max_iters=4,
        init_from=tmp_path / "pause.pt",
    )

    assert torch.allclose(
        torch.tensor(first_half_losses), torch.tensor(uninterrupted_losses[:2])
    )
    assert torch.allclose(
        torch.tensor(resumed_losses), torch.tensor(uninterrupted_losses[2:])
    )
    for name, expected in uninterrupted_state.items():
        assert torch.allclose(resumed_state[name], expected, atol=1e-6), name


def test_sigusr1_pause_uses_named_rank_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    fake_snapshot = types.SimpleNamespace(
        checkpoint=lambda **kwargs: calls.append(kwargs),
        process_was_restored=lambda: True,
    )
    monkeypatch.setitem(sys.modules, "snapshot", fake_snapshot)

    torch.manual_seed(123)
    config = _qwen_config()
    model = QwenForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loader = DataLoader(
        SyntheticTokenDataset(
            size=8,
            seq_len=config.seq_len,
            vocab_size=config.vocab_size,
            seed=456,
        ),
        batch_size=2,
        shuffle=False,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=causal_lm_loss_fn(),
        train_loader=loader,
        config=TrainerConfig(
            max_iters=1,
            save_path=str(tmp_path / "hard.pt"),
            pause_runtime_dir=str(tmp_path / "runtime"),
            pause_snapshot_name="test-point",
            pause_use_snapshotd=False,
        ),
        context=_cpu_context(),
    )
    consume_once = _consume_sequence(True, False)
    monkeypatch.setattr(trainer.signal_handler, "consume", consume_once)

    trainer.fit()

    assert len(calls) == 1
    assert calls[0]["runtime_dir"] == tmp_path / "runtime" / "step_1" / "rank-0"
    assert calls[0]["snapshot_name"] == "test-point"
    assert calls[0]["sudo"] is False
    payload = torch.load(tmp_path / "hard.pt", map_location="cpu")
    assert payload["global_step"] == 1
    assert payload["step_in_pass"] == 1


def _consume_sequence(*values: bool) -> Callable[[], bool]:
    iterator = iter(values)

    def _consume() -> bool:
        return next(iterator, False)

    return _consume
