from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from tools.trainer import (
    DistributedContext,
    Trainer,
    TrainerConfig,
    load_checkpoint,
    supervised_loss_fn,
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


def _loader() -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ]
        ),
        torch.tensor([0, 1, 0, 1]),
    )
    return DataLoader(dataset, batch_size=2, shuffle=False)


def test_trainer_runs_generic_model_and_saves_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=supervised_loss_fn(torch.nn.CrossEntropyLoss()),
        train_loader=_loader(),
        config=TrainerConfig(max_steps=2, save_path=str(checkpoint)),
        context=_cpu_context(),
    )

    trainer.fit()

    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["global_step"] == 2
    assert payload["pass_index"] == 0
    assert payload["step_in_pass"] == 2
    assert "weight" in payload["model"]
    assert "optimizer" in payload


def test_load_checkpoint_accepts_legacy_raw_state_dict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    state_dict = torch.nn.Linear(2, 2).state_dict()
    torch.save(state_dict, checkpoint)

    payload = load_checkpoint(str(checkpoint), map_location=torch.device("cpu"))

    assert payload.keys() == {"model"}
    assert payload["model"].keys() == state_dict.keys()
    for key, expected in state_dict.items():
        assert torch.equal(payload["model"][key], expected)
