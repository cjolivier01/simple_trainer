import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.letnet import LeNet  # noqa: E402
from tools.trainer import (  # noqa: E402
    DistributedContext,
    Trainer,
    TrainerConfig,
    supervised_loss_fn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LeNet on CIFAR-10")
    parser.add_argument("--data-dir", default="./data", help="Dataset directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Optional step limit per epoch"
    )
    parser.add_argument(
        "--save-path", default="./lenet_cifar10.pt", help="Checkpoint output path"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save a checkpoint every N optimizer steps",
    )
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument(
        "-w",
        "--weights-from",
        default=None,
        help="Load model weights from a checkpoint and start training from step 0",
    )
    load_group.add_argument(
        "--init-from",
        default=None,
        help="Resume training from a checkpoint (model, optimizer, step, epoch)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = DistributedContext.initialize()
    if context.enabled:
        import torch.distributed as dist
        from torch.utils.data.distributed import DistributedSampler

    assert torch.cuda.is_available()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    if context.enabled:
        if context.is_primary:
            datasets.CIFAR10(
                root=args.data_dir,
                train=True,
                download=True,
                transform=transform,
            )
        dist.barrier()
        dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=False,
            transform=transform,
        )
        sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2
        )
    else:
        dataset = datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=transform
        )
        sampler = None
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )

    model = context.wrap_model(LeNet())
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=supervised_loss_fn(criterion),
        train_loader=loader,
        train_sampler=sampler,
        context=context,
        config=TrainerConfig(
            epochs=args.epochs,
            max_steps=args.max_steps,
            checkpoint_every=args.checkpoint_every,
            save_path=args.save_path,
            weights_from=args.weights_from,
            init_from=args.init_from,
        ),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
