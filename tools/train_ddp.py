import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.letnet import LeNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP train LeNet on CIFAR-10")
    parser.add_argument("--data-dir", default="./data", help="Dataset directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Optional step limit per epoch"
    )
    parser.add_argument(
        "--save-path", default="./lenet_cifar10_ddp.pt", help="Checkpoint output path"
    )
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    backend = os.environ.get("DDP_BACKEND", "").strip()
    if not backend:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_kwargs = {"backend": backend, "rank": rank, "world_size": world_size}

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        if backend == "nccl":
            init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(**init_kwargs)

    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    rank, _, local_rank = setup_distributed()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    if rank == 0:
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

    model = LeNet().to(device)
    ddp_model = DistributedDataParallel(
        model, device_ids=[local_rank] if torch.cuda.is_available() else None
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)

    ddp_model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        running_loss = 0.0

        for step, (images, labels) in enumerate(loader, start=1):
            if args.max_steps is not None and step > args.max_steps:
                break

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = ddp_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if rank == 0 and step % 100 == 0:
                print(f"epoch={epoch + 1} step={step} loss={running_loss / 100:.4f}")
                running_loss = 0.0

    if rank == 0:
        torch.save(ddp_model.module.state_dict(), args.save_path)
        print(f"Saved checkpoint to {args.save_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
