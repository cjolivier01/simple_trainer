import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.letnet import LeNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LeNet on CIFAR-10")
    parser.add_argument("--data-dir", default="./data", help="Dataset directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional step limit per epoch")
    parser.add_argument("--save-path", default="./lenet_cifar10.pt", help="Checkpoint output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = LeNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, (images, labels) in enumerate(loader, start=1):
            if args.max_steps is not None and step > args.max_steps:
                break

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 100 == 0:
                print(f"epoch={epoch + 1} step={step} loss={running_loss / 100:.4f}")
                running_loss = 0.0

    torch.save(model.state_dict(), args.save_path)
    print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
