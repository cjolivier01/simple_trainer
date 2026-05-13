import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.letnet import LeNet  # noqa: E402


CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LeNet inference")
    parser.add_argument(
        "--checkpoint", default="./lenet_cifar10.pt", help="Model checkpoint path"
    )
    parser.add_argument("--image", default=None, help="Optional image path")
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Dataset directory (used when --image is not set)",
    )
    parser.add_argument(
        "--index", type=int, default=0, help="CIFAR test index when --image is not set"
    )
    return parser.parse_args()


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def load_input(args: argparse.Namespace, transform: transforms.Compose) -> torch.Tensor:
    if args.image:
        image = Image.open(args.image).convert("RGB")
        return transform(image).unsqueeze(0)

    dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True)
    image, _ = dataset[args.index]
    return transform(image).unsqueeze(0)


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    inputs = load_input(args, build_transform()).to(device)

    with torch.no_grad():
        logits = model(inputs)
        predicted = torch.argmax(logits, dim=1).item()

    print(f"predicted_class={CLASSES[predicted]} ({predicted})")


if __name__ == "__main__":
    main()
