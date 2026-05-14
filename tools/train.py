import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.letnet import LeNet  # noqa: E402
from models.qwen import QwenConfig, QwenForCausalLM  # noqa: E402
from tools.trainer import (  # noqa: E402
    DistributedContext,
    Trainer,
    TrainerConfig,
    causal_lm_loss_fn,
    supervised_loss_fn,
    torch_cuda_available,
)

LLAVA_INSTRUCT_REPO = "liuhaotian/LLaVA-Instruct-150K"
LLAVA_INSTRUCT_FILENAME = "llava_instruct_150k.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple model")
    parser.add_argument("--model", choices=("lenet", "qwen"), default="lenet")
    parser.add_argument("--data-dir", default="./data", help="Dataset directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--data-workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes per rank",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=8192,
        help="Qwen training sample count for synthetic data or LLaVA max samples",
    )
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
    parser.add_argument("--qwen-vocab-size", type=int, default=256)
    parser.add_argument("--qwen-seq-len", type=int, default=64)
    parser.add_argument("--qwen-hidden-size", type=int, default=128)
    parser.add_argument("--qwen-heads", type=int, default=4)
    parser.add_argument("--qwen-kv-heads", type=int, default=None)
    parser.add_argument("--qwen-layers", type=int, default=2)
    parser.add_argument("--qwen-intermediate-size", type=int, default=256)
    parser.add_argument(
        "--qwen-dataset",
        choices=("synthetic", "llava-instruct"),
        default="synthetic",
        help="Qwen dataset source. llava-instruct uses LLaVA-Instruct-150K text conversations.",
    )
    parser.add_argument(
        "--qwen-download-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the LLaVA-Instruct-150K JSON into --data-dir when missing.",
    )
    parser.add_argument(
        "--sigusr1-pause",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Coordinate a checkpoint/snapshot pause when any rank receives SIGUSR1",
    )
    parser.add_argument(
        "--pause-runtime-dir",
        default=None,
        help="Runtime directory for per-rank snapshot.checkpoint pause images",
    )
    parser.add_argument("--pause-snapshot-name", default="sigusr1")
    parser.add_argument("--pause-barrier-timeout", type=float, default=60.0)
    parser.add_argument(
        "--pause-snapshotd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use snapshotd for CRIU dump/restore in the SIGUSR1 pause path",
    )
    parser.add_argument("--pause-criu-bin", default="criu")
    parser.add_argument("--pause-criu-ns-bin", default=None)
    parser.add_argument("--pause-criu-log-level", type=int, default=None)
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


class SyntheticTokenDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, *, size: int, seq_len: int, vocab_size: int, seed: int) -> None:
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + index)
        input_ids = torch.randint(
            1,
            self.vocab_size,
            (self.seq_len,),
            generator=generator,
            dtype=torch.long,
        )
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


class LLaVAInstructDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        data_path: str | Path,
        *,
        seq_len: int,
        vocab_size: int,
        max_samples: int | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        with Path(data_path).open(encoding="utf-8") as stream:
            raw_data = json.load(stream)

        self.samples: list[list[dict[str, str]]] = []
        for item in raw_data:
            if "image" in item:
                continue
            conversations = item.get("conversations", [])
            if len(conversations) >= 2:
                self.samples.append(conversations)
            if max_samples and len(self.samples) >= max_samples:
                break

        if len(self.samples) < 1000:
            for item in raw_data:
                if "image" not in item:
                    continue
                conversations = item.get("conversations", [])
                if len(conversations) >= 2:
                    self.samples.append(conversations)
                if max_samples and len(self.samples) >= max_samples:
                    break

        if not self.samples:
            raise ValueError(f"No usable conversations found in {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        text = self.format_conversation(self.samples[index])
        token_ids = self.encode_text(text)
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == 0] = -100
        return {"input_ids": input_ids, "labels": labels}

    def encode_text(self, text: str) -> list[int]:
        if self.vocab_size < 2:
            raise ValueError("qwen vocab size must be at least 2")
        token_ids = [byte % (self.vocab_size - 1) + 1 for byte in text.encode("utf-8")][
            : self.seq_len
        ]
        if len(token_ids) < self.seq_len:
            token_ids.extend([0] * (self.seq_len - len(token_ids)))
        return token_ids

    @staticmethod
    def format_conversation(conversations: list[dict[str, str]]) -> str:
        text = ""
        for turn in conversations:
            role = turn.get("from", "")
            value = turn.get("value", "")
            if role == "human":
                text += f"<|im_start|>user\n{value}<|im_end|>\n"
            elif role == "gpt":
                text += f"<|im_start|>assistant\n{value}<|im_end|>\n"
        return text


def seed_everything(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch_cuda_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)


def build_lenet_components(
    args: argparse.Namespace,
    context: DistributedContext,
) -> tuple[torch.nn.Module, DataLoader, object | None, torch.optim.Optimizer, object]:
    from torchvision import datasets, transforms

    if context.enabled:
        import torch.distributed as dist
        from torch.utils.data.distributed import DistributedSampler

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
        sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
        shuffle = False
    else:
        dataset = datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=transform
        )
        sampler = None
        shuffle = True

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.data_workers,
        worker_init_fn=seed_worker if args.data_workers else None,
        generator=generator,
    )
    model = context.wrap_model(LeNet())
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, loader, sampler, optimizer, supervised_loss_fn(criterion)


def build_qwen_components(
    args: argparse.Namespace,
    context: DistributedContext,
) -> tuple[torch.nn.Module, DataLoader, object | None, torch.optim.Optimizer, object]:
    if context.enabled:
        from torch.utils.data.distributed import DistributedSampler

    config = QwenConfig(
        vocab_size=args.qwen_vocab_size,
        seq_len=args.qwen_seq_len,
        hidden_size=args.qwen_hidden_size,
        num_heads=args.qwen_heads,
        num_key_value_heads=args.qwen_kv_heads or args.qwen_heads,
        num_layers=args.qwen_layers,
        intermediate_size=args.qwen_intermediate_size,
    )
    dataset = build_qwen_dataset(args, config, context)
    if context.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.data_workers,
        worker_init_fn=seed_worker if args.data_workers else None,
        generator=generator,
    )
    model = context.wrap_model(QwenForCausalLM(config))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    return model, loader, sampler, optimizer, causal_lm_loss_fn()


def build_qwen_dataset(
    args: argparse.Namespace,
    config: QwenConfig,
    context: DistributedContext,
) -> Dataset[dict[str, torch.Tensor]]:
    if args.qwen_dataset == "synthetic":
        return SyntheticTokenDataset(
            size=args.train_samples,
            seq_len=config.seq_len,
            vocab_size=config.vocab_size,
            seed=args.seed,
        )

    data_path = llava_instruct_data_path(args.data_dir)
    if context.enabled:
        if context.is_primary and args.qwen_download_dataset:
            download_llava_instruct_dataset(args.data_dir)
        context.barrier()
    elif args.qwen_download_dataset:
        download_llava_instruct_dataset(args.data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist. Re-run with --qwen-download-dataset or "
            "place LLaVA-Instruct-150K there."
        )
    return LLaVAInstructDataset(
        data_path,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        max_samples=args.train_samples,
    )


def llava_instruct_data_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / LLAVA_INSTRUCT_FILENAME


def download_llava_instruct_dataset(data_dir: str | Path) -> Path:
    os.makedirs(data_dir, exist_ok=True)
    target = llava_instruct_data_path(data_dir)
    if target.exists():
        print(f"Dataset already exists at {target}")
        return target

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Using --qwen-dataset llava-instruct requires huggingface_hub. "
            "Install requirements.txt or place the JSON in --data-dir."
        ) from exc

    print(f"Downloading {LLAVA_INSTRUCT_REPO} to {target}")
    os.environ.pop("HF_HUB_OFFLINE", None)
    downloaded = hf_hub_download(
        repo_id=LLAVA_INSTRUCT_REPO,
        filename=LLAVA_INSTRUCT_FILENAME,
        repo_type="dataset",
        local_dir=str(data_dir),
    )
    return Path(downloaded)


def main() -> None:
    args = parse_args()
    context = DistributedContext.initialize()
    seed_everything(args.seed, deterministic=args.deterministic)

    if args.model == "lenet":
        model, loader, sampler, optimizer, loss_fn = build_lenet_components(
            args, context
        )
    else:
        model, loader, sampler, optimizer, loss_fn = build_qwen_components(
            args, context
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
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
            pause_on_sigusr1=args.sigusr1_pause,
            pause_runtime_dir=args.pause_runtime_dir,
            pause_snapshot_name=args.pause_snapshot_name,
            pause_barrier_timeout=args.pause_barrier_timeout,
            pause_use_snapshotd=args.pause_snapshotd,
            pause_criu_bin=args.pause_criu_bin,
            pause_criu_ns_bin=args.pause_criu_ns_bin,
            pause_criu_log_level=args.pause_criu_log_level,
        ),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
