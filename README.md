# simple_trainer

Simple PyTorch LeNet trainer/inference example using CIFAR-10.

## Files

- `models/letnet.py` - LeNet model definition.
- `tools/train.py` - Single-process training.
- `tools/train_ddp.py` - DistributedDataParallel training.
- `tools/inference.py` - Inference from a checkpoint.

## Setup

Install dependencies:

```bash
pip install torch torchvision pillow
```

CIFAR-10 is downloaded automatically to `--data-dir` on first run.

## Train (single process)

```bash
python tools/train.py --epochs 5 --batch-size 64 --save-path ./lenet_cifar10.pt
```

## Train (DDP)

```bash
torchrun --nproc_per_node=2 tools/train_ddp.py --epochs 5 --batch-size 64 --save-path ./lenet_cifar10_ddp.pt
```

## Inference

With a custom image:

```bash
python tools/inference.py --checkpoint ./lenet_cifar10.pt --image /path/to/image.png
```

Or use a CIFAR-10 test image by index (downloads test split automatically if needed):

```bash
python tools/inference.py --checkpoint ./lenet_cifar10.pt --index 0
```
