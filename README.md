# simple_trainer

Simple PyTorch LeNet trainer/inference example using CIFAR-10.

## Files

- `models/letnet.py` - LeNet model definition.
- `tools/trainer.py` - Reusable training loop, checkpoint, and DDP helpers.
- `tools/train.py` - CIFAR-10/LeNet training entrypoint.
- `tools/inference.py` - Inference from a checkpoint.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

CIFAR-10 is downloaded automatically to `--data-dir` on first run.

## Train (single process)

```bash
./run_trainer.sh --epochs 5 --batch-size 64 --save-path ./lenet_cifar10.pt
```

(Equivalent direct command: `python tools/train.py ...`)

## Train (DDP)

```bash
torchrun --nproc-per-node=2 scripts/distributed_launcher.py tools/train.py --epochs 5 --batch-size 64
```

`tools/train.py` flips into DDP mode automatically when torchrun's env vars
(`RANK`/`WORLD_SIZE`/`LOCAL_RANK`) are present.
`scripts/distributed_launcher.py` narrows `CUDA_VISIBLE_DEVICES` to the per-rank
device before exec'ing the inner script.

`./lenet.sh --ddp=2` is the same launch through the snapshot-aware xtrain
wrapper.

## Inference

With a custom image:

```bash
python tools/inference.py --checkpoint ./lenet_cifar10.pt --image /path/to/image.png
```

Or use a CIFAR-10 test image by index (downloads test split automatically if needed):

```bash
python tools/inference.py --checkpoint ./lenet_cifar10.pt --index 0
```
