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
torchrun --nproc_per_node=2 tools/train_ddp.py --epochs 5 --batch-size 64 --save-path ./lenet_cifar10_ddp.pt
```

## Snapshotd CUDA-Mapped DDP Example

```bash
./scripts/run_cuda_mapped_ddp_snapshot.sh --nproc-per-node 2 --cuda-visible-devices 0,1
```

This snapshots `tools/train_ddp.py` after CUDA warmup on the source GPU, then
uses `tools/xdtrain.py` to restore one NCCL DDP rank per target GPU.
`xdtrain.py` launches `tools/xtrain.py` for each rank and owns the DDP rank
environment plus the CUDA device map passed through snapshot restore.

Note: restore-time `cudaDeviceReset()` was tested as a possible follow-up for
refreshing CUDA device identity after remap, but it is not part of this example:
it is destructive to restored GPU allocations and did not resolve the NCCL/IB
identity issue in the current stack.

## Inference

With a custom image:

```bash
python tools/inference.py --checkpoint ./lenet_cifar10.pt --image /path/to/image.png
```

Or use a CIFAR-10 test image by index (downloads test split automatically if needed):

```bash
python tools/inference.py --checkpoint ./lenet_cifar10.pt --index 0
```
