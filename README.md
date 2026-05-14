# simple_trainer

Simple PyTorch LeNet and tiny Qwen-style trainer/inference examples.

## Files

- `models/letnet.py` - LeNet model definition.
- `models/qwen.py` - compact Qwen-style causal LM for synthetic token training.
- `tools/trainer.py` - Reusable training loop, checkpoint, and DDP helpers.
- `tools/train.py` - LeNet/Qwen training entrypoint.
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

## Train Qwen

```bash
./qwen.sh --ddp=2 --data-workers=4
```

`qwen.sh` launches `tools/xtrain.py` with `--model qwen`, deterministic
synthetic token data, and SIGUSR1 pause support. To train from the same
open-source conversation dataset used by `../tiny-qwen`, pass
`--qwen-dataset llava-instruct`; it auto-downloads
LLaVA-Instruct-150K into `--data-dir` when missing.

From another shell:

```bash
./qwen.sh --pause --ddp=2
./qwen.sh --resume --ddp=2
```

The pause path all-gathers a per-rank SIGUSR1 flag after each optimizer step.
If any rank saw the signal, rank 0 writes the PyTorch checkpoint, all ranks
barrier, and each rank attempts a named runtime snapshot under
`.qwen_pause/rank-N/snapshots/qwen-sigusr1`. `--resume` restores those runtime
snapshots when present. If no complete runtime snapshot is available, it falls
back to `--init-from` the PyTorch checkpoint; checkpoints include the current
epoch and consumed batch count so the loader resumes at the same position.

## Inference

With a custom image:

```bash
python tools/inference.py --checkpoint ./lenet_cifar10.pt --image /path/to/image.png
```

Or use a CIFAR-10 test image by index (downloads test split automatically if needed):

```bash
python tools/inference.py --checkpoint ./lenet_cifar10.pt --index 0
```
