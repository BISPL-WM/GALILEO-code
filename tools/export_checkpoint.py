"""Training checkpoint (model + optimizer + scheduler) -> model weights only (.safetensors), as released.

    python tools/export_checkpoint.py runs/rayrope_galileo/ckpts/step-000020000.pt checkpoints/rayrope_galileo.safetensors
"""
import sys

import torch
from safetensors.torch import save_file


def main():
    src, dst = sys.argv[1], sys.argv[2]
    sd = torch.load(src, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v.contiguous() for k, v in sd.items()}
    save_file(sd, dst)
    print(f"{dst}: {len(sd)} tensors, {sum(v.numel() for v in sd.values()) / 1e6:.2f}M parameters")


if __name__ == "__main__":
    main()
