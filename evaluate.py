"""Evaluate a checkpoint on the Stereo4D test set (600 clips; interpolation and extrapolation targets).

    torchrun --standalone --nproc-per-node=1 evaluate.py --config configs/stereo4d/rayrope_galileo.yaml \
        --ckpt checkpoints/rayrope_galileo.safetensors data_dir=$STEREO4D_DIR output_dir=runs/eval_rayrope_galileo

Writes <output_dir>/tests/metrics_dyn_stereo4d_{interp,extrap}.json (PSNR, SSIM, LPIPS of the whole image, PSNR /
SSIM / LPIPS of the dynamic and static regions) and per_entry_*.json (one value per test entry, single process).
The paper averages interpolation and extrapolation. --motion_off: GALILEO's keys stay at their capture time (Tab. 2).
"""
import argparse
import json
import warnings

from galileo import config_utils
from galileo.trainer import TrainConfig, Trainer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--motion_off", action="store_true")
    ap.add_argument("overrides", nargs="*", help="key.sub=value")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics")
    cfg = config_utils.load(TrainConfig, args.config, args.overrides + [f"resume={args.ckpt}"])
    trainer = Trainer(cfg)
    if args.motion_off:
        build = trainer.build_model

        def build_off():
            model = build()
            model.attention.transport = False
            return model
        trainer.build_model = build_off
    res = trainer.test()
    if trainer.world_rank == 0:
        keys = ("psnr_dyn", "psnr", "psnr_static", "ssim", "lpips", "ssim_dyn", "lpips_dyn")
        mean = {k: sum(r[k] for r in res.values()) / len(res) for k in keys}
        print(json.dumps({"mean_of_protocols": mean, **res}, indent=1))


if __name__ == "__main__":
    main()
