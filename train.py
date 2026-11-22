"""Train on Stereo4D.

    torchrun --standalone --nproc-per-node=<gpus> train.py --config configs/stereo4d/rayrope_galileo.yaml \
        data_dir=$STEREO4D_DIR output_dir=runs/rayrope_galileo [key.sub=value ...]

The paper's models: 20k steps, batch 12 per GPU (one GPU per run in the paper), AdamW 1e-4, cosine schedule.
"""
import argparse
import warnings

from galileo import config_utils
from galileo.trainer import TrainConfig, Trainer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key.sub=value")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics")
    Trainer(config_utils.load(TrainConfig, args.config, args.overrides)).train()


if __name__ == "__main__":
    main()
