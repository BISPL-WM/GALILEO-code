# GALILEO: Geometric RoPE with Galilean Key Transport for Dynamic Scenes

Youngmin Kim\*, Seonghyun Jin\*, Jong Chul Ye (KAIST) · [Project page](https://bispl-wm.github.io/GALILEO/) · arXiv (coming soon)

GALILEO makes geometric positional encodings (RayRoPE, URoPE) work for dynamic scenes: every input token gets a
constant 3D velocity from in-model correspondences, and each key is moved to the query's time before its 3D
positional encoding is applied. This repository contains the code, configs and checkpoints of the **Stereo4D**
experiments of the paper (novel-time view synthesis from two stereo cameras at four times), including all baselines.

## Installation

```bash
git clone <this repository> galileo && cd galileo
pip install -r requirements.txt
```

## Data (Stereo4D)

The scripts download the public videos and annotations and rebuild the processed data used in the paper; no
derived data is redistributed. Clip lists: `splits/stereo4d/train_clips.json` (15,000 training clips),
`test_clips.json` (600 test clips) and `test_index.json` (the 1,200 test entries, interpolation and extrapolation).

```bash
export STEREO4D_RAW=/path/to/stereo4d_raw STEREO4D_DIR=/path/to/stereo4d_proc
cd scripts/prepare_stereo4d
for s in train test; do
  # 1. rectified 512x512 videos (Hugging Face, ~10 GB per eye for test, ~100 GB per eye for train)
  python download_videos.py --raw $STEREO4D_RAW --split $s --clips ../../splits/stereo4d/${s}_clips.json
  # 2. camera annotations (gs://stereo4d, camera members only)
  python fetch_cameras.py --raw $STEREO4D_RAW --split $s --clips ../../splits/stereo4d/${s}_clips.json
  # 3. one 30-frame window per clip at 256x256 with metric stereo depth (RAFT, GPU)
  python preprocess.py --raw $STEREO4D_RAW --proc $STEREO4D_DIR --split $s --clips ../../splits/stereo4d/${s}_clips.json
  # 4. ARROW tracks between the input times (in an environment with ARROW, github.com/ifradlin/arrow)
  python arrow_tracks.py --raw $STEREO4D_RAW --proc $STEREO4D_DIR --split $s
  # 5. RAFT token tracks (only for the "RAFT flow" variant)
  python raft_tracks.py --raw $STEREO4D_RAW --proc $STEREO4D_DIR --split $s
done
# 6. moving-pixel masks of the test clips (the dynamic region of PSNR_dyn)
python eval_masks.py --proc $STEREO4D_DIR --split test
```

Every script skips clips that are already done and can be sharded (`--shard i --n_shards n`).

## Checkpoints

All models: 6-layer decoder-only LVSM (12 layers where noted), 256x256, 20k steps, batch 12.
PSNR is the mean of interpolation and extrapolation (paper Tab. 1 and supp.).

| Config (`configs/stereo4d/`) | Paper | PSNR_dyn | PSNR |
|---|---|---|---|
| `plucker.yaml` | Plücker | 23.59 | 29.17 |
| `prope.yaml` | PRoPE | 24.06 | 29.95 |
| `gta.yaml` | GTA | 24.15 | 30.07 |
| `urope.yaml` | URoPE | 24.61 | 30.73 |
| `urope_galileo.yaml` | URoPE + GALILEO | 25.41 | 31.00 |
| `rayrope.yaml` | RayRoPE | 24.61 | 31.04 |
| `rayrope_galileo.yaml` | RayRoPE + GALILEO | **25.78** | 31.34 |
| `rayrope_galileo.yaml` + `--motion_off` | motion off at test (Tab. 2) | 24.47 | 30.95 |
| `rayrope_galileo_raft.yaml` | + GALILEO, RAFT flow (supp.) | 25.60 | 31.43 |
| `rayrope_L12.yaml` | RayRoPE, 12 layers (supp.) | 25.56 | 32.14 |
| `rayrope_galileo_L12.yaml` | RayRoPE + GALILEO, 12 layers (supp.) | 26.39 | 32.26 |

Download: (link coming soon) into `checkpoints/<config name>.safetensors`.

## Evaluation

```bash
python evaluate.py --config configs/stereo4d/rayrope_galileo.yaml --ckpt checkpoints/rayrope_galileo.safetensors \
    data_dir=$STEREO4D_DIR output_dir=runs/eval/rayrope_galileo
```

Prints the metrics per protocol and their mean; writes `tests/metrics_*.json` and per-entry values
(`tests/per_entry_*.json`). Multi-GPU: `torchrun --standalone --nproc-per-node=<n> evaluate.py ...`.

## Training

```bash
torchrun --standalone --nproc-per-node=1 train.py --config configs/stereo4d/rayrope_galileo.yaml \
    data_dir=$STEREO4D_DIR output_dir=runs/rayrope_galileo
```

Any config field can be overridden with `key.sub=value` (see `galileo/trainer.py: TrainConfig` and
`galileo/models/lvsm.py: ModelConfig, GalileoConfig`). `tools/export_checkpoint.py` strips a training checkpoint to
its weights.

## Code map

- `galileo/motion.py`: GALILEO itself: matching (coarse window + fine CNN stage), velocity fit, key displacement.
- `galileo/pos_enc/rayrope.py`: RayRoPE with temporal RoPE; `_motion_from_corr` / `_dynamic_kv_coeffs` apply GALILEO.
- `galileo/pos_enc/urope.py`, `prope.py`: URoPE (+ GALILEO), PRoPE / GTA.
- `galileo/models/lvsm.py`: the LVSM model and its configs; `galileo/trainer.py`: losses, training, evaluation.
- `galileo/data/stereo4d.py`: Stereo4D samples, ARROW pseudo-ground-truth motion.

## Citation

```bibtex
@article{kim2026galileo,
  title   = {GALILEO: Geometric RoPE with Galilean Key Transport for Dynamic Scenes},
  author  = {Kim, Youngmin and Jin, Seonghyun and Ye, Jong Chul},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`). Parts are adapted from RayRoPE, PRoPE and URoPE; see `THIRD_PARTY_NOTICES.md`.
