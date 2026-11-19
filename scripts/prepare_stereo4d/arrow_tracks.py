"""ARROW (github.com/ifradlin/arrow) tracks between the fixed input times of each processed Stereo4D clip, both eyes.
They are the pseudo-ground-truth correspondences and velocities of training (match loss, curriculum) and fix the
clip's input frames (`inputs`) for training and evaluation. Run in an environment with the `arrow` package:

    python scripts/prepare_stereo4d/arrow_tracks.py --raw $STEREO4D_RAW --proc $STEREO4D_DIR --split test

The input times are n_in frames gap apart, starting at a per-clip deterministic offset inside the window. Per eye,
those frames of the 512 video are encoded at 504 (ARROW's 14 px patches) and every token centre (32 x 32 grid of the
256 frame) of every input frame is queried at every input time, reference camera = its own frame. Writes
data_arrow.npz:
    inputs (n_in,) int16              window frame index of the input times
    pts    (2, n_in, n_in, P, 3) f32  token (eye, s, p) at input time t, in ARROW's camera s (its own scale and focal)
    uv     (2, n_in, n_in, P, 2) f16  where frame t observes it, 256-frame pixels (pixel i spans [i, i + 1))
    vis    (2, n_in, n_in, P) f16     visibility (sigmoid)
    conf   (2, n_in, n_in, P) f16     point confidence
"""
import argparse
import json
import os
import time

import av
import numpy as np
import torch
from PIL import Image

from common import EYES, RES, atomic_savez, input_start, video_path

TOK, SIDE = 8, 504


def read_frames(path, start, n):
    out = []
    with av.open(path) as c:
        for k, fr in enumerate(c.decode(video=0)):
            if k >= start:
                out.append(np.asarray(fr.to_image().resize((SIDE, SIDE), Image.Resampling.BOX)))
            if len(out) == n:
                break
    if len(out) < n:
        raise RuntimeError(f"{path}: {len(out)} of {n} frames from {start}")
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0


@torch.inference_mode()
def run_eye(model, images, src_idx, device, chunk):
    from arrow import Queries
    n = len(src_idx)
    g = RES // TOK
    idx = torch.arange(g * g)
    cen = torch.stack([(idx % g) * TOK + TOK / 2, (idx // g) * TOK + TOK / 2], -1).float()
    xy = cen * SIDE / RES - 0.5  # ARROW pixel centres are integers
    P = xy.shape[0]
    enc = model.encode(images[None].to(device))
    si = torch.tensor(src_idx)
    s = si.repeat_interleave(n * P)
    t = si.repeat_interleave(P).repeat(n)
    q = Queries(xy.repeat(n * n, 1)[None], s[None], t[None], s[None])
    pr = model.decode(enc, q, decode_chunk_size=chunk)
    shp = (n, n, P)
    return (pr.points[0].float().reshape(*shp, 3).cpu().numpy(), (pr.uv[0].float() * RES).reshape(*shp, 2).cpu().numpy(),
            torch.sigmoid(pr.visibility_logits[0].float()).reshape(shp).cpu().numpy(),
            pr.point_confidence[0].float().reshape(shp).cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--proc", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--clips", help="clip list json (default: every processed clip of the split)")
    ap.add_argument("--n_in", type=int, default=4)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from arrow import ARROW
    base = os.path.join(args.proc, args.split)
    names = ([c["clip"] if isinstance(c, dict) else c for c in json.load(open(args.clips))] if args.clips
             else os.listdir(base))
    clips = sorted(c for c in names if os.path.exists(os.path.join(base, c, "data.npz")))[args.shard::args.n_shards]
    todo = [c for c in clips if not os.path.exists(os.path.join(base, c, "data_arrow.npz"))]
    print(f"{args.split} shard {args.shard}/{args.n_shards}: {len(todo)} of {len(clips)} clips to do", flush=True)
    model = ARROW.from_pretrained("ifradlin/arrow", device=torch.device(args.device))
    t0 = time.time()
    for i, clip in enumerate(todo):
        d = os.path.join(base, clip)
        meta = json.load(open(os.path.join(d, "meta.json")))
        s0 = input_start(clip, meta["W"], args.n_in, args.gap)
        span = (args.n_in - 1) * args.gap + 1
        src = [k * args.gap for k in range(args.n_in)]
        res = {k: [] for k in ("pts", "uv", "vis", "conf")}
        for eye in EYES:
            imgs = read_frames(video_path(args.raw, eye, args.split, clip), meta["f0"] + s0, span)
            for k, v in zip(res, run_eye(model, imgs, src, args.device, args.chunk)):
                res[k].append(v)
        atomic_savez(os.path.join(d, "data_arrow.npz"), inputs=np.array([s0 + k for k in src], np.int16),
                     pts=np.stack(res["pts"]).astype(np.float32), uv=np.stack(res["uv"]).astype(np.float16),
                     vis=np.stack(res["vis"]).astype(np.float16), conf=np.stack(res["conf"]).astype(np.float16))
        if (i + 1) % 200 == 0 or i + 1 == len(todo):
            print(f"[{i + 1}/{len(todo)}] {(time.time() - t0) / (i + 1):.2f} s/clip", flush=True)


if __name__ == "__main__":
    main()
