"""Stereo4D clips -> one W-frame window per clip at 256 x 256, both eyes, with metric stereo depth.

Per clip <proc>/<split>/<clip>/data.npz (written last; clips that have it are skipped) and meta.json:
    rgb    (2, W, 256, 256, 3) uint8   eye 0 = left, 1 = right; the 512 rectified frames, area-resized
    depth  (2, W, 256, 256) float16    metric depth from the left-right disparity (RAFT-large on the 512 frames, both
                                       directions): z = fx b / d, b = 0.063 m; invalid (0) where left-right inconsistent
                                       > 1 px, vertical flow > 1 px or disparity < 0.25 px at 512; 2 x 2 mean of the
                                       valid values, kept with >= 3 of 4 valid
    K      (3, 3) float32              fx = fy = 128 / tan(30 deg), principal point (128, 128)
    c2w    (2, W, 4, 4) float32        OpenCV camera to world, metric: left = the annotation's camera2world, right =
                                       left moved by b along its own x axis
    times  (W,) float32                seconds from the window's first frame (annotation timestamps)
    meta.json: clip, video_id, split, n_frames, f0 (window start), W

    python scripts/prepare_stereo4d/preprocess.py --raw $STEREO4D_RAW --proc $STEREO4D_DIR --split test \
        --clips splits/stereo4d/test_clips.json
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import BASE, EYES, FX, RES, SRC, atomic_savez, autocast, load_clips, load_raft, video_path, window_start


class Clips(torch.utils.data.Dataset):
    def __init__(self, rows, raw, split, W):
        self.rows, self.raw, self.split, self.W = rows, raw, split, W

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        clip, W = r["clip"], self.W
        f0 = window_start(clip, r["n_frames"], W)
        rgb = np.empty((2, W, RES, RES, 3), np.uint8)
        full = np.empty((2, W, SRC, SRC, 3), np.uint8)
        for e, eye in enumerate(EYES):
            cap = cv2.VideoCapture(video_path(self.raw, eye, self.split, clip))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            for k in range(W):
                ok, fr = cap.read()
                if not ok:
                    return {"clip": clip, "error": f"{eye} frame {f0 + k} unreadable"}
                full[e, k] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                rgb[e, k] = cv2.resize(full[e, k], (RES, RES), interpolation=cv2.INTER_AREA)
            cap.release()
        cam = np.load(os.path.join(self.raw, "cams", self.split, f"{clip}.npz"))
        c2w_l = np.tile(np.eye(4), (W, 1, 1))
        c2w_l[:, :3] = cam["camera2world"][f0:f0 + W]
        c2w_r = c2w_l.copy()
        c2w_r[:, :3, 3] += BASE * c2w_l[:, :3, 0]
        ts = cam["timestamps"][f0:f0 + W].astype(np.float64)
        return {"clip": clip, "row": r, "f0": f0, "rgb": rgb, "full": full,
                "c2w": np.stack([c2w_l, c2w_r]).astype(np.float32), "times": ((ts - ts[0]) / 1e6).astype(np.float32)}


@torch.inference_mode()
def stereo_depth(raft, rgb, device, iters, chunk):
    """rgb (2, W, R, R, 3) uint8 (R = 512) -> depth (2, W, R, R) float32 (0 invalid)."""
    fx = FX * rgb.shape[2] / RES
    W = rgb.shape[1]
    x = torch.from_numpy(rgb).to(device).permute(0, 1, 4, 2, 3).float() / 127.5 - 1.0  # (2, W, 3, R, R)
    flows = {}
    for a, b in ((0, 1), (1, 0)):
        out = []
        for c0 in range(0, W, chunk):
            with autocast(device):
                out.append(raft(x[a, c0:c0 + chunk], x[b, c0:c0 + chunk], num_flow_updates=iters)[-1].float())
        flows[(a, b)] = torch.cat(out)  # (W, 2, R, R): pixel in a -> pixel in b
    R = rgb.shape[2]
    ys, xs = torch.meshgrid(torch.arange(R, device=device, dtype=torch.float32),
                            torch.arange(R, device=device, dtype=torch.float32), indexing="ij")
    depth = torch.zeros(2, W, R, R, device=device)
    for a, b, sign in ((0, 1, -1.0), (1, 0, 1.0)):  # left: x_r = x_l - d; right: x_l = x_r + d
        f, g = flows[(a, b)], flows[(b, a)]
        disp = sign * f[:, 0]
        xt, yt = xs + f[:, 0], ys + f[:, 1]
        grid = torch.stack([(xt + 0.5) / R * 2 - 1, (yt + 0.5) / R * 2 - 1], -1)
        back = F.grid_sample(g, grid, mode="bilinear", padding_mode="zeros", align_corners=False)  # (W, 2, R, R)
        lr = (f + back).norm(dim=1)
        ok = (lr < 1.0) & (f[:, 1].abs() < 1.0) & (disp > 0.25) & (xt >= 0) & (xt <= R - 1)
        depth[a] = torch.where(ok, fx * BASE / disp.clamp_min(1e-3), torch.zeros_like(disp))
    return depth


def depth_256(depth):
    """(2, W, 512, 512) -> (2, W, 256, 256): mean of the valid ones of each 2 x 2 block, kept with >= 3 valid."""
    E, W, H, _ = depth.shape
    b = depth.reshape(E, W, H // 2, 2, H // 2, 2)
    v = (b > 0).float()
    n = v.sum((3, 5))
    return torch.where(n >= 3, (b * v).sum((3, 5)) / n.clamp_min(1), torch.zeros_like(n)).cpu().numpy()


def _as_is(x):
    return x


def _worker_init(_):
    cv2.setNumThreads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder with mp4/ and cams/")
    ap.add_argument("--proc", required=True, help="output folder")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--clips", required=True, help="clip list json (splits/stereo4d/{train,test}_clips.json)")
    ap.add_argument("--W", type=int, default=30, help="window length (frames)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--iters", type=int, default=12, help="RAFT updates")
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--weights_cache", default=None, help="torch hub folder for the RAFT weights")
    args = ap.parse_args()
    torch.set_num_threads(max(4, torch.get_num_threads() if args.device == "cpu" else 4))
    _, rows = load_clips(args.clips)
    rows = [r for r in rows if r["n_frames"] >= args.W][args.shard::args.n_shards]
    rows = [r for r in rows if not os.path.exists(os.path.join(args.proc, args.split, r["clip"], "data.npz"))]
    print(f"{args.split} shard {args.shard}/{args.n_shards}: {len(rows)} clips to do", flush=True)
    raft = load_raft(args.device, args.weights_cache)
    loader = torch.utils.data.DataLoader(Clips(rows, args.raw, args.split, args.W), batch_size=None,
                                         num_workers=args.workers, prefetch_factor=4 if args.workers else None,
                                         collate_fn=_as_is, worker_init_fn=_worker_init)
    t0, n_err = time.time(), 0
    K = np.array([[FX, 0, RES / 2], [0, FX, RES / 2], [0, 0, 1]], np.float32)
    for n, it in enumerate(loader):
        if "error" in it:
            n_err += 1
            print(f"skip {it['clip']}: {it['error']}", flush=True)
            continue
        depth = depth_256(stereo_depth(raft, it["full"], args.device, args.iters, args.chunk))
        d = os.path.join(args.proc, args.split, it["clip"])
        os.makedirs(d, exist_ok=True)
        atomic_savez(os.path.join(d, "data.npz"), rgb=it["rgb"], depth=depth.astype(np.float16), K=K, c2w=it["c2w"],
                     times=it["times"])
        r = it["row"]
        json.dump({"clip": it["clip"], "video_id": r["video_id"], "split": args.split, "n_frames": r["n_frames"],
                   "f0": it["f0"], "W": args.W}, open(os.path.join(d, "meta.json"), "w"))
        if (n + 1) % 200 == 0 or n + 1 == len(rows):
            el = time.time() - t0
            print(f"[{args.split} {args.shard}] {n + 1}/{len(rows)} {el / (n + 1):.2f} s/clip, errors {n_err}", flush=True)


if __name__ == "__main__":
    main()
