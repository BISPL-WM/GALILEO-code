"""RAFT token tracks between the input times of each processed Stereo4D clip, both eyes: the external matches of
the "RAFT flow" variant (GALILEO with RAFT in place of its matcher, supp. Tab. 10). Needs data_arrow.npz (its
`inputs` fix the input frames).

Default (as released): --chain --res 512 -> data_raftc512.npz. RAFT-large on the 512 rectified frames decoded from
the mp4 (a token's patch is 16 x 16 there); far pairs are chained through the adjacent-time flows (step 1 takes the
token's patch-median flow, later steps the dense flow at the current position), fb = the largest per-step
forward-backward error. Without --chain, every ordered pair of input times directly (landing = token centre +
median flow of its patch). Writes:
    inputs (n_in,) int16              window frame index of the input times (as data_arrow.npz)
    uv     (2, n_in, n_in, P, 2) f16  where frame t sees token (eye, s, p), 256 pixels (pixel i spans [i, i + 1));
                                      the token centre for t = s
    fb     (2, n_in, n_in, P) f16     forward-backward error (px), 0 for t = s

    python scripts/prepare_stereo4d/raft_tracks.py --raw $STEREO4D_RAW --proc $STEREO4D_DIR --split train
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import EYES, atomic_savez, autocast, load_raft, video_path

G = 32  # tokens per side (8 px patches of the 256 frames)


@torch.inference_mode()
def tracks(raft, frames, iters):
    """frames (n, 3, R, R) in [-1, 1] -> uv (n, n, P, 2), fb (n, n, P) (float32 numpy, 256 pixels)."""
    n, _, R, _ = frames.shape
    g, p = G, R // G
    pairs = [(s, t) for s in range(n) for t in range(n) if s != t]
    a = frames[[s for s, _ in pairs]]
    b = frames[[t for _, t in pairs]]
    with autocast(frames.device):
        flow = raft(a, b, num_flow_updates=iters)[-1].float()  # (pairs, 2, R, R)
    med = flow.reshape(len(pairs), 2, g, p, g, p).permute(0, 1, 2, 4, 3, 5).reshape(len(pairs), 2, g * g, -1)
    med = med.median(-1).values.permute(0, 2, 1)  # (pairs, P, 2)
    idx = torch.arange(g * g, device=frames.device)
    cen = torch.stack([(idx % g) * p + p / 2, (idx // g) * p + p / 2], -1).float()
    land = cen + med
    rev = {pq: i for i, pq in enumerate(pairs)}
    uv = cen.expand(n, n, -1, -1).clone()
    fb = torch.zeros(n, n, g * g, device=frames.device)
    for i, (s, t) in enumerate(pairs):
        grid = (land[i] / R * 2 - 1)[None, None]  # pixel i spans [i, i + 1) (align_corners=False convention)
        back = F.grid_sample(flow[rev[(t, s)]][None], grid, align_corners=False)[0, :, 0].T  # flow t -> s at the landing
        uv[s, t] = land[i]
        fb[s, t] = (land[i] + back - cen).norm(dim=-1)
    return (uv * 256 / R).cpu().numpy(), (fb * 256 / R).cpu().numpy()


@torch.inference_mode()
def chained_tracks(raft, frames, iters):
    """Far pairs by chaining the adjacent-time flows. frames (n, 3, R, R) -> uv (n, n, P, 2), fb (n, n, P)."""
    n, _, R, _ = frames.shape
    g, p = G, R // G
    pairs = [(k, k + 1) for k in range(n - 1)] + [(k + 1, k) for k in range(n - 1)]
    a = frames[[s for s, _ in pairs]]
    b = frames[[t for _, t in pairs]]
    with autocast(frames.device):
        flow = raft(a, b, num_flow_updates=iters)[-1].float()  # (pairs, 2, R, R)
    fl = {pq: flow[i] for i, pq in enumerate(pairs)}
    med = {pq: f.reshape(2, g, p, g, p).permute(0, 1, 3, 2, 4).reshape(2, g * g, -1).median(-1).values.T
           for pq, f in fl.items()}  # (P, 2)
    idx = torch.arange(g * g, device=frames.device)
    cen = torch.stack([(idx % g) * p + p / 2, (idx // g) * p + p / 2], -1).float()

    def samp(f, pos):
        return F.grid_sample(f[None], (pos / R * 2 - 1)[None, None], align_corners=False)[0, :, 0].T

    uv = cen.expand(n, n, -1, -1).clone()
    fb = torch.zeros(n, n, g * g, device=frames.device)
    for s in range(n):
        for step in (1, -1):
            pos, worst, k = cen.clone(), torch.zeros(g * g, device=frames.device), s
            while 0 <= k + step < n:
                nxt = pos + (med[(k, k + step)] if k == s else samp(fl[(k, k + step)], pos))
                worst = torch.maximum(worst, (nxt + samp(fl[(k + step, k)], nxt) - pos).norm(dim=-1))
                pos, k = nxt, k + step
                uv[s, k], fb[s, k] = pos, worst
    return (uv * 256 / R).cpu().numpy(), (fb * 256 / R).cpu().numpy()


class Frames(torch.utils.data.Dataset):
    """A clip's input frames (2, n, R, R, 3) uint8: the 256 ones of data.npz, or (R = 512) the rectified video frames
    of the clip's window (meta.json f0) decoded from the mp4."""

    def __init__(self, raw, base, split, clips, res):
        self.raw, self.base, self.split, self.clips, self.res = raw, base, split, clips, res

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        clip = self.clips[i]
        d = os.path.join(self.base, clip)
        inputs = np.load(os.path.join(d, "data_arrow.npz"))["inputs"].astype(np.int64)
        if self.res == 256:
            return clip, inputs, np.load(os.path.join(d, "data.npz"))["rgb"][:, inputs]
        f0 = json.load(open(os.path.join(d, "meta.json")))["f0"]
        rgb = np.empty((2, len(inputs), self.res, self.res, 3), np.uint8)
        for e, eye in enumerate(EYES):
            cap = cv2.VideoCapture(video_path(self.raw, eye, self.split, clip))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0 + int(inputs[0]))
            for k in range(int(inputs[0]), int(inputs[-1]) + 1):
                ok, fr = cap.read()
                assert ok, f"{clip} {eye} frame {f0 + k} unreadable"
                if k in inputs:
                    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    assert fr.shape[0] == self.res, (clip, fr.shape)
                    rgb[e, int(np.flatnonzero(inputs == k)[0])] = fr
            cap.release()
        return clip, inputs, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--proc", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--no_chain", action="store_true", help="every pair directly instead of chained -> data_raft*.npz")
    ap.add_argument("--res", type=int, default=512, choices=(256, 512), help="256: the frames of data.npz")
    ap.add_argument("--iters", type=int, default=12, help="RAFT updates")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--weights_cache", default=None)
    args = ap.parse_args()
    out_name = ("data_raft" if args.no_chain else "data_raftc") + ("" if args.res == 256 else str(args.res)) + ".npz"
    track = tracks if args.no_chain else chained_tracks
    raft = load_raft(args.device, args.weights_cache)
    base = os.path.join(args.proc, args.split)
    clips = sorted(c for c in os.listdir(base) if os.path.exists(os.path.join(base, c, "data_arrow.npz")))
    clips = clips[args.shard::args.n_shards]
    todo = [c for c in clips if not os.path.exists(os.path.join(base, c, out_name))]
    print(f"{args.split} shard {args.shard}/{args.n_shards}: {len(todo)} of {len(clips)} clips -> {out_name}", flush=True)
    loader = torch.utils.data.DataLoader(Frames(args.raw, base, args.split, todo, args.res), batch_size=None,
                                         num_workers=args.workers)
    t0 = time.time()
    for n_done, (clip, inputs, rgb) in enumerate(loader):
        uv, fb = [], []
        for e in range(rgb.shape[0]):
            fr = torch.as_tensor(rgb[e]).to(args.device).permute(0, 3, 1, 2).float() / 127.5 - 1
            u, f = track(raft, fr, args.iters)
            uv.append(u)
            fb.append(f)
        atomic_savez(os.path.join(base, clip, out_name), inputs=np.asarray(inputs).astype(np.int16),
                     uv=np.stack(uv).astype(np.float16), fb=np.stack(fb).astype(np.float16))
        if (n_done + 1) % 500 == 0 or n_done + 1 == len(todo):
            print(f"[{n_done + 1}/{len(todo)}] {(time.time() - t0) / (n_done + 1):.3f} s/clip", flush=True)


if __name__ == "__main__":
    main()
