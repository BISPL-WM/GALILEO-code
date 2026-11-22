"""Moving-pixel masks of the Stereo4D test clips (Stereo4D has no object masks): for every eye and every window
frame that is not an input, the pixels whose RAFT flow to the nearest input frame (same eye) differs from their
static reprojection (stereo depth of the frame, cameras) by more than --thresh px; pixels without stereo depth count
as static. These are the dynamic regions of PSNR_dyn. Writes data_evalmask.npz: moving (2, W, 256, 256) bool. The
inputs are the clip's fixed ones (data_arrow.npz).

    python scripts/prepare_stereo4d/eval_masks.py --proc $STEREO4D_DIR --split test
"""
import argparse
import os

import numpy as np
import torch

from common import atomic_savez, autocast, load_raft


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--thresh", type=float, default=1.5)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--weights_cache", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    raft = load_raft(dev, args.weights_cache)
    base = os.path.join(args.proc, args.split)
    clips = sorted(c for c in os.listdir(base) if os.path.exists(os.path.join(base, c, "data_arrow.npz")))
    for n, clip in enumerate(clips):
        d = os.path.join(base, clip)
        out = os.path.join(d, "data_evalmask.npz")
        if os.path.exists(out):
            continue
        z = np.load(os.path.join(d, "data.npz"))
        fr = [int(f) for f in np.load(os.path.join(d, "data_arrow.npz"))["inputs"]]
        rgb, depth, K, c2w = z["rgb"], z["depth"].astype(np.float64), z["K"].astype(np.float64), z["c2w"].astype(np.float64)
        E, W, R = rgb.shape[0], rgb.shape[1], rgb.shape[2]
        moving = np.zeros((E, W, R, R), bool)
        ys, xs = np.mgrid[0:R, 0:R].astype(np.float64)
        frames = [f for f in range(W) if f not in fr]
        for e in range(E):
            near = [min(fr, key=lambda x: abs(x - f)) for f in frames]
            a = torch.from_numpy(rgb[e, frames]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
            b = torch.from_numpy(rgb[e, near]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
            flows = []
            for c0 in range(0, len(frames), 16):
                with autocast(dev):
                    flows.append(raft(a[c0:c0 + 16], b[c0:c0 + 16], num_flow_updates=args.iters)[-1].float())
            flows = torch.cat(flows).cpu().numpy()  # (n, 2, R, R)
            for i, (f, nf) in enumerate(zip(frames, near)):
                dz = depth[e, f]
                ok = dz > 0
                xc = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(xs)],
                              -1) * dz[..., None]
                X = xc[ok] @ c2w[e, f, :3, :3].T + c2w[e, f, :3, 3]
                w2c = np.linalg.inv(c2w[e, nf])
                y = X @ w2c[:3, :3].T + w2c[:3, 3]
                u = K[0, 0] * y[:, 0] / np.maximum(y[:, 2], 1e-6) + K[0, 2] - 0.5
                v = K[1, 1] * y[:, 1] / np.maximum(y[:, 2], 1e-6) + K[1, 2] - 0.5
                mv = np.zeros((R, R), bool)
                mv[ok] = np.hypot(xs[ok] + flows[i, 0][ok] - u, ys[ok] + flows[i, 1][ok] - v) > args.thresh
                moving[e, f] = mv
        atomic_savez(out, moving=moving)
        if (n + 1) % 100 == 0 or n + 1 == len(clips):
            print(f"{n + 1}/{len(clips)}", flush=True)


if __name__ == "__main__":
    main()
