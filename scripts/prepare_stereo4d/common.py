"""Shared constants and helpers of the Stereo4D preparation scripts.

Raw data layout (<raw>): mp4/{left,right}/{train,test}/<clip>-{left,right}_rectified.mp4 (download_videos.py) and
cams/{train,test}/<clip>.npz (fetch_cameras.py). Processed layout (<proc>): <split>/<clip>/{data.npz, meta.json,
data_arrow.npz, data_evalmask.npz}.
"""
import contextlib
import hashlib
import json
import math
import os

import numpy as np
import torch

RES, SRC, BASE = 256, 512, 0.063  # processed / rectified resolution, stereo baseline (m)
FX = (RES / 2) / math.tan(math.radians(30.0))  # 60 deg horizontal FoV at RES
EYES = ("left", "right")


def video_path(raw, eye, split, clip):
    return os.path.join(raw, "mp4", eye, split, f"{clip}-{eye}_rectified.mp4")


def load_clips(path):
    """A clip list json: [{"clip", "n_frames", "video_id"}, ...] or [clip, ...] -> list of names (and rows)."""
    rows = json.load(open(path))
    return [r["clip"] if isinstance(r, dict) else r for r in rows], rows


def window_start(clip, n_frames, W):
    """First frame of the clip's W-frame window, deterministic per clip (md5 of its name)."""
    rng = np.random.default_rng(int(hashlib.md5(clip.encode()).hexdigest()[:8], 16))
    return int(rng.integers(0, n_frames - W + 1))


def input_start(clip, W, n_in, gap):
    """Window index of the first input time, deterministic per clip; the targets need max(1, gap // 2) frames after
    the last input."""
    ext = max(1, gap // 2)
    hi = W - 1 - (n_in - 1) * gap - ext
    rng = np.random.default_rng(int(hashlib.md5((clip + "/inputs").encode()).hexdigest()[:8], 16))
    return int(rng.integers(0, hi + 1))


def load_raft(device, cache=None):
    """torchvision RAFT-large (C+T+S+K+H, v2 weights)."""
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    sd = torch.hub.load_state_dict_from_url(Raft_Large_Weights.DEFAULT.url, model_dir=cache, map_location="cpu",
                                            progress=False)
    raft = raft_large(weights=None)
    raft.load_state_dict(sd)
    return raft.eval().requires_grad_(False).to(device)


def autocast(device):
    """bf16 autocast on CUDA (as used for the released data); full precision on the CPU."""
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def atomic_savez(path, **arrays):
    tmp = f"{path[:-4]}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
