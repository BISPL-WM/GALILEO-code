"""Fetch the camera annotations of Stereo4D clips (gs://stereo4d, public) without the 3D tracks.

Each annotation file gs://stereo4d/<split>/<clip>.npz also holds large track arrays; this script reads only the
camera members with HTTP range requests and writes <out>/<split>/<clip>.npz with
    name, video_id, timestamps (us, one per video frame), camera2world (T, 3, 4: rectified left camera, metric),
    rectified2rig (2, 3, 3), fov_bounds.

    python scripts/prepare_stereo4d/fetch_cameras.py --clips splits/stereo4d/test_clips.json --split test \
        --out $STEREO4D_RAW/cams
"""
import argparse
import io
import json
import os
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

URL = "https://storage.googleapis.com/stereo4d/{split}/{clip}.npz"
MEMBERS = ("name", "video_id", "timestamps", "camera2world", "rectified2rig", "fov_bounds")


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url, retries=5):
        self.url, self.pos, self.retries = url, 0, retries
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off, io.SEEK_END: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def fetch(split, clip, out_dir):
    dst = os.path.join(out_dir, split, f"{clip}.npz")
    if os.path.exists(dst):
        return "skip"
    with zipfile.ZipFile(HTTPRangeFile(URL.format(split=split, clip=clip))) as z:
        arrays = {m: np.load(io.BytesIO(z.read(f"{m}.npy")), allow_pickle=False) for m in MEMBERS}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez(dst + ".tmp.npz", **arrays)
    os.replace(dst + ".tmp.npz", dst)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="json list of {clip, ...} (splits/stereo4d/*_clips.json)")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--first_n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    clips = [c["clip"] if isinstance(c, dict) else c for c in json.load(open(args.clips))]
    if args.first_n:
        clips = clips[: args.first_n]
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda c: fetch(args.split, c, args.out), clips))
    print({k: res.count(k) for k in set(res)}, "of", len(clips))


if __name__ == "__main__":
    main()
