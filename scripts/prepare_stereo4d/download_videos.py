"""Download the rectified Stereo4D videos (Hugging Face KevinMathew/stereo4d-{left,right}eye-perspective: the 512 x 512,
60 deg FoV perspective clips of stereo4d-code's rectify.py; non-commercial research use) and keep the listed clips:
<raw>/mp4/{left,right}/<split>/<clip>-{eye}_rectified.mp4.

The videos come as tar files (test: test_mp4s.tar, ~10 GB per eye; train: train_mp4s.tar.part_a..e, ~100 GB per eye),
downloaded to the Hugging Face cache (resumable) and read as one stream.

    python scripts/prepare_stereo4d/download_videos.py --raw $STEREO4D_RAW --split test --clips splits/stereo4d/test_clips.json
"""
import argparse
import io
import os
import tarfile

from common import EYES, load_clips, video_path

REPO = "KevinMathew/stereo4d-{eye}eye-perspective"


class Concat(io.RawIOBase):
    """The tar parts as one sequential stream."""

    def __init__(self, paths):
        self.paths, self.f = list(paths), None

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self.f is None:
                if not self.paths:
                    return 0
                self.f = open(self.paths.pop(0), "rb")
            n = self.f.readinto(b)
            if n:
                return n
            self.f.close()
            self.f = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--clips", required=True, help="clip list json")
    ap.add_argument("--eyes", default="left,right")
    ap.add_argument("--tar_dir", help="already downloaded tars instead of the Hub: a folder pattern with {eye}, e.g. "
                    "/data/stereo4d-{eye}eye-perspective (holding <split>_mp4s.tar or its .part_* files)")
    ap.add_argument("--delete_tar", action="store_true", help="remove the downloaded tar parts after extraction")
    args = ap.parse_args()
    want, _ = load_clips(args.clips)
    for eye in args.eyes.split(","):
        assert eye in EYES, eye
        todo = {c for c in want if not os.path.exists(video_path(args.raw, eye, args.split, c))}
        if not todo:
            print(f"{eye}/{args.split}: all {len(want)} clips present")
            continue
        if args.tar_dir:
            folder = args.tar_dir.format(eye=eye)
            paths = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(f"{args.split}_mp4s.tar"))
        else:
            from huggingface_hub import hf_hub_download, list_repo_files
            repo = REPO.format(eye=eye)
            parts = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.startswith(f"{args.split}_mp4s.tar"))
            paths = [hf_hub_download(repo, f, repo_type="dataset") for f in parts]
        assert paths, f"no {args.split}_mp4s.tar for {eye}"
        os.makedirs(os.path.dirname(video_path(args.raw, eye, args.split, "x")), exist_ok=True)
        n = 0
        with tarfile.open(fileobj=io.BufferedReader(Concat(paths), 1 << 20), mode="r|") as tar:
            for m in tar:
                name = os.path.basename(m.name)
                clip = name[: -len(f"-{eye}_rectified.mp4")] if name.endswith(f"-{eye}_rectified.mp4") else None
                if clip in todo and m.isfile():
                    dst = video_path(args.raw, eye, args.split, clip)
                    with tar.extractfile(m) as src, open(dst + ".tmp", "wb") as out:
                        out.write(src.read())
                    os.replace(dst + ".tmp", dst)
                    todo.discard(clip)
                    n += 1
                    if not todo:
                        break
        print(f"{eye}/{args.split}: extracted {n}, missing {len(todo)}")
        if args.delete_tar and not args.tar_dir:
            for p in paths:
                os.remove(os.path.realpath(p))


if __name__ == "__main__":
    main()
