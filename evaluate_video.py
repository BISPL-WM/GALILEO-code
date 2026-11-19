"""Video metrics of a checkpoint on the Stereo4D test clips (paper Tab. 1: tOF, delta_avg, JEDi).

    python evaluate_video.py --config configs/stereo4d/rayrope_galileo.yaml \
        --ckpt checkpoints/rayrope_galileo.safetensors --vjepa_dir /path/to/vjepa \
        data_dir=$STEREO4D_DIR output_dir=runs/video/rayrope_galileo

One video per test clip and camera (1,200): the clip's inputs (both cameras at the four input times) and that camera at
every frame from the first input time to two frames past the last (18 frames), rendered in one forward pass. Frames at
input times are the ground-truth frame in the video. Metrics: galileo/video_metrics.py.

Writes <output_dir>/videos/per_video/<video>.json (tOF, delta_avg) and feats/<video>.npz (V-JEPA features of the
rendered and ground-truth videos); done videos are skipped, so a rerun resumes and --shard k/K splits the videos
between processes (e.g. one per GPU). Once every video is done, prints the means and JEDi and writes summary.json.
--save_videos keeps the rendered frames (pred/<video>.npy, uint8). --motion_off as in evaluate.py.
"""
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # the paper's videos were rendered without torch.compile

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from galileo import config_utils  # noqa: E402
from galileo.data.stereo4d import Stereo4DDataset, _npz  # noqa: E402
from galileo.trainer import TrainConfig, Trainer, load_model_weights  # noqa: E402
from galileo.video_metrics import VideoMetrics, jedi  # noqa: E402


def video_entries(ds, clips):
    """[(video id, entry)]: per clip, each camera as the target over the clip's whole window"""
    out = []
    for clip in clips:
        fr = ds.inputs_of(clip)
        frames = list(range(fr[0], fr[-1] + max(1, ds.gap // 2) + 1))
        for eye in (0, 1):
            out.append((f"{clip}_e{eye}", {"split": "test", "clip": clip, "protocol": "interp",
                                           "inputs": [[x, f] for f in fr for x in (0, 1)],
                                           "targets": [[eye, f] for f in frames]}))
    return out


def u8(x):  # [0, 1] -> uint8
    return (x.clamp(0, 1).float().cpu().numpy() * 255).round().astype(np.uint8)


@torch.no_grad()
def render_video(trainer, model, ds, e):
    """rendered and ground-truth frames (T, H, W, 3) uint8 (rendered: the ground-truth frame at input times), moving
    pixels (T, H, W), frame-0 depth and cameras (raw, as in data.npz)"""
    item = ds.load(e)
    p = trainer.preprocess(torch.utils.data.default_collate([item]), input_views=len(e["inputs"]))
    dev = trainer.device.type
    with torch.amp.autocast(dev, enabled=trainer.config.amp, dtype=trainer.amp_dtype):
        out = model(p["ref_imgs"], p["ref_cams"], p["tar_cams"], **trainer.model_kwargs(p))
    pred, gt = u8(torch.sigmoid(out.float())[0]), u8(p["tar_imgs"][0])
    ins = {tuple(v) for v in e["inputs"]}
    is_in = np.array([tuple(v) in ins for v in e["targets"]])
    pred[is_in] = gt[is_in]
    dyn = item["eval_gt"]["dyn_mask"][..., 0].numpy() > 0.5
    z = _npz(os.path.join(ds.root, e["split"], e["clip"], "data.npz"))
    eye, f0 = e["targets"][0]
    K = np.repeat(np.asarray(z["K"], np.float32)[None], len(e["targets"]), 0)
    c2w = np.stack([z["c2w"][x, f] for x, f in e["targets"]]).astype(np.float32)
    return pred, gt, dyn, np.asarray(z["depth"][eye, f0], np.float32), K, c2w


def summarize(out, n_videos):
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/per_video/*.json"))]
    if len(rows) < n_videos:
        print(f"{len(rows)} of {n_videos} videos done; run again (or wait for the other shards) for the summary")
        return None
    mean = lambda k: float(np.mean([r[k] for r in rows if r[k] is not None]))  # noqa: E731
    feats = [np.load(f"{out}/feats/{r['video']}.npz") for r in rows]
    s = {"n_videos": len(rows), "tof": mean("tof"), "delta_avg": mean("delta_avg"),
         "jedi": jedi(np.stack([f["pred"] for f in feats]), np.stack([f["gt"] for f in feats])),
         "n_videos_tof": sum(r["tof"] is not None for r in rows),
         "n_videos_delta_avg": sum(r["delta_avg"] is not None for r in rows)}
    json.dump(s, open(f"{out}/summary.json", "w"), indent=1)
    print(json.dumps(s, indent=1))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vjepa_dir", required=True, help="directory with vith16.pth.tar and ssv2-probe.pth.tar")
    ap.add_argument("--cotracker", default="facebookresearch/co-tracker",
                    help="local clone of facebookresearch/co-tracker, or the GitHub repo")
    ap.add_argument("--clips", default="splits/stereo4d/test_clips.json")
    ap.add_argument("--motion_off", action="store_true")
    ap.add_argument("--save_videos", action="store_true")
    ap.add_argument("--shard", default="0/1", help="k/K: this process takes every K-th video from the k-th")
    ap.add_argument("overrides", nargs="*", help="key.sub=value")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    cfg = config_utils.load(TrainConfig, args.config, args.overrides)
    trainer = Trainer(cfg, device=torch.device("cuda"))
    out = os.path.join(cfg.output_dir, "videos")
    for d in ("per_video", "feats") + (("pred",) if args.save_videos else ()):
        os.makedirs(f"{out}/{d}", exist_ok=True)

    ds = Stereo4DDataset(cfg.data_dir, split="test", n_in_times=cfg.n_in_times, gap=cfg.gap, eval_gt=True,
                         min_partners=cfg.min_partners, ext_raft=cfg.ext_raft)
    videos = video_entries(ds, [c["clip"] for c in json.load(open(args.clips))])
    k, n = (int(x) for x in args.shard.split("/"))
    todo = [(v, e) for v, e in videos[k::n] if not os.path.exists(f"{out}/per_video/{v}.json")]
    if todo:
        model = trainer.build_model()
        load_model_weights(model, args.ckpt)
        model.eval()
        if args.motion_off:
            model.attention.transport = False
        metrics = VideoMetrics(trainer.device, args.vjepa_dir, args.cotracker)
        print(f"{len(todo)} videos to do", flush=True)
        for i, (vid, e) in enumerate(todo):
            pred, gt, dyn, depth0, K, c2w = render_video(trainer, model, ds, e)
            res, f_pred, f_gt = metrics(pred, gt, dyn, depth0, K, c2w)
            if args.save_videos:
                np.save(f"{out}/pred/{vid}.npy", pred)
            np.savez(f"{out}/feats/{vid}.npz", pred=f_pred, gt=f_gt)
            json.dump({"video": vid, **res}, open(f"{out}/per_video/{vid}.json", "w"))
            if i % 50 == 0:
                print(f"{i + 1}/{len(todo)} {vid} {res}", flush=True)
    summarize(out, len(videos))


if __name__ == "__main__":
    main()
