"""Video metrics of the paper: tOF, delta_avg and JEDi (supp. "Metrics").

    tOF        mean over frame pairs of the mean L1 endpoint distance between RAFT flows of consecutive rendered frames
               and of consecutive ground-truth frames, over the dynamic pixels of either frame (frame pairs without
               any are left out; a video without any has no value)
    delta_avg  queries: a 4 px grid on frame 0 (valid depth), tracked on the ground-truth video with CoTracker3;
               those that move beyond their static reprojection (frame-0 depth, cameras) by more than 1.5 px at some
               visible frame, at most 1024; each is tracked on the rendered video and compared with its
               ground-truth track at frames 1.. where that is visible: the mean fraction within 1, 2, 4, 8 and 16 px
    JEDi       V-JEPA ViT-H/16 features with JEDi's Something-Something-v2 attentive-probe pooler (1280-d) of each
               video (16 frames, 224 x 224), MMD with JEDi's polynomial kernel (degree 2, gamma 1 / dim) x 100
               between the rendered and the ground-truth videos of the test set

Per video, tOF and delta_avg; the summary averages them over the videos that have a value. Models: torchvision's RAFT
(raft_large, C_T_SKHT_V2), CoTracker3 offline (torch.hub facebookresearch/co-tracker), V-JEPA ViT-H/16 and its SSv2
probe (dl.fbaipublicfiles.com/jepa/vith16/{vith16,ssv2-probe}.pth.tar; model code from the `vjepa` package).
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

THRESH = (1, 2, 4, 8, 16)  # delta_avg thresholds, px
MOVE_PX = 1.5  # moving-point threshold at 256 px
GRID, MAX_QUERIES = 4, 1024
VJEPA_FRAMES, VJEPA_RES = 16, 224
COTRACKER_COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"  # the co-tracker version of the paper


# ---------------------------------------------------------------- models
def load_raft(device):
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    return raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).eval().requires_grad_(False).to(device)


def load_cotracker(repo, device):
    """CoTracker3 offline from a local clone of facebookresearch/co-tracker (or "facebookresearch/co-tracker" from
    GitHub); the checkpoint is downloaded into the torch hub cache."""
    if os.path.isdir(repo):
        m = torch.hub.load(repo, "cotracker3_offline", source="local")
    else:
        m = torch.hub.load(repo, "cotracker3_offline", trust_repo=True)
    return m.eval().to(device)


def load_vjepa(vjepa_dir, device):
    """V-JEPA ViT-H/16 (target encoder) and the SSv2 attentive-probe pooler, as JEDi's default V-JEPA (config
    vith16_ssv2_16x2x3: 16 frames, tubelet 2, 224 px, uniform power, SDPA)."""
    from vjepa.models import vit_huge
    from vjepa.models.attentive_pooler import AttentiveClassifier
    enc = vit_huge(img_size=VJEPA_RES, patch_size=16, num_frames=VJEPA_FRAMES, tubelet_size=2, uniform_power=True,
                   use_sdpa=True, use_SiLU=False, tight_SiLU=False)
    sd = torch.load(os.path.join(vjepa_dir, "vith16.pth.tar"), map_location="cpu", weights_only=False)
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd["target_encoder"].items()}
    enc.load_state_dict(sd)
    clf = AttentiveClassifier(embed_dim=enc.embed_dim, num_heads=enc.num_heads, depth=1, num_classes=174)
    sd = torch.load(os.path.join(vjepa_dir, "ssv2-probe.pth.tar"), map_location="cpu", weights_only=False)
    clf.load_state_dict({k.removeprefix("module."): v for k, v in sd["classifier"].items()})
    return enc.to(device).eval(), clf.pooler.to(device).eval()


class VideoMetrics:
    def __init__(self, device, vjepa_dir, cotracker="facebookresearch/co-tracker"):
        self.device = device
        self.raft = load_raft(device)
        self.cotracker = load_cotracker(cotracker, device)
        self.vjepa, self.pooler = load_vjepa(vjepa_dir, device)

    @torch.no_grad()
    def flow(self, a, b):  # (N, 3, H, W) in [0, 1] -> RAFT flow a -> b (N, 2, H, W), px
        return self.raft(a * 2 - 1, b * 2 - 1)[-1]

    @torch.no_grad()
    def track(self, video, queries):
        """video (T, 3, H, W) in [0, 1], queries (N, 2) xy at frame 0 -> tracks (T, N, 2), visible (T, N)"""
        q = torch.cat([torch.zeros_like(queries[:, :1]), queries], 1)[None].float()
        tr, vis = self.cotracker(video[None] * 255.0, queries=q)
        return tr[0], vis[0].bool()

    @torch.no_grad()
    def features(self, video):  # (T, 3, H, W) in [0, 1] -> (1280,)
        x = video[np.linspace(0, len(video) - 1, VJEPA_FRAMES).round().astype(int)]  # uniform, nearest
        x = F.interpolate(x, size=(VJEPA_RES, VJEPA_RES), mode="bilinear", antialias=True, align_corners=False)
        mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        x = ((x - mean) / std).permute(1, 0, 2, 3)[None]  # (1, 3, 16, 224, 224)
        return self.pooler(self.vjepa(x)).squeeze(1)[0].float().cpu().numpy()

    @torch.no_grad()
    def __call__(self, pred, gt, dyn, depth0, K, c2w):
        """pred, gt (T, H, W, 3) uint8 (pred with the ground-truth frame at input times), dyn (T, H, W) bool moving
        pixels, depth0 (H, W) metric depth of frame 0, K (T, 3, 3), c2w (T, 4, 4) -> per-video metrics and the V-JEPA
        features of both videos."""
        dev = self.device
        to_t = lambda x: torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float() / 255.0  # noqa: E731
        p, g = to_t(pred), to_t(gt)
        dyn = torch.from_numpy(dyn).to(dev)
        T, _, H, W = g.shape
        res = {}

        # tOF
        a, b = slice(0, T - 1), slice(1, T)
        union = dyn[a] | dyn[b]
        e = (self.flow(p[a], p[b]) - self.flow(g[a], g[b])).norm(dim=1)
        per = [e[t][union[t]].mean().item() for t in range(T - 1) if union[t].any()]
        res["tof"] = float(np.mean(per)) if per else None

        # delta_avg
        ys, xs = torch.meshgrid(torch.arange(GRID // 2, H, GRID, device=dev),
                                torch.arange(GRID // 2, W, GRID, device=dev), indexing="ij")
        xy = torch.stack([xs.flatten(), ys.flatten()], 1).float()
        d0 = torch.from_numpy(depth0).to(dev)
        xy = xy[d0[xy[:, 1].long(), xy[:, 0].long()] > 0]
        tr_g, vis_g = self.track(g, xy)
        still = static_reprojection(xy, d0, torch.from_numpy(K).to(dev).double(),
                                    torch.from_numpy(c2w).to(dev).double()).float()
        move = (((tr_g - still).norm(dim=-1) * 256.0 / W) * vis_g).amax(0) > MOVE_PX
        sel = torch.nonzero(move).flatten()
        if len(sel) > MAX_QUERIES:
            sel = sel[torch.linspace(0, len(sel) - 1, MAX_QUERIES, device=dev).round().long()]
        res["n_queries"], res["delta_avg"] = int(len(sel)), None
        if len(sel):
            tr_p, _ = self.track(p, xy[sel])
            m = vis_g[1:, sel]
            err = (tr_p[1:] - tr_g[1:, sel]).norm(dim=-1)[m].cpu().numpy().astype(np.float32)
            if len(err):
                res["delta_avg"] = float(np.mean([(err < t).mean() for t in THRESH]))

        return res, self.features(p), self.features(g)


def static_reprojection(xy, depth0, K, c2w):
    """frame-0 pixels xy (N, 2, pixel index) at their frame-0 depth, projected into every frame: (T, N, 2)"""
    xi, yi = xy[:, 0].round().long(), xy[:, 1].round().long()
    z = depth0[yi, xi].to(K.dtype)
    u, v = xy[:, 0].to(K.dtype) + 0.5, xy[:, 1].to(K.dtype) + 0.5
    Xc = torch.stack([(u - K[0, 0, 2]) * z / K[0, 0, 0], (v - K[0, 1, 2]) * z / K[0, 1, 1], z], -1)
    Xw = Xc @ c2w[0, :3, :3].T + c2w[0, :3, 3]
    w2c = torch.linalg.inv(c2w)
    Pc = torch.einsum("tij,nj->tni", w2c[:, :3, :3], Xw) + w2c[:, None, :3, 3]
    zs = Pc[..., 2].clamp_min(1e-6)
    return torch.stack([K[:, None, 0, 0] * Pc[..., 0] / zs + K[:, None, 0, 2],
                        K[:, None, 1, 1] * Pc[..., 1] / zs + K[:, None, 1, 2]], -1) - 0.5


def jedi(a, b):
    """JEDi (MMD, polynomial kernel of degree 2, gamma 1 / dim, coef0 0; biased, as JEDi) x 100 of feature sets a, b"""
    a, b = a.astype(np.float64), b.astype(np.float64)
    k = lambda x, y: (x @ y.T / x.shape[1]) ** 2  # noqa: E731
    return float((k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean()) * 100)
