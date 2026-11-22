"""GALILEO: velocity estimation and Galilean key transport (Sec. 3 of the paper).

Every input token gets a constant world velocity v from its matches in the other input times of its own camera:
token features of the matching layer are correlated in a local window (coarse stage, token resolution), refined by a
small CNN at pixel resolution (fine stage), lifted to 3D with the predicted depth at both ends, and fit by least
squares. Each key is then moved to the query's time, u = v (t_q - t_k), before the 3D positional encoding is applied.
"""
import math
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor


def hp(x: Tensor) -> Tensor:
    """At least fp32: upcast half / bfloat16, keep fp32 and fp64 as they are."""
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


class MotionInputs(NamedTuple):
    disp: Tensor  # (B, Cq, Cs, P, 3) world displacement of key token (s, p) to the time of query camera q
    pi: Tensor  # (B, Cs, P) moving weight in [0, 1]


def time_deltas(times: Tensor) -> Tensor:
    """times (B, C) -> dt[b, q, s] = t_q - t_s."""
    return times[:, :, None] - times[:, None, :]


def corr_partners(times: Tensor, cam_ids: Tensor):
    """The other input times of each input view's camera, nearest in time first.

    times (B, V1), cam_ids (B, V1) -> partners (B, V1, Q) long (the view itself where missing), valid (B, V1, Q);
    Q is the largest number of other times of one camera in the batch.
    """
    B, V1 = times.shape
    same_cam = cam_ids[:, :, None] == cam_ids[:, None, :]
    other_t = (times[:, :, None] - times[:, None, :]).abs() > 1e-6
    ok = same_cam & other_t  # (B, s, q)
    Q = max(int(ok.sum(-1).max()), 1)
    key = torch.where(ok, (times[:, :, None] - times[:, None, :]).abs(), torch.full_like(times[:, :, None].expand(
        -1, -1, V1), float("inf")))
    order = key.argsort(dim=-1)[..., :Q]
    valid = torch.gather(ok, 2, order)
    self_idx = torch.arange(V1, device=times.device)[None, :, None].expand(B, -1, Q)
    return torch.where(valid, order, self_idx), valid


def corr_matches(feat: Tensor, partners: Tensor, patches_x: int, patches_y: int, window: int, temp: float):
    """Coarse matching around each token's own grid position (used with URoPE, whose keys carry no depth).

    feat (B, V1, P, D) token features (L2-normalized here); partners (B, V1, Q) index of the q-th partner view of
    view s. For token p of view s: cosine similarity with the window x window partner tokens around p's grid
    position, softmax(sim / temp), expected token offset (soft-argmax: sub-token). Window cells outside the image
    get -inf. Returns match (B, V1, Q, P, 2) normalized image coords (x / W - 0.5) and conf (B, V1, Q, P), the
    softmax peak.
    """
    B, V1, P, D = feat.shape
    Q = partners.shape[-1]
    r = window // 2
    gy, gx = patches_y, patches_x
    f = F.normalize(feat.float(), dim=-1).reshape(B, V1, gy, gx, D).permute(0, 1, 4, 2, 3)  # (B, V1, D, gy, gx)
    pad = F.pad(f, (r, r, r, r))  # zeros outside the image
    inside = F.pad(torch.ones(gy, gx, device=feat.device, dtype=torch.bool), (r, r, r, r))
    idx = torch.arange(P, device=feat.device)
    centre = torch.stack([(idx % gx).float() + 0.5, torch.div(idx, gx, rounding_mode="floor").float() + 0.5], -1)
    size = torch.tensor([gx, gy], dtype=torch.float32, device=feat.device)
    offs = [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    off = torch.tensor(offs, dtype=torch.float32, device=feat.device)  # (w^2, 2) token units
    matches, confs = [], []
    for q in range(Q):
        sel = partners[:, :, q]
        pq = torch.gather(pad, 1, sel[:, :, None, None, None].expand(-1, -1, D, gy + 2 * r, gx + 2 * r))
        sims = []
        for dx, dy in offs:
            sl = pq[..., r + dy:r + dy + gy, r + dx:r + dx + gx]
            s_k = (f * sl).sum(2) / temp  # (B, V1, gy, gx)
            ok = inside[r + dy:r + dy + gy, r + dx:r + dx + gx]
            sims.append(s_k.masked_fill(~ok, float("-inf")))
        sim = torch.stack(sims, 2).reshape(B, V1, len(offs), P)
        w = torch.softmax(sim, dim=2)
        d = torch.einsum("bvkp,kc->bvpc", w, off)  # expected offset (token units)
        matches.append((centre + d) / size - 0.5)
        confs.append(w.max(dim=2).values)
    return torch.stack(matches, 2), torch.stack(confs, 2)


@torch.no_grad()
def corr_static_centres(log_d: Tensor, partners: Tensor, P: Tensor, P_inv: Tensor, patches_x: int, patches_y: int,
                        d_min: float = 1e-2, max_log_d: float = 3.0):
    """Where every input token would land in each partner view if it were static: its centre ray at its own predicted
    log depth, projected into the partner (a moving camera's ego-motion; the token's own position for a fixed one).
    The coarse matching window is centred there (RayRoPE).

    log_d (B, V1, P); partners (B, V1, Q); P, P_inv (B, C, 4, 4) on RayRoPE's normalized ray grid.
    Returns centres (B, V1, Q, P, 2) normalized coords (x / W - 0.5) and front (B, V1, Q, P) bool (in front of the
    partner camera). Centres are clamped to [-1, 1], and points behind the partner go to (-1, -1): beyond the reach
    of any search window (no match).
    """
    B, V1, Np = log_d.shape
    Q = partners.shape[-1]
    with torch.autocast(device_type=log_d.device.type, enabled=False):  # fp16 einsum under AMP: centres off by 1e3
        d = torch.exp(torch.clamp(hp(log_d).float(), max=max_log_d)).clamp_min(d_min)
        idx = torch.arange(Np, device=d.device)
        u = ((idx % patches_x).float() + 0.5) / patches_x - 0.5
        v = (torch.div(idx, patches_x, rounding_mode="floor").float() + 0.5) / patches_y - 0.5
        src = torch.stack([u.expand_as(d), v.expand_as(d), torch.ones_like(d), 1.0 / d], dim=-1)
        Xh = torch.einsum("bcij,bcpj->bcpi", hp(P_inv[:, :V1]).float(), src)
        Xh = Xh / Xh[..., 3:]  # (B, V1, Np, 4) world, w = 1
        P_q = torch.gather(hp(P).float(), 1, partners.reshape(B, V1 * Q)[..., None, None].expand(-1, -1, 4, 4))
        ph = torch.einsum("bsqij,bspj->bsqpi", P_q.reshape(B, V1, Q, 4, 4), Xh)  # (B, V1, Q, Np, 4)
    front = ph[..., 2] > 1e-6
    z = ph[..., 2].clamp_min(1e-6)
    centres = torch.stack([ph[..., 0] / z, ph[..., 1] / z], -1)
    centres = torch.where(front[..., None], centres, torch.full_like(centres, -1.0)).clamp(-1.0, 1.0)
    return centres, front


def corr_matches_at(feat: Tensor, partners: Tensor, centres: Tensor, patches_x: int, patches_y: int, window: int,
                    temp: float):
    """Coarse matching in the window x window cells around a given point of the partner view (centres (B, V1, Q, P, 2)
    normalized coords, from corr_static_centres). Partner features are bilinear on its token grid; cells outside the
    image are dropped, and a token with none inside keeps its centre (conf 0).
    Returns match (B, V1, Q, P, 2) normalized coords and conf (B, V1, Q, P)."""
    B, V1, Np, D = feat.shape
    Q = partners.shape[-1]
    r = window // 2
    gy, gx = patches_y, patches_x
    f = F.normalize(feat.float(), dim=-1)
    fg = f.reshape(B, V1, gy, gx, D).permute(0, 1, 4, 2, 3)  # (B, V1, D, gy, gx)
    fs = f.reshape(B * V1, Np, D)
    size = torch.tensor([gx, gy], dtype=torch.float32, device=feat.device)
    off = torch.tensor([(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)], dtype=torch.float32,
                       device=feat.device) / size  # (w^2, 2) normalized
    matches, confs = [], []
    for q in range(Q):
        pq = torch.gather(fg, 1, partners[:, :, q][:, :, None, None, None].expand(-1, -1, D, gy, gx)).flatten(0, 1)
        c = centres[:, :, q].reshape(B * V1, Np, 2).float()
        dq = torch.ones(B * V1, device=feat.device)
        sims = []
        for k in range(off.shape[0]):
            pos = c + off[k] * dq[:, None, None]
            ft = F.grid_sample(pq, 2 * pos[:, :, None], mode="bilinear", padding_mode="zeros", align_corners=False)
            s_k = (ft[..., 0].transpose(1, 2) * fs).sum(-1) / temp  # (B V1, Np)
            sims.append(s_k.masked_fill(~(pos.abs() < 0.5).all(-1), float("-inf")))
        sim = torch.stack(sims, 1)  # (B V1, w^2, Np)
        any_in = torch.isfinite(sim).any(1)  # (B V1, Np)
        w = torch.softmax(torch.where(any_in[:, None], sim, torch.zeros_like(sim)), dim=1)
        m = c + torch.einsum("nkp,kc->npc", w, off) * dq[:, None, None]
        matches.append(torch.where(any_in[..., None], m, c).reshape(B, V1, Np, 2))
        confs.append(torch.where(any_in, w.max(1).values, torch.zeros_like(w[:, 0])).reshape(B, V1, Np))
    return torch.stack(matches, 2), torch.stack(confs, 2)


class CorrFineNet(nn.Module):
    """Pixel-resolution features for the fine matching stage: (N, H, W, 3) images in [0, 1] -> L2-normalized
    (N, dim, H / 2, W / 2)."""

    def __init__(self, dim: int = 32):
        super().__init__()

        def blk(i, o, s=1, d=1):
            return [nn.Conv2d(i, o, 3, s, padding=d, dilation=d), nn.GroupNorm(8, o), nn.GELU()]

        self.net = nn.Sequential(*blk(3, 32), *blk(32, 64, 2), *blk(64, 64), *blk(64, 64, d=2), *blk(64, 64, d=4),
                                 *blk(64, 64), nn.Conv2d(64, dim, 1))

    def forward(self, img: Tensor) -> Tensor:
        return F.normalize(self.net(img.permute(0, 3, 1, 2) * 2 - 1), dim=1)


def _refine_chunk(fq: Tensor, fs: Tensor, init: Tensor, off: Tensor, temp: float):
    """fq (N, C, h, w) partner features, fs (N, C, p) source descriptors, init (N, p, 2) normalized coarse matches,
    off (K, 2) normalized window offsets -> refined (N, p, 2), softmax peak (N, p)"""
    pos = init[:, :, None] + off[None, None]  # (N, p, K, 2)
    ft = F.grid_sample(fq, 2 * pos.to(fq.dtype), mode="bilinear", padding_mode="border", align_corners=False)
    sim = (ft * fs[..., None]).sum(1).float() / temp  # (N, p, K)
    inside = (pos.abs() < 0.5).all(-1)
    sim = sim.masked_fill(~inside, float("-inf"))
    any_in = inside.any(-1, keepdim=True)
    w = torch.softmax(torch.where(any_in, sim, torch.zeros_like(sim)), dim=-1)
    out = init + torch.einsum("npk,kc->npc", w, off.to(w.dtype))
    return torch.where(any_in, out, init), torch.where(any_in[..., 0], w.max(-1).values, torch.zeros_like(w[..., 0]))


def corr_refine(feat: Tensor, match: Tensor, partners: Tensor, patches_x: int, patches_y: int, img_w: int,
                img_h: int, radius: int, spacing: int, temp: float, chunk: int = 64):
    """Fine matching stage.

    feat (B, V1, C, h, w) L2-normalized pixel features of the input views (spanning the image); match (B, V1, Q, P, 2)
    coarse matches (normalized x / W - 0.5); partners (B, V1, Q). Each token centre is re-matched in its partner view
    by a soft-argmax of cosine similarity / temp over a +-radius px window (spacing px steps, bilinear) around the
    coarse match; window cells outside the image are dropped. Token chunks are checkpointed when grad is on, so the
    window samples are not kept for backward. Returns refined matches (same layout) and the softmax peak (B, V1, Q, P).
    """
    B, V1, C, h, w = feat.shape
    Q, P = match.shape[2], match.shape[3]
    dev = feat.device
    idx = torch.arange(P, device=dev)
    centre = torch.stack([((idx % patches_x).float() + 0.5) / patches_x - 0.5,
                          (torch.div(idx, patches_x, rounding_mode="floor").float() + 0.5) / patches_y - 0.5], -1)
    flat = feat.flatten(0, 1)
    fs = F.grid_sample(flat, (2 * centre).to(flat.dtype)[None, :, None].expand(B * V1, -1, -1, -1),
                       mode="bilinear", padding_mode="border", align_corners=False)[..., 0]  # (B V1, C, P)
    r = torch.arange(-radius, radius + 1, spacing, device=dev, dtype=torch.float32)
    off = torch.stack(torch.meshgrid(r, r, indexing="xy"), -1).reshape(-1, 2)
    off = off / torch.tensor([img_w, img_h], device=dev, dtype=torch.float32)
    outs, confs = [], []
    for q in range(Q):
        fq = torch.gather(feat, 1, partners[:, :, q][:, :, None, None, None].expand(-1, -1, C, h, w)).flatten(0, 1)
        init = match[:, :, q].reshape(B * V1, P, 2).float()
        res, cf = [], []
        for p0 in range(0, P, chunk):
            a = (fq, fs[:, :, p0:p0 + chunk], init[:, p0:p0 + chunk], off, temp)
            m, c = (torch.utils.checkpoint.checkpoint(_refine_chunk, *a, use_reentrant=False)
                    if torch.is_grad_enabled() else _refine_chunk(*a))
            res.append(m)
            cf.append(c)
        outs.append(torch.cat(res, 1).reshape(B, V1, P, 2))
        confs.append(torch.cat(cf, 1).reshape(B, V1, P))
    return torch.stack(outs, 2), torch.stack(confs, 2)


def corr_velocity(log_d: Tensor, match: Tensor, partners: Tensor, pvalid: Tensor, P_inv: Tensor, times: Tensor,
                  patches_x: int, patches_y: int, d_min: float = 1e-2, max_log_d: float = 3.0,
                  depth_sample: str = "nearest", match_w: Optional[Tensor] = None):
    """Constant world velocity of every input token from its matches in the other times of its camera (Eq. 4).

    The token's point X at its own time (centre ray, own log depth) and each match lifted with the partner view's
    log depth at the match (both ends predicted; depth_sample "nearest" partner token or "bilinear" over its token
    grid); v = least-squares slope of position over time through X and the valid matches (pvalid (B, V1, Q);
    matches outside the image are dropped). 0 without a partner at another time.
    log_d (B, V1, P); match (B, V1, Q, P, 2) normalized coords; partners (B, V1, Q); P_inv (B, C, 4, 4); times (B, C);
    match_w (B, V1, Q, P) optional per-match weights (external matches: 0 where the tracker failed).
    Returns v_world (B, V1, P, 3) and the fit weights (B, V1, Q, P).
    """
    B, V1, P = log_d.shape
    Q = partners.shape[-1]
    d = torch.exp(torch.clamp(hp(log_d), max=max_log_d)).clamp_min(d_min)
    dt_ = d.dtype
    idx = torch.arange(P, device=d.device)
    u = ((idx % patches_x).to(dt_) + 0.5) / patches_x - 0.5
    v = (torch.div(idx, patches_x, rounding_mode="floor").to(dt_) + 0.5) / patches_y - 0.5
    ones = torch.ones_like(d)
    src = torch.stack([u.expand_as(d), v.expand_as(d), ones, 1.0 / d], dim=-1)
    Xh = torch.einsum("bcij,bcpj->bcpi", hp(P_inv[:, :V1]).to(dt_), src)
    X = Xh[..., :3] / Xh[..., 3:]  # (B, V1, P, 3)
    m = hp(match).to(dt_)
    col = ((m[..., 0] + 0.5) * patches_x).floor().long().clamp(0, patches_x - 1)
    row = ((m[..., 1] + 0.5) * patches_y).floor().long().clamp(0, patches_y - 1)
    inside = (m[..., 0] >= -0.5) & (m[..., 0] < 0.5) & (m[..., 1] >= -0.5) & (m[..., 1] < 0.5)
    d_q = torch.gather(d[:, None].expand(-1, V1, -1, -1), 2, partners[..., None].expand(-1, -1, -1, P))  # (B,V1,Q,P)
    if depth_sample == "bilinear":  # log depth of the partner view, bilinear over its token grid
        ld = torch.log(d_q).reshape(B * V1 * Q, 1, patches_y, patches_x)
        g = (2 * m).reshape(B * V1 * Q, P, 1, 2).to(ld.dtype)
        z_q = torch.exp(F.grid_sample(ld, g, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0])
        z_q = z_q.reshape(B, V1, Q, P)
    else:
        z_q = torch.gather(d_q, 3, row * patches_x + col)
    dst = torch.stack([m[..., 0], m[..., 1], torch.ones_like(z_q), 1.0 / z_q], dim=-1)  # (B, V1, Q, P, 4)
    P_q = torch.gather(hp(P_inv).to(dt_), 1, partners.reshape(B, V1 * Q)[..., None, None].expand(-1, -1, 4, 4))
    Yh = torch.einsum("bqij,bqpj->bqpi", P_q, dst.reshape(B, V1 * Q, P, 4)).reshape(B, V1, Q, P, 4)
    Y = Yh[..., :3] / Yh[..., 3:]
    t = hp(times).to(dt_)
    t_s = t[:, :V1]
    t_q = torch.gather(t, 1, partners.reshape(B, V1 * Q)).reshape(B, V1, Q)
    w = (pvalid[..., None] & inside & ((t_q - t_s[..., None]).abs() > 1e-6)[..., None]).to(dt_)  # (B, V1, Q, P)
    if match_w is not None:  # the token's own point keeps weight 1
        w = w * match_w.to(dt_)
    return line_fit(X, Y, w, t_s, t_q), w


def line_fit(X: Tensor, Y: Tensor, w: Tensor, t_s: Tensor, t_q: Tensor) -> Tensor:
    """Least-squares line through every token's points {(t_s, X)} U {(t_q, Y_q): w}: v = sum (t - tbar)(x - xbar) /
    sum (t - tbar)^2. X (B, V1, P, 3) at t_s (B, V1); Y (B, V1, Q, P, 3) at t_q (B, V1, Q); w (B, V1, Q, P).
    Returns v (B, V1, P, 3)."""
    n = 1.0 + w.sum(2)  # (B, V1, P)
    tbar = (t_s[..., None] + (w * t_q[..., None]).sum(2)) / n
    xbar = (X + (w[..., None] * Y).sum(2)) / n[..., None]
    ds = t_s[..., None] - tbar  # (B, V1, P)
    dq = t_q[..., None] - tbar[:, :, None]  # (B, V1, Q, P)
    num = ds[..., None] * (X - xbar) + (w[..., None] * dq[..., None] * (Y - xbar[:, :, None])).sum(2)
    den = ds ** 2 + (w * dq ** 2).sum(2)
    return torch.where(den[..., None] > 1e-12, num / den.clamp_min(1e-12)[..., None], torch.zeros_like(num))


def token_logd(raw: Tensor) -> Tensor:
    """Token log depth of a depth-head output raw (..., 2): channel 0 (channel 1 is the log interval width)."""
    return raw[..., 0]


def displace_homogeneous(points_h: Tensor, u: Tensor) -> Tensor:
    """Galilean key transport of RayRoPE's depth points: points_h (..., 4) homogeneous with w = disparity; u
    broadcastable to (..., 3) world displacement. xyz += u * w."""
    w = points_h[..., 3:4]
    xyz = points_h[..., :3] + u * w
    return torch.cat([xyz, w.to(xyz.dtype)], dim=-1)


def phasor_mix(cos_s: Tensor, sin_s: Tensor, cos_d: Tensor, sin_d: Tensor, pi_tok: Tensor, slot_mask: Tensor):
    """Static / transported RoPE coefficients mixed on the depth-point slots: (1 - pi) static + pi transported.

    cos_*, sin_*: (B, T, D). pi_tok: (B, T) in [0, 1]. slot_mask: (D,) bool, True for the depth-point slots.
    Tokens with pi == 0 and the camera-centre slots keep the static coefficients bit-exactly."""
    pi = pi_tok[..., None].to(cos_s.dtype)
    slot = slot_mask.to(cos_s.device)
    sel = (pi_tok[..., None] > 0) & slot
    cos_m = (1 - pi) * cos_s + pi * cos_d
    sin_m = (1 - pi) * sin_s + pi * sin_d
    return torch.where(sel, cos_m, cos_s), torch.where(sel, sin_m, sin_s)


def mask_key_cameras(motion: MotionInputs, key_has_motion: Tensor) -> MotionInputs:
    """u = 0, pi = 0 for key cameras that carry no motion (the target views). key_has_motion: (B, Cs) bool."""
    m = key_has_motion[:, None, :, None, None]
    disp = torch.where(m, motion.disp, torch.zeros_like(motion.disp))
    pi = torch.where(key_has_motion[:, :, None], motion.pi, torch.zeros_like(motion.pi))
    return motion._replace(disp=disp, pi=pi)


def parse_time_planes(spec: str):
    """Which RoPE planes carry the temporal rotation: 'p0:5' -> [('p0', [5], None)], 'x:15,16,17' -> [('x', [15, 16,
    17], None)], 'pd_dir:all:even' -> [('pd_dir', None, 'even')], 'all' -> None (all planes).

    A position name matches a prefix if it starts with it. Frequencies are 'all' or comma separated ints. The optional
    third field picks the position's coordinates: 'all' (default, None), 'even', 'odd' or comma separated ints.
    """
    if spec == "all":
        return None
    out = []
    for part in spec.split("+"):
        fields = part.split(":")
        if len(fields) not in (2, 3):
            raise ValueError(f"time_planes part {part!r}: <pos>:<freqs>[:<coords>]")
        name, freqs = fields[:2]
        freqs = None if freqs == "all" else [int(f) for f in freqs.split(",")]
        coords = None
        if len(fields) == 3 and fields[2] != "all":
            coords = fields[2] if fields[2] in ("even", "odd") else [int(c) for c in fields[2].split(",")]
        out.append((name, freqs, coords))
    return out


def fourier_time_features(dt: Tensor, num_freqs: int, max_period: float = 16.0) -> Tensor:
    """dt (...,) seconds -> (..., 2 * num_freqs) sin/cos features (the Plücker baseline's time embedding)."""
    freqs = torch.exp(torch.linspace(
        math.log(2 * math.pi / max_period), math.log(2 * math.pi / (max_period / 2 ** (num_freqs - 1))),
        num_freqs, device=dt.device))
    ang = hp(dt)[..., None] * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
