"""Depth supervision of the per-token depth that RayRoPE predicts in every layer, and ground-truth token
correspondences (the matching loss and the curriculum of GALILEO).

Each attention layer predicts raw_d = (log depth, width) per token; the target is the log of the median valid GT depth
inside the token's patch.
"""
from typing import Dict, List

import torch
from torch import Tensor

from galileo.models.camera import patchify
from galileo.motion import token_logd


def depth_token_targets(depth_gt: Tensor, n_in: int, patch_size: int) -> Tensor:
    """depth_gt [B, V1+V2, H, W, 1] (0 = invalid) -> log depth per token of every sequence [B*V2, (V1+1)*N].

    Sequences follow the model: (b v2) rows, tokens = V1 input views then target j, patches in patchify order.
    NaN where a patch has no valid pixel.
    """
    b, v = depth_gt.shape[:2]
    v2 = v - n_in
    d = patchify(depth_gt.float(), patch_size)  # [B, V, N, P*P]
    d = torch.where(d > 0, d, torch.full_like(d, float("nan")))
    med = torch.nanmedian(d, dim=-1).values  # [B, V, N]
    seq = torch.stack([torch.cat([med[:, :n_in], med[:, n_in + j:n_in + j + 1]], dim=1) for j in range(v2)], dim=1)
    return torch.log(seq).reshape(b * v2, (n_in + 1) * med.shape[-1])


def depth_captures(model) -> List[Tensor]:
    """raw_d [B*V2, T, 2] of every layer that predicts depth, in layer order (empty if none was captured)."""
    out = []
    for layer in model.encoder.layers:
        raw = getattr(layer.self_attn, "captured_depth", None)
        if raw is not None:
            out.append(raw)
    return out


def enable_depth_capture(model, on: bool = True) -> int:
    """Keep every depth-predicting layer's head output of the next forward; returns the number of such layers."""
    n = 0
    for layer in model.encoder.layers:
        attn = layer.self_attn
        if getattr(attn, "predict_d", "none") != "none":
            attn.capture_depth = on
            attn.captured_depth = None
            n += 1
    return n


def depth_supervision_loss(captures: List[Tensor], target_logd: Tensor) -> Dict[str, object]:
    """Mean over layers of the masked L1 between predicted and GT log depth (input and target tokens).
    Returns {"loss": scalar tensor, "per_layer": [float], "n": int}."""
    valid = torch.isfinite(target_logd)
    target = torch.where(valid, target_logd, torch.zeros_like(target_logd))  # no NaN anywhere in the graph
    w = valid.float()
    n = w.sum().clamp_min(1e-6)
    per_layer = []
    for raw in captures:
        err = token_logd(raw.float()) - target
        per_layer.append((err.abs() * w).sum() / n)
    loss = torch.stack(per_layer).mean() if per_layer else target_logd.new_zeros(())
    return {"loss": loss, "per_layer": [float(x.detach()) for x in per_layer], "n": int(valid.sum())}


@torch.no_grad()
def gt_token_points(depth_gt: Tensor, K: Tensor, c2w: Tensor, n_in: int, patch_size: int):
    """World point of every input token's centre ray at its GT patch-median depth: X (B, n_in, P, 3) and that depth
    z (B, n_in, P) (NaN without a valid GT pixel; X then uses depth 1). depth_gt (B, V, H, W, 1) normalized (0
    invalid); K (B, >=n_in, 3, 3) pixels; c2w (B, >=n_in, 4, 4)."""
    W = depth_gt.shape[3]
    d = patchify(depth_gt[:, :n_in].float(), patch_size)
    d = torch.where(d > 0, d, torch.full_like(d, float("nan")))
    z_tok = torch.nanmedian(d, dim=-1).values  # (B, n_in, P)
    P = z_tok.shape[-1]
    gx = W // patch_size
    idx = torch.arange(P, device=depth_gt.device)
    px = torch.stack([(idx % gx).float() * patch_size + patch_size / 2,
                      torch.div(idx, gx, rounding_mode="floor").float() * patch_size + patch_size / 2], -1)
    K, c2w = K[:, :n_in].float(), c2w[:, :n_in].float()
    z0 = torch.nan_to_num(z_tok, nan=1.0)
    xc = torch.stack([(px[:, 0] - K[:, :, None, 0, 2]) / K[:, :, None, 0, 0],
                      (px[:, 1] - K[:, :, None, 1, 2]) / K[:, :, None, 1, 1], torch.ones_like(z0)], -1) * z0[..., None]
    X = torch.einsum("bsij,bspj->bspi", c2w[..., :3, :3], xc) + c2w[:, :, None, :3, 3]
    return X, z_tok


@torch.no_grad()
def depth_correspondences(depth_gt: Tensor, motion: dict, K: Tensor, c2w: Tensor, times: Tensor, n_in: int,
                          patch_size: int, vis_tol: float = 0.05, pi_dyn: float = 0.5) -> Dict[str, Tensor]:
    """GT token correspondences between the input views (GT landing pixels of the match loss and the URoPE curriculum).

    For every ordered pair of input views (s, c), s != c: token p of s (GT patch-median depth on its centre ray),
    moved by its GT displacement to the time of c (0 at the same time and for static tokens), is projected into c.
    It corresponds to the token t of c it lands in when that token's GT depth agrees with the projected depth
    (|log ratio| < vis_tol: visible there). Tokens with partial motion (0 < pi <= pi_dyn) or an invalid
    displacement label are left out.

    depth_gt (B, V, H, W, 1) normalized (0 invalid); motion: disp (B, n_in, V, P, 3) world, pi / valid (B, n_in, P);
    K (B, >=n_in, 3, 3) pixels; c2w (B, >=n_in, 4, 4); times (B, >=n_in).
    Returns tok (B, n_in(s), n_in(c), P) token index in c, ok (same) bool, dyn (B, n_in, P) bool,
    same_time (B, n_in, n_in) bool, uv (B, n_in(s), n_in(c), P, 2) the landing pixel in c.
    """
    B, _, H, W, _ = depth_gt.shape
    gx = W // patch_size
    X, z_tok = gt_token_points(depth_gt, K, c2w, n_in, patch_size)  # (B, s, P, 3), (B, s, P)
    K, c2w = K[:, :n_in].float(), c2w[:, :n_in].float()
    Y = X[:, :, None] + motion["disp"][:, :, :n_in].float()  # (B, s, c, P, 3)
    yc = torch.einsum("bcji,bscpj->bscpi", c2w[..., :3, :3], Y - c2w[:, None, :, None, :3, 3])  # R_c^T (Y - t_c)
    z = yc[..., 2]
    zs = z.clamp_min(1e-6)
    u = K[:, None, :, None, 0, 0] * yc[..., 0] / zs + K[:, None, :, None, 0, 2]
    v = K[:, None, :, None, 1, 1] * yc[..., 1] / zs + K[:, None, :, None, 1, 2]
    inside = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    tok = (v.clamp(0, H - 1) // patch_size).long() * gx + (u.clamp(0, W - 1) // patch_size).long()
    z_dst = torch.gather(z_tok[:, None].expand(-1, n_in, -1, -1), 3, tok)  # GT depth of the landing token
    visible = (torch.log(zs) - torch.log(z_dst)).abs() < vis_tol  # False for NaN
    pi, valid = motion["pi"].float(), motion["valid"].bool()
    dyn = valid & (pi > pi_dyn)
    usable = torch.isfinite(z_tok) & ((pi <= 0) | dyn)
    eye = torch.eye(n_in, dtype=torch.bool, device=depth_gt.device)
    ok = inside & visible & usable[:, :, None] & ~eye[None, :, :, None]
    if "vis" in motion:  # per-pair label reliability (ARROW visibility)
        ok = ok & (motion["vis"][:, :, :n_in] > 0.5)
    same_time = (times[:, :n_in, None] - times[:, None, :n_in]).abs() < 1e-6
    return {"tok": tok, "ok": ok, "dyn": dyn, "same_time": same_time, "uv": torch.stack([u, v], -1)}
