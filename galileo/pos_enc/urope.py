# Copyright (c) 2026 Applied Intuition, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from urope/urope.py of https://github.com/Applied-Intuition-Open-Source/URoPE (commit 5d71a04), the
# reference implementation of "URoPE: Universal Relative Position Embedding across Geometric Spaces" (arXiv
# 2604.18747). Changes: OpenCV cameras (x right, y down, z forward) instead of OpenGL; a temporal RoPE (time_enc
# "chan": the time_freqs x-RoPE planes carry omega * t of the token's camera instead of the image x); RoPE
# coefficients in the dtype of q and the geometry in fp32; GALILEO key transport: every key's depth-anchor points move
# by v_d (t_q - t_k) before the projection into the query camera, v_d the key's velocity at anchor depth d from its
# matches (both ends lifted at d).
"""URoPE: each key / value token is lifted to fixed depth anchors along its camera ray (one anchor per group of
heads), projected into the query camera and encoded with 2D RoPE at that image position; queries use their own
image position. Parameter-free and invariant to the global frame."""
import math
import os
from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from galileo.motion import corr_matches, corr_refine, corr_velocity, time_deltas


class URoPEDotProductAttention(torch.nn.Module):
    """URoPE attention (self-attention over cameras x patches tokens, camera-major). gcfg: GalileoConfig of URoPE +
    GALILEO (None: no transport)."""

    def __init__(self, head_num: int, head_dim: int, patches_x: int, patches_y: int, image_width: int,
                 image_height: int, min_depth: float = 2.0, max_depth: float = 20.0, freq_base: float = 100.0,
                 freq_scale: float = 1.0, group_size: int = 4, time_enc: str = "none", time_omega: float = 1.0,
                 time_freqs: Optional[List[int]] = None, gcfg=None):
        super().__init__()
        assert head_num % group_size == 0, "every group of heads gets one depth anchor"
        assert head_dim % 4 == 0, "head_dim splits into an x and a y RoPE block of paired channels"
        assert time_enc in ("none", "chan")
        assert time_enc != "chan" or time_freqs, "time_enc chan gives a few x planes to time: time_freqs"
        self.chan_freqs = list(time_freqs) if time_enc == "chan" else None
        self.head_num, self.head_dim = head_num, head_dim
        self.patches_x, self.patches_y = patches_x, patches_y
        self.image_width, self.image_height = image_width, image_height
        self.group_size = group_size
        self.depth_num = head_num // group_size
        self.min_depth, self.max_depth = min_depth, max_depth
        self.freq_base, self.freq_scale = freq_base, freq_scale
        self.time_enc, self.time_omega = time_enc, time_omega
        self.times = None  # (batch, cameras) seconds, set by the model before each forward
        self.gcfg = gcfg
        self.motion = gcfg is not None and gcfg.enabled
        self.transport = True  # False: anchors stay at their capture time (evaluation "motion off")
        self.set_motion_inputs(None, None, None)

    def set_motion_inputs(self, times, key_has_motion, motion_corr):
        """GALILEO inputs of one forward (LVSM.forward)."""
        self.motion_times = times
        self.key_has_motion = key_has_motion
        self.corr_inputs = motion_corr if self.motion else None
        self._corr_state = None  # matches of the matching layer, kept for the later layers (and recomputation)
        self.corr_captures = {}

    def _corr_from_features(self, feats, layer_idx):
        """Matching at the matching layer: coarse window around the token's own grid position, then the fine stage."""
        ci, cfg = self.corr_inputs, self.gcfg
        B, P = feats.shape[0], self.patches_x * self.patches_y
        v1 = ci["partners"].shape[1]
        f = feats[:, : v1 * P].reshape(B, v1, P, -1)
        match, conf = corr_matches(f, ci["partners"], self.patches_x, self.patches_y, cfg.corr_window, cfg.corr_temp)
        coarse = None
        if cfg.corr_fine_radius > 0 and ci.get("fine_feat") is not None:  # pixel-resolution fine stage
            coarse = match
            match, conf = corr_refine(ci["fine_feat"], coarse.detach(), ci["partners"], self.patches_x,
                                      self.patches_y, self.image_width, self.image_height, cfg.corr_fine_radius,
                                      cfg.corr_fine_spacing, cfg.corr_fine_temp)
        self._corr_state = match
        self.corr_captures = {"match": match, "match_coarse": coarse}

    def _anchor_motion(self, viewmats, Ks):
        """Displacement (batch, q_cams, depth_num, seqlen, 3) of every key's anchors to each query time and the
        moving weight pi (batch, seqlen): the velocity at each anchor depth from the matches lifted at that depth."""
        ci, cfg = self.corr_inputs, self.gcfg
        match = self._corr_state
        B, C = viewmats.shape[:2]
        v1, P = match.shape[1], match.shape[3]
        if ci.get("gt_alpha") is not None and ci.get("gt_match") is not None:  # training: GT matches blended in
            a = ci["gt_alpha"].to(match.dtype)[:, None, None, None, None]
            gt = torch.where(ci["gt_match_ok"][..., None], ci["gt_match"].to(match.dtype), match)
            match = a * gt + (1 - a) * match
        with torch.autocast(device_type=viewmats.device.type, enabled=False):
            w2c = viewmats.float()
            Kn = _normalize_K(Ks.float(), self.image_width, self.image_height)
            P_inv = _invert_SE3(w2c) @ _lift_K(_invert_K(Kn))  # (u, v, 1, 1 / d) on the normalized ray grid -> world
            times = self.motion_times.float()
            dt = time_deltas(times)[:, :, :v1]  # (B, Cq, V1)
            depths = self._anchor_depths(viewmats.device)
            disp = []
            for d in depths.tolist():
                log_d = torch.full((B, v1, P), math.log(d), device=viewmats.device)
                vel, _ = corr_velocity(log_d, match.float(), ci["partners"], ci["pvalid"], P_inv, times,
                                       self.patches_x, self.patches_y, max_log_d=math.log(self.max_depth) + 1.0,
                                       depth_sample=cfg.corr_depth_sample)
                u = vel[:, None] * dt[..., None, None]  # (B, Cq, V1, P, 3)
                u = torch.cat([u, u.new_zeros(B, C, C - v1, P, 3)], dim=2)  # target keys stay
                disp.append(u.reshape(B, C, C * P, 3))
            disp = torch.stack(disp, 2)
            pi = torch.ones(B, v1, P, device=viewmats.device)
            pi = torch.cat([pi, pi.new_zeros(B, C - v1, P)], 1)
            if self.key_has_motion is not None:
                keep = self.key_has_motion.to(pi.device)
                pi = torch.where(keep[:, :, None], pi, torch.zeros_like(pi))
                disp = torch.where(keep.repeat_interleave(P, 1)[:, None, None, :, None], disp, torch.zeros_like(disp))
        return disp, pi.reshape(B, C * P)

    def _anchor_depths(self, device):
        return self.min_depth + (self.max_depth - self.min_depth) / self.depth_num * torch.arange(
            self.depth_num, device=device).float()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, viewmats: torch.Tensor, Ks: torch.Tensor,
                layer_idx: Optional[int] = None, corr_feats: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """q, k, v (batch, num_heads, seqlen, head_dim); viewmats (batch, cameras, 4, 4) world -> camera; Ks (batch,
        cameras, 3, 3) pixels; corr_feats at the matching layer; kwargs go to scaled_dot_product_attention."""
        times = self.times if self.time_enc == "chan" else None
        disp = pi = None
        if self.motion and self.corr_inputs is not None and self.motion_times is not None:
            if corr_feats is not None:
                self._corr_from_features(corr_feats, layer_idx)
            if (self.transport and layer_idx is not None and layer_idx >= self.gcfg.corr_layer
                    and self._corr_state is not None):
                disp, pi = self._anchor_motion(viewmats, Ks)
        return urope_dot_product_attention(
            q, k, v, viewmats=viewmats, Ks=Ks, patches_x=self.patches_x, patches_y=self.patches_y,
            image_width=self.image_width, image_height=self.image_height, group_size=self.group_size,
            depth_num=self.depth_num, min_depth=self.min_depth, max_depth=self.max_depth, freq_base=self.freq_base,
            freq_scale=self.freq_scale, times=times, time_omega=self.time_omega, disp=disp, pi=pi,
            chan_freqs=self.chan_freqs, **kwargs)


def urope_dot_product_attention(q, k, v, *, viewmats, Ks, patches_x: int, patches_y: int, image_width: int,
                                image_height: int, group_size: int, depth_num: int, min_depth: float,
                                max_depth: float, freq_base: float = 100.0, freq_scale: float = 1.0,
                                times: Optional[torch.Tensor] = None, time_omega: float = 1.0,
                                disp: Optional[torch.Tensor] = None, pi: Optional[torch.Tensor] = None,
                                chan_freqs: Optional[List[int]] = None, **kwargs) -> torch.Tensor:
    """disp (batch, q_cams, depth_num, seqlen, 3) world displacement of the anchors and pi (batch, seqlen) its phasor
    weight (GALILEO); chan_freqs: the x frequencies that carry omega * t instead of the image x."""
    batch, num_heads, seqlen, head_dim = q.shape
    cameras = viewmats.shape[1]
    patch_num = patches_x * patches_y
    assert q.shape == k.shape == v.shape and seqlen == cameras * patch_num
    assert Ks is not None and Ks.shape == (batch, cameras, 3, 3) and num_heads == depth_num * group_size
    with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):  # camera geometry only
        q_x, q_y, kv_x, kv_y = urope_positions(viewmats, Ks, patches_x, patches_y, image_width, image_height,
                                               group_size, depth_num, min_depth, max_depth)
        q_x, q_y = q_x[:1], q_y[:1]  # the same for every query camera: (1, heads, patch_num)
        kv_x = kv_x.reshape(batch, cameras, num_heads, seqlen)
        kv_y = kv_y.reshape(batch, cameras, num_heads, seqlen)
        q_phase = kv_phase = None
        if times is not None and time_omega != 0:
            t = time_omega * times.float()  # (batch, cameras)
            q_phase = t[:, :, None, None]  # the query camera's time: (batch, q_cams, 1, 1)
            kv_phase = t.repeat_interleave(patch_num, dim=1)[:, None, :]  # each key's camera time: (batch, 1, seqlen)
    kvm_x = kvm_y = pim = None
    if disp is not None:  # moved anchors (grad flows to the matches and pi)
        with torch.autocast(device_type=q.device.type, enabled=False):
            _, _, kvm_x, kvm_y = urope_positions(viewmats, Ks, patches_x, patches_y, image_width, image_height,
                                                 group_size, depth_num, min_depth, max_depth, disp=disp)
        kvm_x = kvm_x.reshape(batch, cameras, num_heads, seqlen)
        kvm_y = kvm_y.reshape(batch, cameras, num_heads, seqlen)
        pim = pi[:, None, :, None].to(q.dtype)  # (batch, 1, seqlen, 1)
    chan_mask = None
    if chan_freqs:
        chan_mask = torch.zeros(head_dim // 4, dtype=torch.bool, device=q.device)
        chan_mask[chan_freqs] = True
    fn = _camera_fn(q)
    outs = []
    for c in range(cameras):
        qc = q[:, :, c * patch_num:(c + 1) * patch_num]
        args = (qc, k, v, q_x, q_y, kv_x[:, c], kv_y[:, c], None if q_phase is None else q_phase[:, c], kv_phase,
                freq_base, freq_scale, kwargs, None if kvm_x is None else kvm_x[:, c],
                None if kvm_y is None else kvm_y[:, c], pim, chan_mask)
        if torch.is_grad_enabled():
            outs.append(torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False))
        else:
            outs.append(fn(*args))
    return torch.cat(outs, dim=2)


_COMPILED = None


def _camera_fn(q: torch.Tensor):
    """_one_query_camera, compiled on CUDA (fuses the rotations: ~2x faster per attention call, same memory);
    UROPE_COMPILE=0 keeps it eager."""
    global _COMPILED
    if not q.is_cuda or os.environ.get("UROPE_COMPILE", "1") == "0":
        return _one_query_camera
    if _COMPILED is None:
        # one graph per (train / eval shape, grad, with / without transport): with transport the default limit of 8
        # is reached at the first evaluation and dynamo falls back to eager (2x slower) for the rest of the run
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        _COMPILED = torch.compile(_one_query_camera, dynamic=False)
    return _COMPILED


def _one_query_camera(qc, k, v, q_x, q_y, kv_x, kv_y, q_phase, kv_phase, freq_base, freq_scale, sdpa_kwargs,
                      kvm_x=None, kvm_y=None, pim=None, chan_mask=None):
    """Queries of one camera (batch, heads, patch_num, head_dim) against all keys / values, encoded for that camera:
    kv_x / kv_y (batch, heads, seqlen) token-unit positions in it; phases broadcastable to the positions or None.
    With transport, kvm_x / kvm_y the moved anchors' positions and pim (batch, 1, seqlen, 1) the weight of their
    encoding: cos / sin = (1 - pi) static + pi moved (phasor mix)."""
    feat = qc.shape[-1] // 2
    cq_x = _rope_coeffs(q_x, freq_base, freq_scale, feat, q_phase, qc.dtype, chan_mask)
    cq_y = _rope_coeffs(q_y, freq_base, freq_scale, feat, None, qc.dtype)
    ck_x = _rope_coeffs(kv_x, freq_base, freq_scale, feat, kv_phase, qc.dtype, chan_mask)
    ck_y = _rope_coeffs(kv_y, freq_base, freq_scale, feat, None, qc.dtype)
    if kvm_x is not None:
        mx = _rope_coeffs(kvm_x, freq_base, freq_scale, feat, kv_phase, qc.dtype, chan_mask)
        my = _rope_coeffs(kvm_y, freq_base, freq_scale, feat, None, qc.dtype)
        ck_x = tuple((1 - pim) * s_ + pim * m_ for s_, m_ in zip(ck_x, mx))
        ck_y = tuple((1 - pim) * s_ + pim * m_ for s_, m_ in zip(ck_y, my))
    rot_q = [(partial(_rope_apply_coeffs, coeffs=cq_x), feat), (partial(_rope_apply_coeffs, coeffs=cq_y), feat)]
    rot_kv = [(partial(_rope_apply_coeffs, coeffs=ck_x), feat), (partial(_rope_apply_coeffs, coeffs=ck_y), feat)]
    rot_o = [(partial(_rope_apply_coeffs, coeffs=cq_x, inverse=True), feat),
             (partial(_rope_apply_coeffs, coeffs=cq_y, inverse=True), feat)]
    out = F.scaled_dot_product_attention(_apply_block_diagonal(qc, rot_q), _apply_block_diagonal(k, rot_kv),
                                         _apply_block_diagonal(v, rot_kv), **sdpa_kwargs)
    return _apply_block_diagonal(out, rot_o)


def urope_positions(viewmats: torch.Tensor, Ks: torch.Tensor, patches_x: int, patches_y: int, image_width: int,
                    image_height: int, group_size: int, depth_num: int, min_depth: float, max_depth: float,
                    disp: Optional[torch.Tensor] = None):
    """Token-unit image positions: queries (batch * cameras, heads, patch_num) at their own patch centres; keys
    (batch * q_cams, heads, cameras * patch_num): head h's depth anchor (h // group_size) on the key's ray, projected
    into the query camera. disp (batch, q_cams, depth_num, cameras * patch_num, 3): world displacement of the anchor
    points to each query camera's time, added before the projection. fp32."""
    device = viewmats.device
    batch, cameras = viewmats.shape[:2]
    patch_num = patches_x * patches_y
    w2c = viewmats.float()
    Kf = Ks.float()
    c2w = _invert_SE3(w2c)
    px_x = image_width / patches_x
    px_y = image_height / patches_y
    gy, gx = torch.meshgrid(torch.arange(patches_y, device=device, dtype=torch.float32),
                            torch.arange(patches_x, device=device, dtype=torch.float32), indexing="ij")
    tok_x = (gx + 0.5).reshape(-1)  # (patch_num,) token-unit image coordinates, row-major
    tok_y = (gy + 0.5).reshape(-1)
    pix = torch.stack([tok_x * px_x, tok_y * px_y, torch.ones_like(tok_x)], -1)  # (patch_num, 3)
    rays = pix[None, None] @ _invert_K(Kf).transpose(-1, -2)  # (batch, cameras, patch_num, 3), z = 1
    # depth anchors (one per group of heads) along every key ray, in world coordinates
    depths = min_depth + (max_depth - min_depth) / depth_num * torch.arange(depth_num, device=device).float()
    pts_cam = rays[:, :, None] * depths[None, None, :, None, None]  # (batch, cams, depth_num, patch_num, 3)
    pts_w = pts_cam @ c2w[:, :, None, :3, :3].transpose(-1, -2) + c2w[:, :, None, None, :3, 3]
    # into every query camera: (batch, q_cams, depth_num, k_cams * patch_num, 3)
    pts_w = pts_w.permute(0, 2, 1, 3, 4).reshape(batch, 1, depth_num, cameras * patch_num, 3)
    if disp is not None:
        pts_w = pts_w + disp.float()
    pts_q = pts_w @ w2c[:, :, None, :3, :3].transpose(-1, -2) + w2c[:, :, None, None, :3, 3]
    img = pts_q @ Kf[:, :, None].transpose(-1, -2)
    z = img[..., 2].abs() + 1e-5  # as upstream: points behind the query camera are mirrored, not dropped
    kv_x = (img[..., 0] / z / px_x).clamp(-64, 96)
    kv_y = (img[..., 1] / z / px_y).clamp(-64, 96)
    kv_x = kv_x.repeat_interleave(group_size, dim=2).flatten(0, 1)  # (batch * q_cams, heads, seqlen)
    kv_y = kv_y.repeat_interleave(group_size, dim=2).flatten(0, 1)
    num_heads = depth_num * group_size
    q_x = tok_x[None, None].expand(batch * cameras, num_heads, -1)
    q_y = tok_y[None, None].expand(batch * cameras, num_heads, -1)
    return q_x, q_y, kv_x, kv_y


def _rope_coeffs(positions: torch.Tensor, freq_base: float, freq_scale: float, feat_dim: int,
                 phase: Optional[torch.Tensor], dtype: torch.dtype,
                 chan_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """positions (batch, heads, seqlen) -> cos, sin (batch, heads, seqlen, feat_dim // 2); phase (broadcastable to
    (batch, heads, seqlen)) is added to every frequency's angle, or, with chan_mask (feat_dim // 2,) bool, replaces
    the angle of those frequencies (0 without a phase) and leaves the others time-free."""
    num_freqs = feat_dim // 2
    freqs = freq_scale * freq_base ** (-torch.arange(num_freqs, device=positions.device).float() / num_freqs)
    angles = positions[..., None] * freqs
    if chan_mask is not None:
        t_angle = phase[..., None] if phase is not None else torch.zeros((), device=angles.device)
        angles = torch.where(chan_mask, t_angle.to(angles.dtype), angles)
    elif phase is not None:
        angles = angles + phase[..., None]
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def _rope_apply_coeffs(feats: torch.Tensor, coeffs: Tuple[torch.Tensor, torch.Tensor],
                       inverse: bool = False) -> torch.Tensor:
    """'split' channel ordering (first half paired with second half), as upstream."""
    cos, sin = coeffs
    x_in, y_in = feats[..., : feats.shape[-1] // 2], feats[..., feats.shape[-1] // 2:]
    if not inverse:
        return torch.cat([cos * x_in + sin * y_in, -sin * x_in + cos * y_in], dim=-1)
    return torch.cat([cos * x_in - sin * y_in, sin * x_in + cos * y_in], dim=-1)


def _apply_block_diagonal(feats: torch.Tensor, func_size_pairs: List[Tuple[Callable, int]]) -> torch.Tensor:
    funcs, sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(sizes)
    return torch.cat([f(x) for f, x in zip(funcs, torch.split(feats, sizes, dim=-1))], dim=-1)


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _normalize_K(Ks: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    Ks_norm = torch.zeros_like(Ks)
    Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
    Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
    Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
    Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
    Ks_norm[..., 2, 2] = 1.0
    return Ks_norm
