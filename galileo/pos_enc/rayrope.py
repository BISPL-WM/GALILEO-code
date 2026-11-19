"""RayRoPE (projective ray positional encoding for multi-view attention, github.com/wuyu-wy/RayRoPE, MIT) with a
temporal RoPE and GALILEO's Galilean key transport.

Every token is encoded by its camera centre (3D, in the query camera's frame) and by points on its rays at a predicted
depth interval [exp(d - s), exp(d + s)] (projected into the query camera: direction and depth); the RoPE coefficients
of the depth points are averaged over the interval in closed form. Temporal RoPE ("chan"): the time_planes frequencies
of the camera-centre coordinates carry omega (t_k - t_q) instead. GALILEO: before the encoding for query camera q, the
depth points of every moving key token are displaced by its estimated motion u = v (t_q - t_k) (galileo/motion.py).

This module replaces torch.nn.functional.scaled_dot_product_attention: the model calls
_precompute_and_cache_apply_fns once per forward (cameras, times, GALILEO inputs) and every layer calls forward with
its predicted depths (and, at the matching layer, its matching features).
"""
from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from galileo.motion import (MotionInputs, corr_refine, corr_static_centres, corr_matches_at, corr_velocity,
                            displace_homogeneous, hp, mask_key_cameras, parse_time_planes, phasor_mix, time_deltas,
                            token_logd)

MAX_DEPTH = 100.0
MAX_LOG_DEPTH = 3.0
MAX_D_F = 10.0


class RayRoPE_DotProductAttention(torch.nn.Module):
    """RayRoPE self-attention over cameras x patches tokens (camera-major).

    head_dim must be a multiple of 2 * rope_coord_dim, rope_coord_dim = 3 (camera centre) + 3 rays x 3 (depth point
    direction (2) and depth (1)) = 12; head_dim 144 gives 6 frequencies per coordinate. gcfg: GalileoConfig.
    """

    def __init__(self, head_dim: int, patches_x: int, patches_y: int, image_width: int, image_height: int,
                 pos_enc_type: str = "d_pj+0_3d", num_rays_per_patch: int = 3, depth_type: str = "predict_dsig",
                 denc_type: str = "d", freq_base: float = 3.0, gcfg=None):
        super().__init__()
        assert pos_enc_type == "d_pj+0_3d" and depth_type == "predict_dsig" and denc_type == "d"
        assert num_rays_per_patch == 3
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.num_patches = patches_x * patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.num_rays_per_patch = num_rays_per_patch
        self.freq_base = freq_base
        self.rope_coord_dim = 3 + self.num_rays_per_patch * 3
        assert head_dim % (self.rope_coord_dim * 2) == 0, f"head_dim {head_dim}: a multiple of {2 * self.rope_coord_dim}"
        self.num_rope_freqs = head_dim // (2 * self.rope_coord_dim)
        self.offsets = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]  # ray offsets within a patch (patch units)
        self.gcfg = gcfg
        assert gcfg.time_enc == "chan", "RayRoPE carries time on its camera-centre planes (time_enc chan)"
        self.time_planes = parse_time_planes(gcfg.time_planes)
        self.time_chan = (self.time_planes, gcfg.time_omega)
        self.times = None
        self.time_active = False
        self.key_has_motion = None
        self.corr_inputs = None
        self._corr_state = None
        self.corr_captures = {}
        self.transport = True  # False: keys stay at their capture time (evaluation "motion off")

    # ---------------------------------------------------------------- per forward
    def _precompute_and_cache_apply_fns(self, w2cs: torch.Tensor, Ks: torch.Tensor, times: Optional[torch.Tensor] = None,
                                        key_has_motion: Optional[torch.Tensor] = None,
                                        motion_corr: Optional[dict] = None):
        """w2cs (batch, cameras, 4, 4), Ks (batch, cameras, 3, 3) pixels, times (batch, cameras) seconds,
        key_has_motion (batch, cameras) bool (False for the target view), motion_corr: GALILEO inputs (LVSM.forward)."""
        self.key_has_motion = key_has_motion
        if times is not None:
            times = hp(times)
        self.times = times
        self.time_active = times is not None and bool((times != times[:, :1]).any())
        self.corr_inputs = motion_corr if self.gcfg.enabled else None
        self._corr_state = None  # matches of the matching layer, kept until the next forward (checkpoint recompute)
        self.corr_captures = {}
        (batch, num_cameras, _, _) = w2cs.shape
        self.batch = batch
        self.num_cameras = num_cameras
        self.w2cs = w2cs
        self.c2ws = _invert_SE3(w2cs)
        Ks_norm = normalize_K(Ks, self.image_width, self.image_height)
        self.P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), w2cs)
        self.P_inv = torch.einsum("...ij,...jk->...ik", self.c2ws, _lift_K(_invert_K(Ks_norm)))
        self.p0_world = _get_cam_centers(self.c2ws, self.num_patches).unsqueeze(-2)  # (B, C, P, 1, 4)

    # ---------------------------------------------------------------- per layer
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, predicted_d: torch.Tensor,
                layer_idx: Optional[int] = None, corr_feats: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """q, k, v (batch, num_heads, seqlen, head_dim); predicted_d (batch, seqlen, 2) log depth and log width;
        corr_feats (batch, seqlen, corr_dim) at the matching layer; kwargs go to scaled_dot_product_attention."""
        if corr_feats is not None:
            self._corr_from_features(corr_feats, layer_idx, predicted_d)
        motion = self._motion_from_corr(predicted_d, layer_idx) if self.gcfg.enabled and self.transport else None
        apply_fn_q, all_apply_fns_kv, apply_fn_o = self._prepare_apply_fns(
            predicted_d=predicted_d, motion=motion, use_dyn=motion is not None, time_active=self.time_active)
        return self.rayrope_dot_product_attention(q, k, v, num_cameras=self.num_cameras, apply_fn_q=apply_fn_q,
                                                  all_apply_fns_kv=all_apply_fns_kv, apply_fn_o=apply_fn_o, **kwargs)

    def _corr_from_features(self, feats, layer_idx=None, predicted_d=None):
        """GALILEO matching at the matching layer: coarse window search around the static reprojection (predicted
        depth, detached), then the pixel-resolution fine stage."""
        ci = self.corr_inputs
        if ci is None:
            return
        cfg = self.gcfg
        assert cfg.corr_center == "reproj", "RayRoPE + GALILEO centres the coarse window on the static reprojection"
        B, P = self.batch, self.num_patches
        partners = ci["partners"]
        v1 = partners.shape[1]
        f = feats[:, : v1 * P].reshape(B, v1, P, -1)
        log_d = token_logd(predicted_d.reshape(B, self.num_cameras, P, -1))[:, :v1].detach()
        centres, _ = corr_static_centres(log_d, partners, self.P, self.P_inv, self.patches_x, self.patches_y,
                                         max_log_d=MAX_LOG_DEPTH)
        match, _ = corr_matches_at(f, partners, centres, self.patches_x, self.patches_y, cfg.corr_window, cfg.corr_temp)
        coarse = None
        if cfg.corr_fine_radius > 0 and ci.get("fine_feat") is not None:
            coarse = match
            match, _ = corr_refine(ci["fine_feat"], coarse.detach(), partners, self.patches_x, self.patches_y,
                                   self.image_width, self.image_height, cfg.corr_fine_radius, cfg.corr_fine_spacing,
                                   cfg.corr_fine_temp)
        self._corr_state = match
        if layer_idx is not None:
            self.corr_captures = {"match": match, "match_coarse": coarse}

    def _motion_from_corr(self, predicted_d, layer_idx=None) -> Optional[MotionInputs]:
        """Velocity of every input token from its matches and this layer's depths (Eq. 4), and its displacement to
        every query time. None before the matching layer."""
        cfg = self.gcfg
        ci = self.corr_inputs
        if ci is None or self._corr_state is None or self.times is None or predicted_d is None:
            return None
        if layer_idx is not None and layer_idx < cfg.corr_layer:
            return None
        match = self._corr_state
        B, C, P = self.batch, self.num_cameras, self.num_patches
        v1 = match.shape[1]
        log_d = token_logd(predicted_d.reshape(B, C, P, -1))[:, :v1]
        match_w = None
        if ci.get("ext_match") is not None:  # external (RAFT) matches instead of the model's, failures dropped
            match, match_w = ci["ext_match"].to(match.dtype), ci["ext_match_ok"].float()
        vel, _ = corr_velocity(log_d, match, ci["partners"], ci["pvalid"], self.P_inv, self.times, self.patches_x,
                               self.patches_y, max_log_d=MAX_LOG_DEPTH, depth_sample=cfg.corr_depth_sample,
                               match_w=match_w)
        if ci.get("gt_alpha") is not None:  # training curriculum: GT velocity blended into the estimate
            a = ci["gt_alpha"].to(vel.dtype)[:, None, None, None]
            gt = torch.where(ci["gt_vel_ok"][..., None], ci["gt_vel"].to(vel.dtype), vel)
            vel = a * gt + (1 - a) * vel
        vel = torch.cat([vel, vel.new_zeros(B, C - v1, P, 3)], dim=1)
        disp = vel[:, None] * time_deltas(hp(self.times)).to(vel.dtype)[..., None, None]  # (B, Cq, Cs, P, 3)
        pi = vel.new_ones(B, v1, P)  # every input token moves with its estimated velocity (pi = 1)
        pi = torch.cat([pi.to(vel.dtype), vel.new_zeros(B, C - v1, P)], dim=1)
        motion = MotionInputs(disp=disp, pi=pi)
        if self.key_has_motion is not None:
            motion = mask_key_cameras(motion, self.key_has_motion)
        return motion

    def _pd_positions_pj(self, points, P_q, w2c_q):
        pd_dir, pd_d = _transform_to_query_frame(points, P_q, w2c_q)
        pd_dir = pd_dir.flatten(0, 1)[..., :2].flatten(start_dim=-2, end_dim=-1)
        pd_d = pd_d.flatten(0, 1).flatten(start_dim=-2, end_dim=-1)
        return {"pd_dir": pd_dir, "pd_depth": pd_d}

    def _dynamic_kv_coeffs(self, cam_idx, pd_world, positions_KV, cos_s, sin_s, motion, time_delta):
        """Key coefficients for query camera cam_idx with the depth points of moving keys displaced by u, mixed with
        the static ones by pi (phasor_mix)."""
        batch, num_cameras, num_patches = self.batch, self.num_cameras, self.num_patches
        P_q = self.P[:, cam_idx, :, :]
        w2c_q = self.w2cs[:, cam_idx, :, :]
        pi = motion.pi  # (B, C, P)
        active = pi > 0
        u = motion.disp[:, cam_idx]  # (B, C, P, 3)
        u = torch.where(active[..., None], u, torch.zeros_like(u))
        u_b = u[None, :, :, :, None, :]  # (1, B, C, P, 1, 3): same u for both endpoints and all rays
        pd_dyn = displace_homogeneous(pd_world, u_b)
        positions_dyn = dict(positions_KV)  # camera-centre entries are shared with the static branch
        positions_dyn.update(self._pd_positions_pj(pd_dyn, P_q, w2c_q))
        cos_d, sin_d = _prepare_rope_coeff_uniformd(positions_dyn, self.num_rope_freqs, self.freq_base, batch,
                                                    num_cameras, num_patches, time_delta=time_delta,
                                                    time_chan=self.time_chan)
        slot_mask = _rope_slot_mask(positions_KV, self.num_rope_freqs, device=cos_s.device)
        return phasor_mix(cos_s, sin_s, cos_d, sin_d, pi.reshape(batch, num_cameras * num_patches), slot_mask)

    def rayrope_dot_product_attention(self, q, k, v, num_cameras: int, apply_fn_q: Callable, all_apply_fns_kv: List,
                                      apply_fn_o: Callable, **kwargs) -> torch.Tensor:
        """Per query camera: its queries attend to the keys / values encoded relative to it."""
        (batch, num_heads, seqlen, head_dim) = q.shape
        num_patches = seqlen // num_cameras
        out = torch.zeros_like(q)
        q = apply_fn_q(q)
        for cam_idx, apply_fn_kv in enumerate(all_apply_fns_kv):
            q_idx = q[:, :, cam_idx * num_patches: (cam_idx + 1) * num_patches, :]
            k_idx = apply_fn_kv(k)
            v_idx = apply_fn_kv(v)
            out_idx = F.scaled_dot_product_attention(query=q_idx.contiguous(), key=k_idx.contiguous(),
                                                     value=v_idx.contiguous(), **kwargs)
            out[:, :, cam_idx * num_patches: (cam_idx + 1) * num_patches, :] = out_idx
        out = apply_fn_o(out)
        return out.contiguous()

    @torch.compile()
    def _prepare_apply_fns(self, predicted_d: torch.Tensor, motion: Optional[MotionInputs] = None,
                           use_dyn: bool = False, time_active: bool = False):
        """RoPE transforms of the queries, of the keys / values for every query camera, and of the outputs."""
        batch = self.batch
        num_cameras = self.num_cameras
        patches_x, patches_y = self.patches_x, self.patches_y
        num_patches = patches_x * patches_y
        depths = _prepare_depths(predicted_d, batch=batch, num_cameras=num_cameras, num_patches=num_patches,
                                 num_rays_per_patch=self.num_rays_per_patch)
        pd_world = _get_point_coords(self.P_inv, patches_x, patches_y, self.offsets, depths)  # (2, B, C, P, R, 4)
        positions_Q = {"p0": [], "pd_dir": [], "pd_depth": []}
        all_apply_fns_kv = []
        for cam_idx in range(num_cameras):
            positions_KV = {}
            P_q = self.P[:, cam_idx, :, :]
            w2c_q = self.w2cs[:, cam_idx, :, :]
            # camera centres: 3D in the query camera
            p0_3d = _transform_to_query_frame(self.p0_world, P_q, w2c_q, "3d").flatten(-2, -1)
            positions_KV["p0"] = p0_3d
            positions_Q["p0"].append(p0_3d[:, cam_idx])
            # depth points: projected into the query camera (direction and depth), both interval ends
            pd_dir, pd_d = _transform_to_query_frame(pd_world, P_q, w2c_q, "pj")
            pd_dir = pd_dir.flatten(0, 1)[..., :2].flatten(start_dim=-2, end_dim=-1)
            pd_d = pd_d.flatten(0, 1).flatten(start_dim=-2, end_dim=-1)
            positions_KV["pd_dir"] = pd_dir
            positions_Q["pd_dir"].append(pd_dir[:, cam_idx])
            positions_KV["pd_depth"] = pd_d
            positions_Q["pd_depth"].append(pd_d[:, cam_idx])

            time_delta = None  # key time relative to the query camera: t_s - t_q (the query itself sits at 0)
            if time_active:
                time_delta = (self.times - self.times[:, cam_idx:cam_idx + 1])[:, :, None].expand(
                    batch, num_cameras, num_patches)
            cos_KV, sin_KV = _prepare_rope_coeff_uniformd(positions_KV, self.num_rope_freqs, self.freq_base, batch,
                                                          num_cameras, num_patches, time_delta=time_delta,
                                                          time_chan=self.time_chan)
            if use_dyn:  # GALILEO: depth points of the moving keys at the query time
                cos_KV, sin_KV = self._dynamic_kv_coeffs(cam_idx, pd_world, positions_KV, cos_KV, sin_KV, motion,
                                                         time_delta)
            all_apply_fns_kv.append(partial(_apply_rope_coeffs, cos=cos_KV, sin=sin_KV, inverse=True))
        positions_Q = {key: torch.stack(val, dim=1) for key, val in positions_Q.items()}
        cos_Q, sin_Q = _prepare_rope_coeff_uniformd(positions_Q, self.num_rope_freqs, self.freq_base, batch,
                                                    num_cameras, num_patches, time_delta=None,
                                                    time_chan=self.time_chan)
        apply_fn_q = partial(_apply_rope_coeffs, cos=cos_Q, sin=sin_Q, inverse=True)
        apply_fn_o = partial(_apply_rope_coeffs, cos=cos_Q, sin=sin_Q, inverse=False)
        return apply_fn_q, all_apply_fns_kv, apply_fn_o


def _transform_to_query_frame(points_world: torch.Tensor, P: torch.Tensor, w2c: torch.Tensor,
                              transform_type: str = "pj"):
    """points_world (..., batch, cameras, patches, rays, 4) -> "3d": xyz in the query camera; "pj": projected
    direction (unit) and depth z / w."""
    if transform_type == "3d":
        points_cam = torch.einsum("bij,...bcprj->...bcpri", w2c, points_world)
        points_cam = points_cam / torch.clamp(points_cam[..., -1:], min=1e-4)
        return points_cam[..., :3]
    points_cam = torch.einsum("bij,...bcprj->...bcpri", P, points_world)
    safe_abs = torch.sqrt(points_cam[..., 2:3].pow(2) + 1e-9)
    z = torch.clamp(safe_abs, min=1e-4)
    w = torch.clamp(points_cam[..., -1:], min=1e-4)
    pd_dir = points_cam[..., :3] / (torch.norm(points_cam[..., :3], dim=-1, keepdim=True) + 1e-6)
    return pd_dir, z / w


def _get_cam_centers(c2ws: torch.Tensor, num_patches: int) -> torch.Tensor:
    """Camera centres in homogeneous coordinates, one per patch: (batch, cameras, patches, 4)."""
    batches, num_cameras = c2ws.shape[:2]
    cam_centers = c2ws[:, :, :, 3]
    return cam_centers.view(batches, num_cameras, 1, 4).expand(batches, num_cameras, num_patches, 4)


def _get_point_coords(P_inv: torch.Tensor, patches_x: int, patches_y: int, offsets: List[Tuple[float, float]],
                      depths: torch.Tensor) -> torch.Tensor:
    """World points (homogeneous, w = disparity) on every ray at depths (2, batch, cameras, patches, rays, 1):
    (2, batch, cameras, patches, rays, 4)."""
    device = P_inv.device
    batches, num_cameras = P_inv.shape[:2]
    num_patches = patches_x * patches_y
    num_rays_per_patch = len(offsets)
    u_base, v_base = torch.meshgrid(torch.arange(patches_x, device=device), torch.arange(patches_y, device=device),
                                    indexing="xy")
    coords = []
    for offset in offsets:
        u = ((u_base + offset[0]) / patches_x) - 0.5  # normalized K: principal point at 0
        v = ((v_base + offset[1]) / patches_y) - 0.5
        coords.append(torch.stack([u, v], dim=-1).reshape(-1, 2))
    coords = torch.stack(coords, dim=1)  # (patches, rays, 2)
    coords = coords.view(1, 1, num_patches, num_rays_per_patch, 2).expand(batches, num_cameras, -1, -1, -1)
    depths = torch.clamp(depths, min=1e-2, max=MAX_DEPTH)
    disparity = 1 / depths
    assert depths.shape[1:] == (batches, num_cameras, num_patches, num_rays_per_patch, 1)
    coords = coords.unsqueeze(0).expand(depths.shape[0], -1, -1, -1, -1, -1)
    coords_4d = torch.cat([coords, torch.ones_like(disparity), disparity], dim=-1)
    return torch.einsum("bcij,ebcprj->ebcpri", P_inv, coords_4d)


def _prepare_depths(predicted_d: torch.Tensor, batch: int, num_cameras: int, num_patches: int,
                    num_rays_per_patch: int) -> torch.Tensor:
    """predicted_d (batch, seqlen, 2) log depth d and log width s -> interval ends exp(d -+ s) (clamped at
    exp(MAX_LOG_DEPTH)): (2, batch, cameras, patches, rays, 1)."""
    predicted_d = predicted_d.reshape(batch, num_cameras, num_patches, 1, 2)
    predicted_logd = predicted_d[..., 0:1]
    predicted_sigma = predicted_d[..., 1:2]
    predicted_d1 = torch.exp(torch.clamp(predicted_logd - predicted_sigma, max=MAX_LOG_DEPTH))
    predicted_d2 = torch.exp(torch.clamp(predicted_logd + predicted_sigma, max=MAX_LOG_DEPTH))
    depths = torch.stack([predicted_d1, predicted_d2], dim=0)
    depths = depths.expand(-1, -1, -1, -1, num_rays_per_patch, -1).contiguous()
    if torch.isnan(depths).any() or torch.isinf(depths).any():
        raise ValueError("NaN/inf values found in predicted depths.")
    return depths


@torch.compile
def _prepare_rope_coeff_uniformd(positions: dict, num_freqs: int, freq_base: float, batch: int, num_cameras: int,
                                 num_patches: int, time_delta: Optional[torch.Tensor] = None, time_chan=None):
    """RoPE cos / sin of every coordinate and frequency, (batch, cameras * patches, num_freqs * coord_dim).
    Coordinates with two values (the depth interval ends, stacked on the batch axis) get the coefficients averaged
    uniformly over the interval in closed form. time_chan (parsed planes, omega): those planes carry
    omega * time_delta (0 without time_delta) instead of their geometric angle."""
    coord_dim = 0
    for key, value in positions.items():
        coord_dim += value.shape[-1]
        device = value.device
    cosine_list = []
    sine_list = []
    for pos_name, pos in positions.items():
        max_period, clamp_min, clamp_max = _coord_spec(pos_name)
        if clamp_min is not None:
            pos = torch.clamp(pos, min=clamp_min, max=clamp_max)
        min_period = max_period / (freq_base ** (num_freqs - 1))
        freqs = _get_frequency(num_freqs, max_period=max_period, min_period=min_period).to(device)
        rope_angle = torch.einsum("f,bcpd->bcpfd", freqs, pos)
        if time_chan is not None:
            mask = _time_freq_mask(pos_name, time_chan[0], num_freqs, rope_angle.shape[-1])
            if mask is not None:
                chan_mask = torch.tensor(mask, dtype=torch.bool, device=device)  # (num_freqs, num_coords)
                if time_delta is None:
                    t_angle = torch.zeros((), dtype=rope_angle.dtype, device=device)
                else:
                    dt = time_delta if rope_angle.shape[0] == batch else torch.cat([time_delta, time_delta], dim=0)
                    t_angle = (time_chan[1] * hp(dt))[..., None, None].to(rope_angle.dtype)
                rope_angle = torch.where(chan_mask, t_angle, rope_angle)
        if rope_angle.shape[0] == batch * 2:  # interval: E[cos], E[sin] of a uniform angle between the two ends
            rope_angles1 = rope_angle[:batch]
            rope_angles2 = rope_angle[batch:]
            same_mask = torch.isclose(rope_angles1, rope_angles2, atol=1e-2, rtol=0)
            cosine1 = torch.cos(rope_angles1)
            cosine2 = torch.cos(rope_angles2)
            sine1 = torch.sin(rope_angles1)
            sine2 = torch.sin(rope_angles2)
            delta = rope_angles2 - rope_angles1
            delta_safe = torch.where(same_mask, torch.ones_like(delta), delta)
            E_cosine = (sine2 - sine1) / delta_safe
            E_sine = (cosine1 - cosine2) / delta_safe
            cosine_final = torch.where(same_mask, cosine1, E_cosine)
            sine_final = torch.where(same_mask, sine1, E_sine)
            assert cosine_final.abs().max() <= 1.0 + 1e-2
            assert sine_final.abs().max() <= 1.0 + 1e-2
        elif rope_angle.shape[0] == batch:
            cosine_final = torch.cos(rope_angle)
            sine_final = torch.sin(rope_angle)
        else:
            raise ValueError(f"unexpected leading dimension {rope_angle.shape[0]} for batch {batch}")
        cosine_list.append(cosine_final)
        sine_list.append(sine_final)
    cosine_out = torch.cat(cosine_list, dim=-1).reshape(batch, num_cameras * num_patches,
                                                        num_freqs * coord_dim).contiguous()
    sine_out = torch.cat(sine_list, dim=-1).reshape(batch, num_cameras * num_patches,
                                                    num_freqs * coord_dim).contiguous()
    return cosine_out, sine_out


def _coord_spec(pos_name: str):
    """(max period, clamp min, clamp max) of a coordinate."""
    if pos_name in ["p0", "pd_3d"]:
        return 1.0 * 4, None, None
    if pos_name in ["pinf_dir", "pd_dir", "p0_dir"]:
        return 2.0 * 4, None, None
    if pos_name in ["pd_depth", "p0_depth"]:
        return MAX_D_F * 2 * 4, -MAX_D_F, MAX_D_F
    raise ValueError(f"unknown coordinate {pos_name}")


def _rope_slot_mask(positions: dict, num_freqs: int, prefix: str = "pd_", device=None) -> torch.Tensor:
    """(num_freqs * coord_dim,) bool: True on the slots of the coordinates named prefix* (the depth points)."""
    per_coord = []
    for name, pos in positions.items():
        per_coord += [name.startswith(prefix)] * pos.shape[-1]
    return torch.tensor(per_coord * num_freqs, dtype=torch.bool, device=device)


def _time_freq_mask(pos_name: str, time_planes, num_freqs: int, num_coords: int):
    """(num_freqs, num_coords) nested list of the planes of this coordinate that carry time, or None."""
    if time_planes is None:
        return [[True] * num_coords for _ in range(num_freqs)]
    for prefix, freqs, coords in time_planes:
        if pos_name == prefix or pos_name.startswith(prefix + "_"):
            if coords is None:
                cm = [True] * num_coords
            elif coords == "even":
                cm = [c % 2 == 0 for c in range(num_coords)]
            elif coords == "odd":
                cm = [c % 2 == 1 for c in range(num_coords)]
            else:
                cm = [c in coords for c in range(num_coords)]
            return [[(freqs is None or f in freqs) and c for c in cm] for f in range(num_freqs)]
    return None


def _get_frequency(num_freqs: int, max_period: float, min_period: float):
    log_min_frequency = torch.log(torch.tensor(2 * torch.pi / max_period))
    log_max_frequency = torch.log(torch.tensor(2 * torch.pi / min_period))
    log_freqs = torch.linspace(log_min_frequency, log_max_frequency, num_freqs)
    return torch.exp(log_freqs)


@torch.compile
def _apply_rope_coeffs(feats: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False):
    """feats (batch, num_heads, seqlen, feat_dim), cos / sin (batch, seqlen, feat_dim // 2); split (not interleaved)
    halves."""
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    assert cos.shape == (batch, seqlen, feat_dim // 2)
    assert sin.shape == (batch, seqlen, feat_dim // 2)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x1, x2 = torch.chunk(feats, 2, dim=-1)
    if inverse:
        x_rotated_1 = x1 * cos + x2 * sin
        x_rotated_2 = -x1 * sin + x2 * cos
    else:
        x_rotated_1 = x1 * cos - x2 * sin
        x_rotated_2 = x1 * sin + x2 * cos
    out = torch.cat([x_rotated_1, x_rotated_2], dim=-1)
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def normalize_K(Ks: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    """Pixel intrinsics -> normalized (focal / size, principal point / size - 0.5)."""
    Ks_norm = torch.zeros_like(Ks)
    Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
    Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
    Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
    Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
    Ks_norm[..., 2, 2] = 1.0
    return Ks_norm
