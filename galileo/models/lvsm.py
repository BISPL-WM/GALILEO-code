"""Decoder-only LVSM (Jin et al., arXiv 2410.17242) with a choice of positional encodings, and GALILEO.

Input views are tokenized from image + ray patches, target views from ray patches only; one transformer attends over
all input tokens and the tokens of one target view (each target view is rendered separately). The camera / time
relation of every token pair enters through the attention's positional encoding:
    none     Plücker rays in the tokens and a learned time embedding (the "Plücker" baseline)
    prope    PRoPE (projective, image-plane RoPE), gta: GTA (PRoPE without intrinsics), urope: URoPE (depth anchors),
    rayrope  RayRoPE (3D points at a predicted depth interval along each key's ray)
each with a temporal RoPE ("chan": a few RoPE planes rotate by omega (t_k - t_q)). GALILEO (galileo.enabled) moves
every key to the query's time before the 3D encoding of RayRoPE / URoPE (galileo/motion.py).
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
from torch import Tensor

from galileo.motion import CorrFineNet, fourier_time_features, parse_time_planes
from galileo.models.camera import Camera, camera_to_raymap, patchify, raymap_to_plucker, unpatchify
from galileo.models.transformer import TransformerConfig, TransformerEncoder
from galileo.pos_enc.prope import PropeDotProductAttention
from galileo.pos_enc.rayrope import RayRoPE_DotProductAttention
from galileo.pos_enc.urope import URoPEDotProductAttention


@dataclass
class GalileoConfig:
    """Temporal encoding of every model, and GALILEO's velocity estimation and key transport."""
    # time signal: "chan" = the time_planes RoPE planes carry omega (t_k - t_q) instead of their geometric angle
    # (RayRoPE "p0:<freq>", PRoPE / GTA / URoPE "x:<freqs>"); "embed" = additive Fourier embedding of t - t_target
    # (Plücker)
    time_enc: str = "chan"
    time_omega: float = 3.3  # rad / s
    time_planes: str = "p0:5"
    time_embed_freqs: int = 8
    # GALILEO
    enabled: bool = False
    corr_layer: int = 1  # matching layer: its input features are matched; layers before it stay static
    corr_dim: int = 64  # matching feature dimension (linear projection of the layer input)
    corr_window: int = 7  # coarse search window (tokens)
    corr_temp: float = 0.05  # coarse soft-argmax temperature
    corr_center: str = "reproj"  # coarse window centre: "reproj" = static reprojection at the predicted depth
    # (RayRoPE), "self" = the token's own grid position (URoPE)
    corr_depth_sample: str = "bilinear"  # depth at the match end: "bilinear" over the partner's token grid / "nearest"
    corr_fine_radius: int = 12  # fine stage: +- radius px window around the coarse match
    corr_fine_spacing: int = 2  # px step of the fine window
    corr_fine_temp: float = 0.02
    corr_fine_dim: int = 32  # CNN feature dimension of the fine stage
    corr_detach: bool = False  # matching features from detached token features (URoPE + GALILEO)


@dataclass
class ModelConfig:
    pos_enc: str = "rayrope"  # none / prope / gta / urope / rayrope
    ray_encoding: str = "camray"  # token rays: "camray" (in the camera frame) / "plucker" (world, Plücker baseline)
    img_size: int = 256
    patch_size: int = 8
    encoder: TransformerConfig = field(default_factory=TransformerConfig)
    galileo: GalileoConfig = field(default_factory=GalileoConfig)
    # RayRoPE
    num_rays_per_patch: int = 3
    freq_base: float = 3.0
    init_d: float = 0.0  # initial log depth
    init_sig: float = 3.0  # initial log interval width
    # URoPE
    urope_min_depth: float = 2.0
    urope_max_depth: float = 20.0
    urope_group_size: int = 4

    @property
    def depth_type(self) -> str:
        return "predict_dsig" if self.pos_enc == "rayrope" else "none"


def _x_time_freqs(gcfg: GalileoConfig) -> Optional[Tuple[int, ...]]:
    """time_enc chan on PRoPE / GTA / URoPE: the x-RoPE frequencies given over to time (time_planes "x:<freqs>")."""
    if gcfg.time_enc != "chan":
        return None
    planes = parse_time_planes(gcfg.time_planes)
    if planes is None or planes[0][0] != "x":
        return None
    assert len(planes) == 1 and planes[0][1] is not None, f"time planes on the x RoPE: x:<freqs>, got {gcfg.time_planes}"
    return planes[0][1]


class LVSM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        gcfg = config.galileo
        enc = config.encoder
        head_dim = enc.d_model // enc.nhead
        res, ps = config.img_size, config.patch_size
        grid = dict(patches_x=res // ps, patches_y=res // ps, image_width=res, image_height=res)
        if config.pos_enc == "rayrope":
            self.attention = RayRoPE_DotProductAttention(head_dim=head_dim, **grid, pos_enc_type="d_pj+0_3d",
                                                         num_rays_per_patch=config.num_rays_per_patch,
                                                         depth_type="predict_dsig", denc_type="d",
                                                         freq_base=config.freq_base, gcfg=gcfg)
        elif config.pos_enc == "none":
            self.attention = None
        elif config.pos_enc == "urope":
            self.attention = URoPEDotProductAttention(
                head_num=enc.nhead, head_dim=head_dim, **grid, min_depth=config.urope_min_depth,
                max_depth=config.urope_max_depth, group_size=config.urope_group_size,
                time_enc=gcfg.time_enc if gcfg.time_enc == "chan" else "none", time_omega=gcfg.time_omega,
                time_freqs=_x_time_freqs(gcfg), gcfg=gcfg if gcfg.enabled else None)
        elif config.pos_enc in ("prope", "gta"):
            self.attention = PropeDotProductAttention(
                head_dim=head_dim, **grid, time_enc=gcfg.time_enc if gcfg.time_enc == "chan" else "none",
                time_omega=gcfg.time_omega, time_freqs=",".join(map(str, _x_time_freqs(gcfg) or [])) or "all")
        else:
            raise ValueError(f"unknown pos_enc {config.pos_enc!r}")
        assert not gcfg.enabled or config.pos_enc in ("rayrope", "urope"), "GALILEO transports RayRoPE / URoPE keys"

        ray_dim = 6
        self.query_tokenizer = nn.Linear(ray_dim * ps ** 2, enc.d_model, bias=enc.bias)
        self.input_tokenizer = nn.Linear((3 + ray_dim) * ps ** 2, enc.d_model, bias=enc.bias)
        self.encoder = TransformerEncoder(enc, predict_d=config.depth_type, init_depth=config.init_d,
                                          init_sigma=config.init_sig)
        self.corr_fine_net = None
        if gcfg.enabled:
            if config.pos_enc == "urope":  # every layer passes its index (and the matching layer its features)
                for layer in self.encoder.layers:
                    layer.self_attn.urope_motion = True
            attn = self.encoder.layers[gcfg.corr_layer].self_attn
            with torch.random.fork_rng(devices=[]):  # leave the other weights and the data order unchanged
                torch.manual_seed(1234)
                w = torch.empty(gcfg.corr_dim, enc.d_model)
                nn.init.xavier_uniform_(w)
            attn.corr_proj_weight = nn.Parameter(w)
            attn.corr_detach = gcfg.corr_detach
            if gcfg.corr_fine_radius > 0:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(1235)
                    self.corr_fine_net = CorrFineNet(gcfg.corr_fine_dim)
        self.time_embed = None
        if gcfg.time_enc == "embed":  # zero-init additive embedding of t - t_target (Plücker)
            with torch.random.fork_rng(devices=[]):
                self.time_embed = nn.Linear(2 * gcfg.time_embed_freqs, enc.d_model, bias=False)
            nn.init.zeros_(self.time_embed.weight)
        self.output_layer = nn.Linear(enc.d_model, 3 * ps ** 2, bias=enc.bias)
        self.init_weights()

    def init_weights(self):
        # LVSM A.1: normal init with std 0.02 / (2 (idx + 1)) ** 0.5 for the linear layers of transformer layer idx
        for idx, layer in enumerate(self.encoder.layers):
            def _init(m, idx=idx):
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.02 / (2 * (idx + 1)) ** 0.5)
            layer.apply(_init)

    def create_rays(self, cams: Camera) -> Tensor:
        """Ray maps [B, V, H, W, 6]: (direction, origin x direction), in the camera frame for "camray"."""
        cw = cams.camtoworld
        rays = camera_to_raymap(
            Ks=cams.K,
            camtoworlds=(torch.eye(4, dtype=cw.dtype, device=cw.device).broadcast_to(cw.shape)
                         if self.config.ray_encoding == "camray" else cw),
            height=cams.height, width=cams.width, downscale=1)
        return raymap_to_plucker(rays)

    def forward(self, ref_imgs: Tensor, ref_cams: Camera, tar_cams: Camera, ref_times: Optional[Tensor] = None,
                tar_times: Optional[Tensor] = None, motion_corr: Optional[dict] = None) -> Tensor:
        """ref_imgs [B, V1, H, W, 3] in [0, 1]; ref_times [B, V1], tar_times [B, V2] seconds; motion_corr (GALILEO):
        partners / pvalid [B, V1, Q] (galileo.motion.corr_partners) and, in training, the curriculum's gt_vel
        [B, V1, P, 3], gt_vel_ok [B, V1, P], gt_alpha [B], or ext_match / ext_match_ok (external matches).
        Returns logits [B, V2, H, W, 3] (sigmoid -> image)."""
        config = self.config
        batch_size, v2 = tar_cams.camtoworld.shape[:2]
        ref_rays = self.create_rays(ref_cams)
        tar_rays = self.create_rays(tar_cams)
        if self.corr_fine_net is not None and motion_corr is not None:  # pixel features of the fine matching stage
            imgs_flat = ref_imgs.flatten(0, 1)
            fine = (torch.utils.checkpoint.checkpoint(self.corr_fine_net, imgs_flat, use_reentrant=False)
                    if torch.is_grad_enabled() else self.corr_fine_net(imgs_flat))
            motion_corr = {**motion_corr, "fine_feat": fine.reshape(*ref_imgs.shape[:2], *fine.shape[1:])}
        ref_imgs = patchify(ref_imgs, config.patch_size)
        ref_rays = patchify(ref_rays, config.patch_size)
        tar_rays = patchify(tar_rays, config.patch_size)
        x = self.input_tokenizer(torch.cat([ref_imgs, ref_rays], dim=-1))
        x = repeat(x, "b v1 n d -> (b v2) (v1 n) d", v2=v2)
        q = self.query_tokenizer(tar_rays)
        q = rearrange(q, "b v2 n d -> (b v2) n d")
        q_tokens = q.shape[1]

        # per target view: the input cameras then the target camera
        ref_c2ws = repeat(ref_cams.camtoworld, "b v1 x y -> (b v2) v1 x y", v2=v2)
        ref_Ks = repeat(ref_cams.K, "b v1 x y -> (b v2) v1 x y", v2=v2)
        tar_c2ws = rearrange(tar_cams.camtoworld, "b v2 x y -> (b v2) 1 x y", v2=v2)
        tar_Ks = rearrange(tar_cams.K, "b v2 x y -> (b v2) 1 x y")
        c2ws = torch.cat([ref_c2ws, tar_c2ws], dim=1)
        Ks = torch.cat([ref_Ks, tar_Ks], dim=1)
        viewmats = torch.inverse(c2ws)
        v1 = ref_imgs.shape[1]
        times = None
        if ref_times is not None and tar_times is not None:
            times = torch.cat([repeat(ref_times, "b v1 -> (b v2) v1", v2=v2),
                               rearrange(tar_times, "b v2 -> (b v2) 1")], dim=1)  # [B*V2, V1+1]
        key_has_motion = torch.zeros(batch_size * v2, v1 + 1, dtype=torch.bool, device=x.device)
        key_has_motion[:, :v1] = True  # target tokens are static keys (u = 0, pi = 0)
        corr_seq = None
        if motion_corr is not None:
            corr_seq = {k: repeat(val, "b ... -> (b v2) ...", v2=v2) for k, val in motion_corr.items()
                        if val is not None}
        if self.time_embed is not None and times is not None:
            dt = times - times[:, -1:]  # relative to the target time: time-shift invariant
            emb = self.time_embed(fourier_time_features(dt, config.galileo.time_embed_freqs).to(x.dtype))
            x = x + repeat(emb[:, :v1], "bv v1 d -> bv (v1 n) d", n=x.shape[1] // v1)
            q = q + emb[:, v1:]

        if isinstance(self.attention, (PropeDotProductAttention, URoPEDotProductAttention)):
            self.attention.times = times
        if isinstance(self.attention, URoPEDotProductAttention) and self.attention.motion:
            self.attention.set_motion_inputs(times, key_has_motion, corr_seq)
        if isinstance(self.attention, RayRoPE_DotProductAttention):
            self.attention._precompute_and_cache_apply_fns(w2cs=viewmats, Ks=Ks, times=times,
                                                           key_has_motion=key_has_motion, motion_corr=corr_seq)

        def sdpa_fn(q, k, v, **sdpa_kwargs):
            if config.pos_enc == "gta":  # GTA is PRoPE without intrinsics
                return self.attention(q, k, v, viewmats=viewmats, Ks=None, **sdpa_kwargs)
            if config.pos_enc in ("prope", "urope"):
                return self.attention(q, k, v, viewmats=viewmats, Ks=Ks, **sdpa_kwargs)
            return self.attention(q, k, v, **sdpa_kwargs)

        xq = torch.cat([x, q], dim=1)
        xq = self.encoder(xq, sdpa_fn=F.scaled_dot_product_attention if config.pos_enc == "none" else sdpa_fn)
        q = xq[:, -q_tokens:, :]
        q = rearrange(q, "(b v) n d -> b v n d", b=batch_size, v=v2)
        o = self.output_layer(q)
        return unpatchify(o, height=config.img_size, width=config.img_size, patch_size=config.patch_size)
