"""Training and evaluation of LVSM with RayRoPE / URoPE / PRoPE / GTA / Plücker and GALILEO on Stereo4D.

Losses: MSE + 0.5 VGG perceptual on the target views; for RayRoPE, L1 on every layer's predicted log depth against the
stereo depth (depth_sup_weight); for GALILEO, the matching loss of both matching stages against the ARROW
correspondences (corr_weight: coarse, corr_fine_weight: fine, dynamic tokens x5), and in training the ground-truth
velocity curriculum: GT velocity for corr_gt_vel_start steps, then linearly annealed to the estimate by corr_gt_vel_end.
"""
import glob
import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import yaml
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from galileo.data.stereo4d import Stereo4DDataset
from galileo.losses.depth import (depth_captures, depth_correspondences, depth_supervision_loss, depth_token_targets,
                                  enable_depth_capture)
from galileo.losses.perceptual import Perceptual
from galileo.models.camera import Camera
from galileo.models.lvsm import LVSM, ModelConfig
from galileo.motion import corr_partners


@dataclass
class TrainConfig:
    output_dir: str = "runs/stereo4d"
    data_dir: str = os.environ.get("STEREO4D_DIR", "data/stereo4d/proc")
    test_index: str = "splits/stereo4d/test_index.json"
    model: ModelConfig = field(default_factory=ModelConfig)
    # data
    n_in_times: int = 4
    gap: int = 5  # frames between input times (30 fps)
    min_partners: int = 1
    protocols: tuple = ("interp", "extrap")
    ext_raft: str = ""  # RAFT variant: tracks file of the external matches (data_raftc512.npz)
    num_workers: int = 12
    # optimization
    batch_size: int = 12
    max_steps: int = 20000
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 2500
    grad_clip: float = 0.0
    amp: bool = True
    amp_dtype: str = "fp16"
    seed: int = 42
    # losses
    perceptual_loss_w: float = 0.5
    depth_sup_weight: float = 0.0  # RayRoPE: 0.1
    corr_weight: float = 0.1  # GALILEO coarse matching loss
    corr_fine_weight: float = 1.0  # GALILEO fine matching loss
    corr_fine_lr: float = 1e-3  # learning rate of the fine-stage CNN
    corr_gt_vel_start: int = 5000  # GALILEO curriculum: GT velocity until this step ...
    corr_gt_vel_end: int = 15000  # ... then linearly annealed to the estimate by this step (0: no curriculum)
    corr_partner_max_gap: float = 1.5  # match partners up to this x the nearest partner's time gap
    corr_ext_match: str = ""  # "raft": external RAFT matches instead of the matcher (RAFT variant)
    depth_corr_vis_tol: float = 0.05
    # bookkeeping
    ckpt_every: int = 5000
    ckpt_keeps: int = 3
    test_every: int = 5000
    test_n: Optional[int] = None  # first n test entries per protocol (None: all)
    print_every: int = 10
    visual_every: int = 1000
    resume: Optional[str] = None
    auto_resume: bool = False


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def nested_to_device(data, device):
    if isinstance(data, dict):
        return {k: nested_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [nested_to_device(v, device) for v in data]
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def load_model_weights(model: torch.nn.Module, path: str):
    """Model weights from a checkpoint (.pt with a "model" entry, or a released .safetensors / .pt state dict)."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = sd.get("model", sd)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"checkpoint mismatch: missing {missing}, unexpected {unexpected}"


class Trainer:
    def __init__(self, config: TrainConfig, device: Optional[torch.device] = None):
        self.config = config
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = device or torch.device(f"cuda:{self.local_rank}")
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[config.amp_dtype]
        self.use_grad_scaler = config.amp and config.amp_dtype == "fp16"
        self.output_dir = config.output_dir
        self.ckpt_dir = f"{self.output_dir}/ckpts"
        self.test_dir = f"{self.output_dir}/tests"
        self.visual_dir = f"{self.output_dir}/visuals"
        self.writer = None
        if self.world_rank == 0:
            for d in (self.ckpt_dir, self.test_dir, self.visual_dir):
                os.makedirs(d, exist_ok=True)

    @property
    def galileo(self) -> bool:
        return self.config.model.galileo.enabled

    def log(self, msg: str):
        if self.world_rank == 0:
            print(msg, flush=True)

    # ------------------------------------------------------------------ inputs
    def preprocess(self, data: Dict, input_views: int) -> Dict[str, Any]:
        data = nested_to_device(data, self.device)
        images = data["image"] / 255.0
        Ks, camtoworlds = data["K"], data["camtoworld"]
        _, _, height, width, _ = images.shape
        return {
            "ref_imgs": images[:, :input_views],
            "tar_imgs": images[:, input_views:],
            "ref_cams": Camera(K=Ks[:, :input_views], camtoworld=camtoworlds[:, :input_views], width=width,
                               height=height),
            "tar_cams": Camera(K=Ks[:, input_views:], camtoworld=camtoworlds[:, input_views:], width=width,
                               height=height),
            "ref_times": data["time"][:, :input_views],
            "tar_times": data["time"][:, input_views:],
            "ref_cam_ids": data["cam_id"][:, :input_views],
            "motion": data.get("motion"),
            "eval_gt": data.get("eval_gt"),
            "depth_gt": data.get("depth_gt"),
        }

    def model_kwargs(self, processed) -> Dict[str, Any]:
        """Times and, for GALILEO, the match partners (and external matches) of every input view."""
        kw = {"ref_times": processed["ref_times"], "tar_times": processed["tar_times"]}
        if not self.galileo:
            return kw
        partners, pvalid = corr_partners(processed["ref_times"], processed["ref_cam_ids"])
        if self.config.corr_partner_max_gap > 0:  # near partners only (also the only ones the match loss sees)
            t = processed["ref_times"]
            dt = (torch.gather(t, 1, partners.flatten(1)).view_as(partners) - t[:, :partners.shape[1], None]).abs()
            gmin = torch.where(pvalid, dt, torch.full_like(dt, float("inf"))).flatten(1).min(1).values
            pvalid = pvalid & (dt <= self.config.corr_partner_max_gap * gmin[:, None, None] + 1e-6)
        kw["motion_corr"] = {"partners": partners, "pvalid": pvalid}
        if self.config.corr_ext_match:
            assert self.config.corr_ext_match == "raft", self.config.corr_ext_match
            motion = processed["motion"]
            P = motion["raft_uv"].shape[-2]
            g = partners.clamp_min(0)[..., None, None]
            kw["motion_corr"]["ext_match"] = torch.gather(motion["raft_uv"].float(), 2, g.expand(-1, -1, -1, P, 2))
            kw["motion_corr"]["ext_match_ok"] = torch.gather(motion["raft_uv_ok"], 2, g[..., 0].expand(-1, -1, -1, P))
        return kw

    def corr_gt_velocity(self, kw, processed, step) -> float:
        """GALILEO curriculum: the GT least-squares velocity (world axes) and its blend weight for motion_corr (URoPE:
        the GT landing pixels, its transport lifting matches at fixed depth anchors). Returns the weight."""
        s, e = self.config.corr_gt_vel_start, self.config.corr_gt_vel_end
        if e <= 0 or "motion_corr" not in kw or step >= e:
            return 0.0
        alpha = 1.0 if step < s else 1.0 - (step - s) / max(e - s, 1)
        m = processed["motion"]
        n_in = m["vel_local"].shape[1]
        R = processed["ref_cams"].camtoworld[:, :n_in, :3, :3].float()
        labeled = m["valid"].bool() & m["vel_valid"].bool()
        v = torch.einsum("bsij,bspj->bspi", R, m["vel_local"].float())
        kw["motion_corr"]["gt_vel"] = torch.where(labeled[..., None], v, torch.zeros_like(v))
        kw["motion_corr"]["gt_vel_ok"] = labeled | (m["pi"] <= 0)
        kw["motion_corr"]["gt_alpha"] = torch.full((v.shape[0],), alpha, device=v.device)
        if self.config.model.pos_enc == "urope" and processed.get("depth_gt") is not None:
            H, W = processed["ref_imgs"].shape[2:4]
            corr = depth_correspondences(processed["depth_gt"], m, processed["ref_cams"].K,
                                         processed["ref_cams"].camtoworld, processed["ref_times"], n_in,
                                         self.config.model.patch_size, vis_tol=self.config.depth_corr_vis_tol)
            partners, pvalid = kw["motion_corr"]["partners"], kw["motion_corr"]["pvalid"]
            P = corr["uv"].shape[3]
            uv = torch.gather(corr["uv"], 2, partners[..., None, None].expand(-1, -1, -1, P, 2))  # (B, V1, Q, P, 2)
            ok = torch.gather(corr["ok"], 2, partners[..., None].expand(-1, -1, -1, P)) & pvalid[..., None]
            size = torch.tensor([W, H], dtype=torch.float32, device=uv.device)
            kw["motion_corr"]["gt_match"] = uv.float() / size - 0.5  # normalized like the matches
            kw["motion_corr"]["gt_match_ok"] = ok
        return alpha

    def corr_match_loss(self, base_model, processed, motion_corr, v2):
        """L1 (token units) between the matches of both stages and the GT landing pixel of every input token in its
        partner views (ARROW displacement + stereo depth, visible only); the fine stage weights dynamic tokens x5."""
        caps = base_model.attention.corr_captures
        n_in = processed["ref_imgs"].shape[1]
        patch = base_model.config.patch_size
        H, W = processed["ref_imgs"].shape[2:4]
        ref_cams = processed["ref_cams"]
        corr = depth_correspondences(processed["depth_gt"], processed["motion"], ref_cams.K, ref_cams.camtoworld,
                                     processed["ref_times"], n_in, patch, vis_tol=self.config.depth_corr_vis_tol)
        partners, pvalid = motion_corr["partners"], motion_corr["pvalid"]
        P = corr["uv"].shape[3]
        uv = torch.gather(corr["uv"], 2, partners[..., None, None].expand(-1, -1, -1, P, 2))  # (B, V1, Q, P, 2)
        ok = torch.gather(corr["ok"], 2, partners[..., None].expand(-1, -1, -1, P)) & pvalid[..., None]
        size = torch.tensor([W, H], dtype=torch.float32, device=uv.device)
        dyn_ok = ok & corr["dyn"][:, :, None]
        wp = ok.float()
        n = wp.sum().clamp_min(1.0)
        pred = (caps["match"][::v2].float() + 0.5) * size  # pixels
        err = (pred - uv).abs().sum(-1) / patch  # token units
        fine_loss = None
        if caps.get("match_coarse") is not None:
            pred_c = (caps["match_coarse"][::v2].float() + 0.5) * size
            loss = ((pred_c - uv).abs().sum(-1) / patch * wp).sum() / n
            wf = wp * (1 + 4 * dyn_ok.float())
            fine_loss = (err * wf).sum() / wf.sum().clamp_min(1.0)
        else:
            loss = (err * wp).sum() / n
        stats = {"loss": float(loss.detach())}
        if fine_loss is not None:
            stats["loss_fine"] = float(fine_loss.detach())
        with torch.no_grad():
            if bool(ok.any()):
                stats["epe_px"] = float((pred - uv).norm(dim=-1)[ok].median())
                if bool(dyn_ok.any()):
                    stats["epe_px_dyn"] = float((pred - uv).norm(dim=-1)[dyn_ok].median())
        return loss, fine_loss, stats

    # ------------------------------------------------------------------ model / data
    def build_model(self) -> LVSM:
        return LVSM(self.config.model).to(self.device)

    def train_dataset(self) -> Stereo4DDataset:
        c = self.config
        return Stereo4DDataset(c.data_dir, split="train", n_in_times=c.n_in_times, gap=c.gap, protocols=c.protocols,
                               min_partners=c.min_partners, eval_gt=False, ext_raft=c.ext_raft,
                               get_depth_all=c.depth_sup_weight > 0 or self.galileo)

    def test_datasets(self) -> Dict[str, Stereo4DDataset]:
        c = self.config
        return {f"dyn_stereo4d_{p}": Stereo4DDataset(
            c.data_dir, split="test", n_in_times=c.n_in_times, gap=c.gap, eval_index=c.test_index, protocols=[p],
            first_n=c.test_n, rank=self.world_rank, world_size=self.world_size, eval_gt=True, get_depth_all=True,
            min_partners=c.min_partners, ext_raft=c.ext_raft) for p in c.protocols}

    def optimizer(self, model):
        """AdamW (0.9, 0.95), weight decay on all but the norms; the fine-stage CNN at its own lr and light decay."""
        c = self.config
        groups = []
        for kind in ("base", "fine"):
            for decay in (True, False):
                ps = [p for n, p in model.named_parameters()
                      if (("corr_fine_net." in n) == (kind == "fine")) and (("norm" not in n) == decay)]
                if ps:
                    wd = 1e-4 if kind == "fine" else c.weight_decay
                    groups.append({"params": ps, "weight_decay": wd if decay else 0.0,
                                   "lr": c.corr_fine_lr if kind == "fine" else c.lr})
        optimizer = torch.optim.AdamW(groups, lr=c.lr, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=c.warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.max_steps - c.warmup_steps)])
        return optimizer, scheduler

    # ------------------------------------------------------------------ training
    def train_loss(self, model, perceptual, data, step: int):
        """Loss of one batch (and the outputs / stats for logging)."""
        c = self.config
        input_views = data["K"].shape[1] - 1  # one target view per training sample
        processed = self.preprocess(data, input_views=input_views)
        ref_imgs, tar_imgs = processed["ref_imgs"], processed["tar_imgs"]
        kw = self.model_kwargs(processed)
        self._corr_gt_alpha = self.corr_gt_velocity(kw, processed, step)
        stats = {}
        with torch.amp.autocast(self.device.type, enabled=c.amp, dtype=self.amp_dtype):
            outputs = torch.sigmoid(model(ref_imgs, processed["ref_cams"], processed["tar_cams"], **kw))
            loss = F.mse_loss(outputs, tar_imgs)
            if c.perceptual_loss_w > 0:
                loss = loss + c.perceptual_loss_w * perceptual(rearrange(outputs, "b v h w c -> (b v) c h w"),
                                                               rearrange(tar_imgs, "b v h w c -> (b v) c h w"))
            base_model = getattr(model, "module", model)
            if self.galileo and base_model.attention.corr_captures and processed.get("depth_gt") is not None:
                corr_loss, corr_fine_loss, stats["corr"] = self.corr_match_loss(base_model, processed,
                                                                                kw["motion_corr"], tar_imgs.shape[1])
                if c.corr_weight > 0:
                    loss = loss + c.corr_weight * corr_loss
                if corr_fine_loss is not None:
                    loss = loss + c.corr_fine_weight * corr_fine_loss
            if c.depth_sup_weight > 0 and processed.get("depth_gt") is not None:
                target_logd = depth_token_targets(processed["depth_gt"], ref_imgs.shape[1],
                                                  base_model.config.patch_size)
                captures = depth_captures(base_model)
                assert captures, "depth supervision needs RayRoPE's predicted depths"
                depth = depth_supervision_loss(captures, target_logd)
                stats["depth"] = {"l1_logd": float(depth["loss"])}
                loss = loss + c.depth_sup_weight * depth["loss"]
        return loss, outputs, processed, stats

    def train(self):
        c = self.config
        set_random_seed(c.seed + self.world_rank)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.local_rank)
        model = self.build_model()
        if c.depth_sup_weight > 0:
            assert enable_depth_capture(model) > 0, "depth supervision needs RayRoPE's predicted depths"
        perceptual = Perceptual().to(self.device)
        optimizer, scheduler = self.optimizer(model)
        dataset = self.train_dataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=c.batch_size, num_workers=c.num_workers,
                                             pin_memory=True, persistent_workers=c.num_workers > 0, shuffle=False,
                                             drop_last=True)
        self.log(f"Stereo4D train clips: {len(dataset.clips)}")
        init_step = self.maybe_resume(model, optimizer, scheduler)
        psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        test_state = self.test_initialize(model) if c.test_every > 0 else None
        self.log(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        if self.world_rank == 0:
            self.writer = SummaryWriter(log_dir=f"{self.output_dir}/tb")
            with open(f"{self.output_dir}/config.yaml", "w") as f:
                yaml.safe_dump(json.loads(json.dumps(asdict(c))), f, sort_keys=False)
        if self.device.type == "cuda":
            torch.distributed.init_process_group(backend="nccl")
        ddp_model = DDP(model, device_ids=[self.local_rank]) if self.world_size > 1 else model
        grad_scaler = torch.amp.GradScaler(device="cuda") if self.use_grad_scaler else None
        dataiter = iter(loader)
        for step in range(init_step, c.max_steps + 1):
            ddp_model.train()
            try:
                data = next(dataiter)
            except StopIteration:
                dataiter = iter(loader)
                data = next(dataiter)
            loss, outputs, processed, stats = self.train_loss(ddp_model, perceptual, data, step)
            if loss.isnan():
                if grad_scaler is None:
                    raise FloatingPointError(f"step {step}: loss is NaN")
                print(f"Warning: [step={step}] rank={self.world_rank} | loss is NaN.")
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                if c.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), c.grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                if c.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), c.grad_clip)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            self.log_step(step, loss, outputs, processed, stats, psnr_fn, scheduler)
            self.save_checkpoint(step, model, optimizer, scheduler)
            if c.test_every > 0 and step % c.test_every == 0 and step > 0:
                self.test_iteration(step, test_state)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def log_step(self, step, loss, outputs, processed, stats, psnr_fn, scheduler):
        c = self.config
        if self.world_rank != 0:
            return
        if c.visual_every > 0 and step % c.visual_every == 0:
            for tag, x in (("outputs", outputs), ("gts", processed["tar_imgs"]), ("inputs", processed["ref_imgs"])):
                grid = rearrange(x[:1].detach().float().clamp(0, 1), "b v h w c -> (b h) (v w) c")
                self.writer.add_image(f"visuals/{tag}", grid.cpu().numpy(), step, dataformats="HWC")
        if step % c.print_every == 0:
            with torch.no_grad():
                psnr = psnr_fn(rearrange(outputs.float(), "b v h w c -> (b v) c h w"),
                               rearrange(processed["tar_imgs"], "b v h w c -> (b v) c h w"))
            self.log(f"Step: {step}, Loss: {loss:.3f}, PSNR: {psnr:.3f}, LR: {scheduler.get_last_lr()[0]:.3e}")
            self.writer.add_scalar("train/loss", loss, step)
            self.writer.add_scalar("train/psnr", psnr, step)
            for group, st in stats.items():
                for k, v in st.items():
                    self.writer.add_scalar(f"train/{group}/{k}", v, step)
            if self.galileo and c.corr_gt_vel_end > 0:
                self.writer.add_scalar("train/corr/gt_alpha", self._corr_gt_alpha, step)

    def save_checkpoint(self, step, model, optimizer, scheduler):
        c = self.config
        if self.world_rank != 0 or step == 0 or not (step % c.ckpt_every == 0 or step == c.max_steps):
            return
        torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()}, f"{self.ckpt_dir}/step-{step:09d}.pt")
        for fp in sorted(glob.glob(f"{self.ckpt_dir}/*.pt"))[: -c.ckpt_keeps]:
            os.remove(fp)

    def maybe_resume(self, model, optimizer=None, scheduler=None) -> int:
        c = self.config
        cands = sorted(glob.glob(f"{self.ckpt_dir}/*.pt"), reverse=True) if c.auto_resume else (
            [c.resume] if c.resume else [])
        for path in cands:
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
            except Exception as e:  # a checkpoint cut off by a crash: try the next one
                self.log(f"Error loading checkpoint {path}: {e}")
                continue
            load_model_weights(model, path)
            if optimizer is not None and "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                self.log(f"Resumed from {path} (step {ckpt['step']})")
                return ckpt["step"] + 1
            return ckpt.get("step", 0)
        return 0

    # ------------------------------------------------------------------ evaluation
    def test_initialize(self, model):
        loaders = {label: (len(ds.entries[0]["inputs"]),
                           torch.utils.data.DataLoader(ds, batch_size=1, num_workers=4, pin_memory=True))
                   for label, ds in self.test_datasets().items() if len(ds) > 0}
        return {"model": model, "dataloaders": loaders,
                "psnr_fn": PeakSignalNoiseRatio(data_range=1.0).to(self.device),
                "ssim_fn": StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
                "lpips_fn": LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(self.device)}

    @torch.no_grad()
    def render(self, model, processed):
        """Rendered target views [B, V2, H, W, 3] in [0, 1]."""
        with torch.amp.autocast(self.device.type, enabled=self.config.amp, dtype=self.amp_dtype):
            return torch.sigmoid(model(processed["ref_imgs"], processed["ref_cams"], processed["tar_cams"],
                                       **self.model_kwargs(processed)))

    def _region_metrics(self, outputs, tar_imgs, masks):
        """SSIM map and alex LPIPS spatial map averaged inside each mask {name: (b, v, H, W, 1)}; outputs, tar_imgs
        ((b v), 3, H, W) in [0, 1]. nan where a mask is empty."""
        from torchmetrics.functional.image import structural_similarity_index_measure
        from torchmetrics.functional.image.lpips import _LPIPS
        if getattr(self, "_lpips_spatial", None) is None:  # the same weights as the eval's lpips_fn, per pixel
            self._lpips_spatial = _LPIPS(net="alex", spatial=True).to(outputs.device).eval()
        out, tar = outputs.float(), tar_imgs.float()
        _, ssim_map = structural_similarity_index_measure(out, tar, data_range=1.0, return_full_image=True)
        lp_map = self._lpips_spatial(out, tar, normalize=True)
        nan = torch.tensor(float("nan"), device=outputs.device)
        res = {}
        for name, mk in masks.items():
            m = rearrange(mk, "b v h w c -> (b v) c h w").to(outputs.device, torch.float32)
            n = m.sum()
            res[f"ssim_{name}"] = (ssim_map * m).sum() / (3 * n) if n > 0 else nan
            res[f"lpips_{name}"] = (lp_map * m).sum() / n if n > 0 else nan
        return res

    @torch.inference_mode()
    def test_iteration(self, step: int, state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """PSNR / SSIM / LPIPS on the whole image and PSNR, SSIM, LPIPS on the dynamic and static regions, per
        protocol; tests/metrics_<label>.json (means) and tests/per_entry_<label>.json (one value per test entry)."""
        model = state["model"]
        model.eval()
        results = {}
        for label, (input_views, loader) in state["dataloaders"].items():
            per = {k: [] for k in ("psnr", "ssim", "lpips", "psnr_dyn", "psnr_static")}
            for data in loader:
                processed = self.preprocess(data, input_views=input_views)
                outputs = rearrange(self.render(model, processed), "b v h w c -> (b v) c h w")
                tar_imgs = rearrange(processed["tar_imgs"], "b v h w c -> (b v) c h w")
                per["psnr"].append(state["psnr_fn"](outputs, tar_imgs))
                per["ssim"].append(state["ssim_fn"](outputs, tar_imgs))
                per["lpips"].append(state["lpips_fn"](outputs, tar_imgs))
                eval_gt = processed["eval_gt"]
                m = rearrange(eval_gt["dyn_mask"], "b v h w c -> (b v) c h w").expand_as(outputs)
                if m.sum() > 0:
                    mse_dyn = (((outputs - tar_imgs) ** 2) * m).sum() / m.sum()
                    per["psnr_dyn"].append(-10 * torch.log10(mse_dyn.clamp_min(1e-10)))
                else:
                    per["psnr_dyn"].append(torch.tensor(float("nan"), device=outputs.device))
                m = rearrange(eval_gt["static_mask"], "b v h w c -> (b v) c h w").expand_as(outputs)
                mse_st = (((outputs - tar_imgs) ** 2) * m).sum() / m.sum().clamp_min(1.0)
                per["psnr_static"].append(-10 * torch.log10(mse_st.clamp_min(1e-10)))
                masks = {"dyn": eval_gt["dyn_mask"], "static": eval_gt["static_mask"]}
                for k, v in self._region_metrics(outputs, tar_imgs, masks).items():
                    per.setdefault(k, []).append(v)
            metrics = {"label": label, "step": step}
            for k, vals in per.items():
                metrics[k], n = self._distributed_mean(vals)
            metrics["n_total"] = n
            results[label] = metrics
            self.log(f"{label} step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in metrics.items()
                                                          if isinstance(v, float)))
            if self.world_rank == 0:
                with open(f"{self.test_dir}/metrics_{label}.json", "w") as f:
                    json.dump(metrics, f)
                if self.world_size == 1:
                    with open(f"{self.test_dir}/per_entry_{label}.json", "w") as f:
                        json.dump({k: [float(x) for x in v] for k, v in per.items() if v}, f)
                if self.writer is not None:
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            self.writer.add_scalar(f"test/{k}{label}", v, step)
        return results

    def _distributed_mean(self, data):
        if self.world_size == 1:
            collected = torch.tensor([float(x) for x in data], device=self.device, dtype=torch.float32)
        else:
            sizes = [None] * self.world_size
            torch.distributed.all_gather_object(sizes, len(data))
            collected = [torch.empty(s, device=self.device) for s in sizes]
            torch.distributed.all_gather(collected, torch.tensor([float(x) for x in data], device=self.device))
            collected = torch.cat(collected)
        collected = collected[torch.isfinite(collected)]
        return collected.mean().item(), len(collected)

    def test(self):
        """Evaluate config.resume (all test entries unless test_n)."""
        assert self.config.resume, "evaluation needs a checkpoint (resume=...)"
        if self.device.type == "cuda":
            torch.cuda.set_device(self.local_rank)
            if self.world_size > 1 and not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
        model = self.build_model()
        step = self.maybe_resume(model)
        res = self.test_iteration(step, self.test_initialize(model))
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return res
