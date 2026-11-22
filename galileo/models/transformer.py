"""Pre-norm transformer encoder whose attention takes a custom scaled-dot-product function (the positional encoding).

Adapted from RayRoPE (github.com/wuyu-wy/RayRoPE, MIT), itself modified from the torch implementation. The attention of
every layer can predict a per-token log depth and log interval width for RayRoPE (`predict_dsig`), and the matching
layer of GALILEO projects its input features to matching features (`corr_proj_weight`, attached by the model).
"""
import copy
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.parameter import Parameter


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight.type_as(x)


class MultiheadAttention(torch.nn.Module):
    """Bidirectional multi-head attention with RMS q/k norm and a custom attention function.

    predict_d "predict_dsig": one log depth and one log interval width per token (shared by the heads), a linear head
    on the layer's input features; RayRoPE places each key's depth points at exp(d -+ sigma).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True,
                 qk_norm: bool = False, predict_d: str = "none", init_depth: float = 0.0, init_sigma: float = 3.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.predict_d = predict_d
        self.capture_depth = False  # set by the trainer: keep the depth head output of the last forward
        self.captured_depth = None
        self.layer_idx = None  # set by TransformerEncoder
        self.urope_motion = False  # set by the model: URoPE + GALILEO (the attention needs the layer index)
        self.dropout = dropout
        self.bias = bias
        self.qk_norm = qk_norm

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if self.bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        if predict_d == "predict_dsig":
            self.d_proj_weight = Parameter(torch.zeros((2, embed_dim)))
            self.d_proj_bias = torch.nn.Parameter(torch.tensor([init_depth, init_sigma]))
        else:
            assert predict_d == "none", predict_d
            self.register_parameter("d_proj_weight", None)
            self.register_parameter("d_proj_bias", None)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            torch.nn.init.constant_(self.in_proj_bias, 0.0)
            torch.nn.init.constant_(self.out_proj.bias, 0.0)

    def corr_features(self, query: Tensor):
        """GALILEO matching features of the matching layer (None elsewhere)."""
        if getattr(self, "corr_proj_weight", None) is None:
            return None
        return F.linear(query.detach() if getattr(self, "corr_detach", False) else query, self.corr_proj_weight)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, sdpa_fn: Callable) -> Tensor:
        """query, key, value (B, T, D); sdpa_fn (q, k, v, **kwargs) -> (B, H, T, C). Returns (B, T, D)."""
        raw_d = None
        if self.predict_d != "none":
            raw_d = F.linear(query, self.d_proj_weight, self.d_proj_bias)
            if self.capture_depth:
                self.captured_depth = raw_d

        q_proj_weight, k_proj_weight, v_proj_weight = self.in_proj_weight.chunk(3, dim=0)
        if self.in_proj_bias is not None:
            q_proj_bias, k_proj_bias, v_proj_bias = self.in_proj_bias.chunk(3)
        else:
            q_proj_bias = k_proj_bias = v_proj_bias = None
        q = F.linear(query, q_proj_weight, q_proj_bias)
        k = F.linear(key, k_proj_weight, k_proj_bias)
        v = F.linear(value, v_proj_weight, v_proj_bias)
        q, k, v = (rearrange(x, "b t (h c) -> b t h c", h=self.num_heads) for x in [q, k, v])
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k, v = (rearrange(x, "b t h c -> b h t c") for x in [q, k, v])

        dropout_p = self.dropout if self.training else 0.0
        extra = {}
        corr = self.corr_features(query)
        if corr is not None:
            extra["corr_feats"] = corr
        if self.predict_d != "none":  # RayRoPE
            o = sdpa_fn(q, k, v, dropout_p=dropout_p, predicted_d=raw_d, layer_idx=self.layer_idx, **extra)
        elif self.urope_motion:  # URoPE + GALILEO
            o = sdpa_fn(q, k, v, dropout_p=dropout_p, layer_idx=self.layer_idx, **extra)
        else:
            o = sdpa_fn(q, k, v, dropout_p=dropout_p)
        o = rearrange(o, "b h t c -> b t (h c)", h=self.num_heads)
        return F.linear(o, self.out_proj.weight, self.out_proj.bias)


@dataclass
class TransformerConfig:
    d_model: int = 1152
    nhead: int = 8
    dim_feedforward: int = 1024
    num_layers: int = 6
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    bias: bool = False
    qk_norm: bool = True
    checkpointing: bool = True  # gradient checkpointing of every layer


class TransformerEncoderLayer(Module):
    """Pre-norm layer: x + attn(norm1(x)), then x + ffn(norm2(x)) with ReLU."""

    def __init__(self, cfg: TransformerConfig, predict_d: str = "none", init_depth: float = 0.0,
                 init_sigma: float = 3.0) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(cfg.d_model, cfg.nhead, dropout=cfg.dropout, bias=cfg.bias,
                                            qk_norm=cfg.qk_norm, predict_d=predict_d, init_depth=init_depth,
                                            init_sigma=init_sigma)
        self.linear1 = Linear(cfg.d_model, cfg.dim_feedforward, bias=cfg.bias)
        self.dropout = Dropout(cfg.dropout)
        self.linear2 = Linear(cfg.dim_feedforward, cfg.d_model, bias=cfg.bias)
        self.norm1 = LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps, bias=cfg.bias)
        self.norm2 = LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps, bias=cfg.bias)
        self.dropout1 = Dropout(cfg.dropout)
        self.dropout2 = Dropout(cfg.dropout)

    def forward(self, src: Tensor, sdpa_fn: Callable) -> Tensor:
        x = src
        y = self.norm1(x)
        x = x + self.dropout1(self.self_attn(y, y, y, sdpa_fn=sdpa_fn))
        x = x + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(self.norm2(x))))))
        return x


class TransformerEncoder(Module):
    def __init__(self, cfg: TransformerConfig, predict_d: str = "none", init_depth: float = 0.0,
                 init_sigma: float = 3.0) -> None:
        super().__init__()
        layer = TransformerEncoderLayer(cfg, predict_d, init_depth, init_sigma)
        # every layer starts as a copy of the first (the linear weights are re-drawn per layer by the model)
        self.layers = ModuleList([copy.deepcopy(layer) for _ in range(cfg.num_layers)])
        for idx, lay in enumerate(self.layers):
            lay.self_attn.layer_idx = idx
        self.in_norm = LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps, bias=cfg.bias)
        self.out_norm = LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps, bias=cfg.bias)
        self.checkpointing = cfg.checkpointing

    def forward(self, src: Tensor, sdpa_fn: Callable) -> Tensor:
        output = self.in_norm(src)
        for mod in self.layers:
            if self.checkpointing:
                output = torch.utils.checkpoint.checkpoint(mod, output, sdpa_fn=sdpa_fn, use_reentrant=False)
            else:
                output = mod(output, sdpa_fn=sdpa_fn)
        return self.out_norm(output)
