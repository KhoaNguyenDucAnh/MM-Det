import os

import lightning as L
import numpy as np
import torch
from einops import rearrange
from torch import nn

from .vit.stv_transformer_hybrid import vit_base_r50_s16_224_with_recons_iafa


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DynamicFusion(nn.Module):
    def __init__(self, in_planes):
        super(DynamicFusion, self).__init__()
        self.channel_attn = Attention(dim=in_planes, num_heads=8)

    def forward(self, x, output_weights=False):
        cw_weights = self.channel_attn(x)
        x = x * cw_weights.expand_as(x)
        if output_weights:
            out = x, cw_weights
        else:
            out = x
        return out


class MMDet(nn.Module):
    def __init__(self, config, **kwargs):
        super(MMDet, self).__init__()
        self.window_size = config["window_size"]
        self.st_pretrained = config["st_pretrained"]
        self.st_ckpt = config["st_ckpt"]
        self.lmm_ckpt = config["lmm_ckpt"]
        if (not self.st_ckpt or not os.path.exists(self.st_ckpt)) and config[
            "st_pretrained"
        ]:
            print(
                "Local pretrained checkpoint for Hybrid ViT not found. Using the default interface in timm."
            )
            self.st_ckpt = None
        self.backbone = vit_base_r50_s16_224_with_recons_iafa(
            window_size=config["window_size"],
            pretrained=config["st_pretrained"],
            ckpt_path=self.st_ckpt,
        )
        self.load_mm_encoder = not config["cache_mm"]
        if self.load_mm_encoder:
            self.mm_encoder = MMEncoder(config)
            for m in self.mm_encoder.modules():
                m.required_grad = False
            print("Freeze MM Encoder.")
        self.clip_proj = nn.Linear(1024, 768)
        self.mm_proj = nn.Linear(4096, 768)
        self.final_fusion = DynamicFusion(in_planes=768)
        self.head = nn.Linear(768, 2)

        new_component_list = [
            self.clip_proj,
            self.mm_proj,
            self.final_fusion,
            self.head,
        ]
        for component in new_component_list:
            for m in component.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)

    def forward(self, x_input, cached_features={}):
        x_original, x_recons = x_input
        B = x_original.size(0)
        x_st = self.backbone(x_input)  # spatial temporal feature
        visual_feat = cached_features.get("visual", None)
        textual_feat = cached_features.get("textual", None)
        if not self.load_mm_encoder:
            assert visual_feat is not None and textual_feat is not None
        if visual_feat is None or textual_feat is None:
            visual_feat, textual_feat = self.mm_encoder(x_original)
        visual_feat, textual_feat = visual_feat.float(), textual_feat.float()
        x_visual = self.clip_proj(visual_feat).unsqueeze(1)
        x_mm = self.mm_proj(textual_feat)
        x_feat = torch.cat([x_st, x_visual, x_mm], dim=1)
        x_feat = self.final_fusion(x_feat)
        x_feat = torch.mean(x_feat, dim=1)
        out = self.head(x_feat)
        return out
