import os
import random

import lightning as L
import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torchmetrics.classification import BinaryAUROC

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


class MMDet(L.LightningModule):
    def __init__(self, config, **kwargs):
        super(MMDet, self).__init__()
        self.config = config
        self.window_size = config["window_size"]
        self.interval = config["interval"]

        if "predict_path" in config:
            self.predict_path = config["predict_path"]
            if "predict_flag" in config:
                self.predict_flag = config["predict_flag"]
            else:
                # Use a local RNG instance to set flag
                rng = random.Random()
                self.predict_flag = rng.randint(0, 2**64)

        if "max_epochs" in config:
            self.max_epochs = config["max_epochs"]
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

        self.train_auc = BinaryAUROC()
        self.validation_auc = BinaryAUROC()

    def forward(
        self,
        original_frames,
        reconstructed_frames,
        visual_feature,
        textual_feature,
    ):
        x_st = self.backbone(
            (original_frames, reconstructed_frames)
        )  # spatial temporal feature
        visual_feature, textual_feature = (
            visual_feature.float(),
            textual_feature.float(),
        )
        x_visual = self.clip_proj(visual_feature)
        x_mm = self.mm_proj(textual_feature).squeeze(1)
        x_feat = torch.cat([x_st, x_visual, x_mm], dim=1)
        x_feat = self.final_fusion(x_feat)
        x_feat = torch.mean(x_feat, dim=1)
        out = self.head(x_feat)
        return out

    def training_step(self, batch):
        (
            video_list,
            original_frames,
            reconstructed_frames,
            visual_feature,
            textual_feature,
            label,
        ) = batch
        logits = self.forward(
            original_frames, reconstructed_frames, visual_feature, textual_feature
        )
        loss = torch.nn.functional.cross_entropy(logits, label)

        y_hat = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
        self.train_auc.update(y_hat, label)

        return {"loss": loss}

    def validation_step(self, batch):
        (
            video_list,
            original_frames,
            reconstructed_frames,
            visual_feature,
            textual_feature,
            label,
        ) = batch
        logits = self.forward(
            original_frames, reconstructed_frames, visual_feature, textual_feature
        )
        loss = torch.nn.functional.cross_entropy(logits, label)

        y_hat = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
        self.validation_auc.update(y_hat, label)

        return {"loss": loss}

    def predict_step(self, batch):
        (
            video_id,
            original_frames,
            reconstructed_frames,
            visual_feature,
            textual_feature,
        ) = batch
        repeat = 2
        center = self.window_size // 2
        video_length = original_frames.shape[1]
        final_logits = [None] * center

        for timestamp in range(0, video_length - self.window_size + 1, repeat):
            logits = self.forward(
                original_frames[
                    :,
                    timestamp : timestamp + self.window_size,
                    :,
                    :,
                    :,
                ],
                reconstructed_frames[
                    :,
                    timestamp : timestamp + self.window_size,
                    :,
                    :,
                    :,
                ],
                visual_feature[
                    :, timestamp // self.interval : timestamp // self.interval + 1, :
                ],
                textual_feature[
                    :, timestamp // self.interval : timestamp // self.interval + 1, :
                ],
            )
            for i in range(repeat):
                if timestamp + i < video_length:
                    final_logits.append(logits)

        for i in range(center):
            final_logits[i] = final_logits[center]

        for i in range(video_length - len(final_logits)):
            final_logits.append(final_logits[-1])

        final_logits = torch.stack(final_logits, dim=1)  # shape: (1, video length, 2)
        final_logits = final_logits.detach().cpu().numpy()

        return {
            os.path.join(self.predict_path, video_id[0]): final_logits[0],
            os.path.join(self.predict_path + "_flag", video_id[0]): np.array(
                [self.predict_flag]
            ),
        }

    def configure_optimizers(self):
        # ###
        # optimizer = Adam(self.parameters(), lr=1e-4)
        # return optimizer

        ###
        # optimizer = Adam(self.parameters(), lr=2e-5, weight_decay=1e-6)
        # scheduler = ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.2, min_lr=1e-8, patience=4, cooldown=5
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "validation_loss", "strict": True},
        # }

        ###
        optimizer = Adam(self.parameters(), weight_decay=1e-6)
        scheduler = OneCycleLR(
            optimizer, max_lr=1e-4, total_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log_dict(
            {
                "train_loss": outputs["loss"],
                "lr": self.trainer.lr_scheduler_configs[
                    0
                ].scheduler.optimizer.param_groups[0]["lr"],
            },
            sync_dist=True,
            prog_bar=True,
        )

    def on_train_epoch_end(self):
        self.log_dict(
            {"train_auc": self.train_auc.compute()}, sync_dist=True, prog_bar=True
        )
        self.train_auc.reset()

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.log_dict(
            {"validation_loss": outputs["loss"]}, sync_dist=True, prog_bar=True
        )

    def on_validation_epoch_end(self):
        self.log_dict(
            {"validation_auc": self.validation_auc.compute()},
            sync_dist=True,
            prog_bar=True,
        )
        self.validation_auc.reset()
