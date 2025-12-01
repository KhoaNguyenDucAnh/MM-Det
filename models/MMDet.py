import os

import lightning as L
import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
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
        self.test_auc = BinaryAUROC()

    def forward(
        self, original_frames, reconstructed_frames, visual_feature, textual_feature
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

    def test_step(self, batch):
        (
            video_list,
            original_frames,
            reconstructed_frames,
            visual_feature,
            textual_feature,
            label,
        ) = batch
        final_logits = []
        for timestamp in range(
            0, original_frames.shape[1] - self.window_size + 1, self.window_size
        ):
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
            final_logits.append(logits.unsqueeze(-1).repeat(1, 1, 10))
        final_logits = torch.cat(final_logits, dim=-1)

        diff = original_frames.shape[1] - final_logits.shape[-1]
        if diff > 0:
            last_slice = final_logits[:, :, -1:].repeat(1, 1, diff)
            final_logits = torch.cat([final_logits, last_slice], dim=-1)

        loss = torch.nn.functional.cross_entropy(final_logits, label)
        self.log_dict({"test_loss": loss}, sync_dist=True, prog_bar=True)

        y_hat = torch.nn.functional.softmax(final_logits, dim=-1)[:, 1, :]
        self.test_auc.update(y_hat, label)

        y_hat = y_hat.detach().cpu().numpy()

        return {"loss": loss} | {
            os.path.join("test", video_id): y_hat[index]
            for index, video_id in enumerate(video_list)
        }

    def predict_step(self, batch):
        (
            video_list,
            original_frames,
            reconstructed_frames,
            visual_feature,
            textual_feature,
        ) = batch
        final_logits = []
        for timestamp in range(
            0, original_frames.shape[1] - self.window_size + 1, self.window_size
        ):
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
            final_logits.append(logits.unsqueeze(-1).repeat(1, 1, 10))
        final_logits = torch.cat(final_logits, dim=-1)

        diff = original_frames.shape[1] - final_logits.shape[-1]
        if diff > 0:
            last_slice = final_logits[:, :, -1:].repeat(1, 1, diff)
            final_logits = torch.cat([final_logits, last_slice], dim=-1)

        y_hat = torch.nn.functional.softmax(final_logits, dim=-1)[:, 1, :]
        y_hat = y_hat.detach().cpu().numpy()

        return {
            os.path.join("predict", video_id): y_hat[index]
            for index, video_id in enumerate(video_list)
        }

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        # scheduler = ReduceLROnPlateau(
        #     optimizer,
        #     mode="max",
        #     factor=self.config["step_factor"],
        #     min_lr=1e-08,
        #     patience=self.config["patience"],
        #     cooldown=self.config["cooldown"],
        # )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "strict": True,
                "monitor": "validation_auc",
            }
        ]

    # def lr_scheduler_step(self, scheduler, metric):
    #     scheduler.step(metric)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log_dict({"train_loss": outputs["loss"]}, sync_dist=True, prog_bar=True)

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

    def on_test_epoch_end(self):
        self.log_dict(
            {"test_auc": self.test_auc.compute()}, sync_dist=True, prog_bar=True
        )
        self.test_auc.reset()
