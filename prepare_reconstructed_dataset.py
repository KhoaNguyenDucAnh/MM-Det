import json
import os

import cv2
import lightning as L
import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F
from torchvision import transforms

from dataset import ReconstructDataModule
from models import VectorQuantizedVAE
from options.base_options import BaseOption
from utils.utils import CustomWriter


class VectorQuantizedVAEWrapper(L.LightningModule):

    def __init__(self, ckpt, batch_size):
        super().__init__()
        self.recons_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        self.model = VectorQuantizedVAE(3, 256, 512)
        self.model.load_state_dict(torch.load(ckpt), strict=True)

        self.batch_size = batch_size

    def forward(self, X):
        return self.model(X)

    def denormalize_batch_t(self, img_t, mean, std):
        mean = torch.as_tensor(mean, dtype=img_t.dtype, device=img_t.device)[
            None, :, None, None
        ]
        std = torch.as_tensor(std, dtype=img_t.dtype, device=img_t.device)[
            None, :, None, None
        ]

        return img_t * std + mean

    def postprocess_reconstructed_frames(
        self, frames_batch, reconstructed_frames_batch
    ):
        if reconstructed_frames_batch.shape != frames_batch.shape:
            reconstructed_frames_batch = F.interpolate(
                reconstructed_frames_batch,
                (frames_batch.shape[-2], frames_batch.shape[-1]),
                mode="nearest",
            )
        reconstructed_frames_batch = self.denormalize_batch_t(
            reconstructed_frames_batch,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
        reconstructed_frames_batch = rearrange(
            reconstructed_frames_batch.cpu().numpy(),
            "b c h w -> b h w c",
        )
        reconstructed_frames_batch = np.uint8(reconstructed_frames_batch * 255.0)
        return reconstructed_frames_batch

    def predict_step(self, batch):
        reconstructed_batch = {}
        for video_id, extracted_frames in batch:
            reconstructed_frames = np.empty_like(extracted_frames, dtype=np.uint8)
            for i in range(0, extracted_frames.shape[0], self.batch_size):
                frames_batch = torch.stack(
                    [
                        self.recons_transform(frame)
                        for frame in extracted_frames[i : i + self.batch_size]
                    ]
                )
                frames_batch = frames_batch.to(self.device)
                reconstructed_frames_batch, _, _ = self.forward(frames_batch)

                reconstructed_frames_batch = self.postprocess_reconstructed_frames(
                    frames_batch, reconstructed_frames_batch
                )

                reconstructed_frames[i : i + self.batch_size] = (
                    reconstructed_frames_batch
                )
            reconstructed_batch[os.path.join("reconstruct", video_id)] = (
                reconstructed_frames
            )
        return reconstructed_batch


def main(args):
    os.makedirs(args["cache_dir"], exist_ok=True)
    cache_file_path = os.path.join(args["cache_dir"], args["cache_file_name"])
    prediction_writer = CustomWriter(output_file=cache_file_path)

    reconstruct_datamodule = ReconstructDataModule(
        cache_file_path=cache_file_path,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
    )

    model = VectorQuantizedVAEWrapper(ckpt=args["vqvae_ckpt"], batch_size=64)

    trainer = L.Trainer(callbacks=[prediction_writer])
    trainer.predict(model, reconstruct_datamodule, return_predictions=False)


if __name__ == "__main__":
    opt = BaseOption()
    args = opt.parse().__dict__

    main(args)
