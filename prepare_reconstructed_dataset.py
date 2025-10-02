import argparse
import json
import os

import cv2
import lightning as L
import numpy as np
import torch
import torch.utils.data as data
from einops import rearrange
from torch.nn import functional as F
from torchvision import transforms

from models import VectorQuantizedVAE
from utils.utils import CustomWriter
from dataset import ZarrDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="outputs/",
        help="output path",
    )
    parser.add_argument(
        "-fn",
        "--file-name",
        type=str,
        default="output.zarr",
        help="output file name",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="weights/vqvae/model.pt",
        help="checkpoint path for vqvae",
    )
    return parser.parse_args()



class ReconstructDataModule(L.LightningDataModule):

    def __init__(self, input_file, batch_size, num_workers):
        super().__init__()
        self.input_file = input_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = ZarrDataset(self.input_file, "original")

    def predict_dataloader(self):
        return data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: batch,
        )


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
        try:
            assert len(mean) == len(std)
            assert len(mean) == img_t.shape[1]
        except:
            print(
                f"Unmatched channels between image tensors and normalization mean and std. Got {img_t.shape[1]}, {len(mean)}, {len(std)}."
            )
        img_denorm = torch.empty_like(img_t)
        for t in range(img_t.shape[1]):
            img_denorm[:, t, :, :] = (img_t[:, t, :, :].clone() * std[t]) + mean[t]
        return img_denorm

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
                frames_batch = frames_batch.to(next(self.model.parameters()).device)
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


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)
    zarr_file_path = os.path.join(args.output, args.file_name)
    prediction_writer = CustomWriter(output_file=zarr_file_path)

    reconstruct_datamodule = ReconstructDataModule(
        input_file=zarr_file_path,
        batch_size=6,
        num_workers=6,
    )
    model = VectorQuantizedVAEWrapper(ckpt=args.ckpt, batch_size=64)
    trainer = L.Trainer(callbacks=[prediction_writer])
    trainer.predict(model, reconstruct_datamodule, return_predictions=False)
