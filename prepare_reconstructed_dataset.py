import argparse
import json
import os

import cv2
import lightning as L
import numpy as np
import torch
import torch.utils.data as data
import zarr
from einops import rearrange
from lightning.pytorch.callbacks import BasePredictionWriter
from torch.nn import functional as F
from torchvision import transforms

from models import VectorQuantizedVAE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-root", type=str, default="data", help="data root for videos"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="outputs/",
        help="output path",
    )
    parser.add_argument(
        "--ext",
        type=str,
        nargs="+",
        default=["mp4", "avi", "mpeg", "wmv", "mov", "flv"],
        help="target extensions",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="weights/vqvae/model.pt",
        help="checkpoint path for vqvae",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device for reconstruction model"
    )
    return parser.parse_args()


class AV1MDataModule(L.LightningDataModule):

    def __init__(self, metadata_file, data_root, batch_size, num_workers):
        super().__init__()
        self.metadata_file = metadata_file
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        with open(self.metadata_file, "r") as file:
            self.metadata = json.load(file)
        self.metadata = [
            os.path.join(self.data_root, video_info["file"])
            for video_info in self.metadata
            if os.path.exists(os.path.join(self.data_root, video_info["file"]))
        ]

    def predict_dataloader(self):
        return data.DataLoader(
            self.metadata, batch_size=self.batch_size, num_workers=self.num_workers
        )


class ZarrLazyDataset(data.Dataset):
    def __init__(self, zarr_group):
        self.zarr_group = zarr_group
        self.keys = list(zarr_group.array_keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        array = self.zarr_group[key]
        return key, array[:]


class ReconstructDataModule(L.LightningDataModule):

    def __init__(self, input_file, batch_size, num_workers):
        super().__init__()
        self.input_file = input_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    # def prepare_data(self):
    #     file = zarr.open_group(self.input_file, mode="r")
    #     self.dataset = list(file["original"].arrays())

    def setup(self, stage=None):
        zarr_file = zarr.open_group(self.input_file, mode="r")
        self.dataset = ZarrLazyDataset(zarr_file["original"])

    def predict_dataloader(self):
        return data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: batch,
        )


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_file):
        super().__init__("batch")
        self.output_file = output_file

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        if prediction == None:
            return
        file = zarr.open_group(self.output_file, mode="a")
        for key, value in prediction.items():
            array = file.create_array(
                name=key,
                shape=value.shape,
                dtype=value.dtype,
                overwrite=True,
            )
            array[...] = value


class VideoFrameExtractor(L.LightningModule):
    def __init__(self):
        super().__init__()

    def predict_step(self, batch):
        extracted_batch = {}
        for video_path in batch:
            vc = cv2.VideoCapture(video_path)

            if not vc.isOpened():
                continue

            extracted_frames = []
            while True:
                ret, frame = vc.read()
                if not ret:
                    break
                extracted_frames.append(frame)
            vc.release()

            extracted_batch[os.path.join("original", video_path.replace("/", "_"))] = (
                np.array(extracted_frames)
            )
        return extracted_batch


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

    def forward(self, X):
        return self.model(X)

    def predict_step(self, batch):
        reconstructed_batch = {}
        for video_path, extracted_frames in batch:
            reconstructed_frames = np.empty_like(extracted_frames, dtype=np.uint8)
            for i in range(0, extracted_frames.shape[0], self.batch_size):
                frames_batch = torch.stack(
                    [
                        self.recons_transform(frame)
                        for frame in extracted_frames[i : i + self.batch_size]
                    ]
                )
                frames_batch = frames_batch.to(next(self.model.parameters()).device)
                reconstructed_frames_batch, _, _ = model(frames_batch)

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
                reconstructed_frames_batch = np.uint8(
                    reconstructed_frames_batch * 255.0
                )
                reconstructed_frames[i : i + self.batch_size] = (
                    reconstructed_frames_batch
                )
            reconstructed_batch[os.path.join("reconstruct", video_path)] = (
                reconstructed_frames
            )
        return reconstructed_batch


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)
    zarr_file = os.path.join(args.output, "output.zarr")
    prediction_writer = CustomWriter(output_file=zarr_file)

    if False:
        av1m_datamodule = AV1MDataModule(
            metadata_file="data/train_metadata.json",
            data_root="data/train/train",
            batch_size=6,
            num_workers=6,
        )
        video_frame_extractor = VideoFrameExtractor()
        trainer = L.Trainer(
            accelerator="cpu",
            devices=6,
            callbacks=[prediction_writer],
        )
        trainer.predict(
            video_frame_extractor, av1m_datamodule, return_predictions=False
        )
    if True:
        reconstruct_datamodule = ReconstructDataModule(
            input_file=zarr_file,
            batch_size=6,
            num_workers=6,
        )
        model = VectorQuantizedVAEWrapper(ckpt=args.ckpt, batch_size=64)
        trainer = L.Trainer(callbacks=[prediction_writer])
        trainer.predict(model, reconstruct_datamodule, return_predictions=False)
