import argparse
import json
import os

import cv2
import lightning as L
import numpy as np
import zarr
from lightning.pytorch.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader

from dataset.video_dataset import *
from utils.utils import CustomWriter


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
    return parser.parse_args()


class VideoFrameExtractor(L.LightningModule):
    def __init__(self):
        super().__init__()

    def predict_step(self, batch):
        extracted_batch = {}
        for video_path, label in batch:
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

            label = np.asarray(label)

            extracted_batch[os.path.join("original", video_path.replace("/", "_"))] = (
                np.array(extracted_frames)
            )
            extracted_batch[os.path.join("label", video_path.replace("/", "_"))] = label
        return extracted_batch


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)
    zarr_file = os.path.join(args.output, args.file_name)
    prediction_writer = CustomWriter(output_file=zarr_file)

    # av1m_datamodule = AV1MDataModule(
    #     metadata_file="data/train_metadata.json",
    #     data_root="data/train/train",
    #     batch_size=6,
    #     num_workers=6,
    # )

    genvidbench_datamodule = GenVidBenchDataModule(
        data_root="/scratch/gilbreth/nguy1053/GenVidBench",
        batch_size=6,
        num_workers=6,
    )

    video_frame_extractor = VideoFrameExtractor()

    trainer = L.Trainer(
        accelerator="cpu",
        devices=6,
        callbacks=[prediction_writer],
    )

    trainer.predict(video_frame_extractor, genvidbench_datamodule, return_predictions=False)
