import json
import os

import cv2
import lightning as L
import numpy as np
import zarr
from datasets import load_dataset
from lightning.pytorch.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader

from dataset.video_dataset import *
from options.base_options import BaseOption
from utils.utils import CustomWriter


class VideoFrameExtractor(L.LightningModule):
    def __init__(self):
        super().__init__()

    def predict_step(self, batch):
        extracted_batch = {}
        for video_id, video_path, label in batch:
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

            extracted_batch[os.path.join("id", video_id)] = np.array([video_path])
            extracted_batch[os.path.join("original", video_id)] = np.array(
                extracted_frames
            )
            extracted_batch[os.path.join("label", video_id)] = label
        return extracted_batch


class SAFEVideoChallengeFrameExtractor(L.LightningModule):
    def __init__(self):
        super().__init__()

    def predict_step(self, batch):
        extracted_batch = {}
        for video in batch:
            video_id = video["id"]

            byte = io.BytesIO(video["video"]["bytes"])
            byte.seek(0)
            container = av.open(byte)

            extracted_frames = []
            for frame in container.decode(video=0):
                extracted_frames.append(frame.to_ndarray(format="rgb24"))

            extracted_frames = []
            while True:
                ret, frame = vc.read()
                if not ret:
                    break
                extracted_frames.append(frame)
            vc.release()

            label = np.asarray([0] * len(extracted_frames))

            extracted_batch[os.path.join("id", video_id)] = np.array([video_path])
            extracted_batch[os.path.join("original", video_id)] = np.array(
                extracted_frames
            )
            extracted_batch[os.path.join("label", video_id)] = label
        return extracted_batch


def main(args):
    os.makedirs(args["cache_dir"], exist_ok=True)
    cache_file_path = os.path.join(args["cache_dir"], args["cache_file_name"])
    prediction_writer = CustomWriter(output_file=cache_file_path)

    av1m_datamodule = AV1MDataModule(
        metadata_file="train_metadata.json",
        data_root=args["data_root"],
        batch_size=6,
        num_workers=6,
        cache_file_path=cache_file_path,
    )

    # genvidbench_datamodule = GenVidBenchDataModule(
    #     data_root=args["data_root"],
    #     batch_size=4,
    #     num_workers=16,
    #     cache_file_path=cache_file_path,
    # )

    # safevideochallenge_datamodule = SAFEVideoChallengeDataModule(
    #     data_root=args["data_root"],
    #     batch_size=4,
    #     num_workers=16,
    #     cache_file_path=cache_file_path,
    # )

    video_frame_extractor = VideoFrameExtractor()

    # video_frame_extractor = SAFEVideoChallengeFrameExtractor()

    trainer = L.Trainer(
        accelerator="cpu",
        callbacks=[prediction_writer],
    )

    trainer.predict(
        video_frame_extractor, av1m_datamodule, return_predictions=False
    )


if __name__ == "__main__":
    opt = BaseOption()
    args = opt.parse().__dict__

    main(args)
