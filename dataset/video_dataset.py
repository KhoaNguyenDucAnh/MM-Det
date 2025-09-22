import os
import random

import cv2
import lightning as L
import numpy as np
import torch
import zarr
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from .process import (
    get_image_transformation_from_cfg,
    get_video_transformation_from_cfg,
)
from .utils import get_default_transformation_cfg

"""
1: fake
0: real
"""


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

        self.metadata = []
        for video_info in self.metadata:
            video_path = os.path.join(self.data_root, video_info["file"])
            if os.path.exists(video_path):
                vc = cv2.VideoCapture(video_path)
                label = [0 for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))]
                for fake_segment in video_info["fake_segments"]:
                    for index in range(
                        int(fake_segment[0] * 25), int(fake_segment[1] * 25) + 1
                    ):
                        label[index] = 1
                self.metadata.append([video_path, label])

    def predict_dataloader(self):
        return DataLoader(
            self.metadata,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: batch,
        )


class GenVidBenchDataModule(L.LightningDataModule):

    def __init__(self, data_root, batch_size, num_workers):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        self.metadata = []
        with open(os.path.join(self.data_root, "Pair1_labels.txt"), "r") as file:
            self.metadata += [line.strip().rsplit(" ", 1) for line in file]
        with open(os.path.join(self.data_root, "Pair2_labels.txt"), "r") as file:
            self.metadata += [line.strip().rsplit(" ", 1) for line in file]

        for video_info in self.metadata:
            video_path = os.path.join(self.data_root, video_info[0])
            if os.path.exists(video_path):
                vc = cv2.VideoCapture(video_path)

                if video_info[1] == "0":
                    label = [0 for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))]
                else:
                    label = [1 for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))]

                self.metadata.append([video_path, label])

    def predict_dataloader(self):
        return DataLoader(
            self.metadata,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: batch,
        )


class VideoDataset(Dataset):
    def __init__(
        self,
        input_file,
        sample_size=10,
        sample_method="continuous",
        transform_cfg=get_default_transformation_cfg(),
        repeat_sample_prob=0.0,
        interval=200,
    ):
        super().__init__()

        zarr_file = zarr.open_group(input_file, mode="r")
        self.original = zarr_file["original"]
        self.reconstruct = zarr_file["reconstruct"]
        self.visual = zarr_file["visual"]
        self.textual = zarr_file["textual"]
        self.label = zarr_file["label"]

        self.sample_size = sample_size
        self.sample_method = sample_method
        if sample_method not in ["random", "continuous", "entire"]:
            raise ValueError(
                f'Sample method should be either "random" or "continuous", but not {self.sample_method}'
            )
        self.transform = get_video_transformation_from_cfg(transform_cfg)
        self.repeat_sample_prob = repeat_sample_prob
        self.interval = interval
        self.is_test = False
        self.test_sample_index = {}
        self.is_predict = False

        self.setup()

    def setup(self):
        self.videos = []
        for video in self.original:
            video_length = self.original[video].shape[0]
            if video not in self.reconstruct:
                # Video is not reconstructed
                continue
            if video_length != self.reconstruct[video].shape[0]:
                # Number of frames does not match
                continue
            self.videos.append((video, video_length))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video, video_length = self.videos[index]

        if self.sample_size > video_length:
            raise ValueError(
                f"The sample size {self.sample_size} is longer than the total frame number {video_length} of video {video}"
            )

        sample_index = []
        if video in self.test_sample_index:
            sample_index = self.test_sample_index[video]
        elif self.sample_method == "random":
            sample_index = random.sample(range(video_length), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(0, video_length - self.sample_size)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(video_length))
        else:
            raise ValueError(
                f'Sample method should be either "random" or "continuous", but not {self.sample_method}'
            )

        if self.is_test:
            self.test_sample_index[video] = sample_index

        original_frames, reconstructed_frames = [], []
        for frame_index in sample_index:
            # Original frame
            original_frame = Image.fromarray(
                self.original[video][frame_index].astype("uint8"), "RGB"
            )
            transformed_frame = self.transform(original_frame)
            original_frames.append(transformed_frame)

            # Reconstructed frame
            reconstructed_frame = Image.fromarray(
                self.reconstruct[video][frame_index].astype("uint8"), "RGB"
            )
            transformed_reconstructed_frame = self.transform(reconstructed_frame)
            reconstructed_frames.append(transformed_reconstructed_frame)

        visual_feature = self.visual[video][sample_index[0] // self.interval]
        textual_feature = self.textual[video][sample_index[0] // self.interval]

        if self.is_predict:
            return (
                video,
                torch.stack(original_frames, dim=0),
                torch.stack(reconstructed_frames, dim=0),
                torch.tensor(visual_feature),
                torch.tensor(textual_feature),
            )
        else:
            label = np.max(self.label[video][sample_index])
            return (
                video,
                torch.stack(original_frames, dim=0),
                torch.stack(reconstructed_frames, dim=0),
                torch.tensor(visual_feature),
                torch.tensor(textual_feature),
                torch.tensor(label, dtype=torch.long),
            )


class VideoDataModule(L.LightningDataModule):

    def __init__(
        self, dataset, batch_size, num_workers, split=[0.8, 0.2], is_predict=False
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        if is_predict:
            self.dataset = dataset
        else:
            self.train, self.validation = random_split(dataset, split)

    def collate_fn(self, batch):
        (
            video_list,
            original_frames_list,
            reconstructed_frames_list,
            visual_feature_list,
            textual_feature_list,
            label_list,
        ) = list(zip(*batch))
        return (
            video_list,
            torch.stack(original_frames_list),
            torch.stack(reconstructed_frames_list),
            torch.cat(visual_feature_list),
            torch.cat(textual_feature_list),
            torch.stack(label_list),
        )

    def predict_collate_fn(self, batch):
        (
            video_list,
            original_frames_list,
            reconstructed_frames_list,
            visual_feature_list,
            textual_feature_list,
        ) = list(zip(*batch))
        return (
            video_list,
            torch.stack(original_frames_list),
            torch.stack(reconstructed_frames_list),
            torch.cat(visual_feature_list),
            torch.cat(textual_feature_list),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        self.validation.dataset.is_test = True
        return DataLoader(
            self.validation,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        self.dataset.is_predict = True
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.predict_collate_fn,
        )
