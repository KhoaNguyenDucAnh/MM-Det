import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil

import cv2
import lightning as L
import numpy as np
import torch
import zarr
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .process import (
    get_image_transformation_from_cfg,
    get_video_transformation_from_cfg,
)
from .utils import get_default_transformation_cfg

"""
1: fake
0: real
"""


def filter_already_processed(cache_file_path, metadata):
    if os.path.exists(cache_file_path):
        zarr_file = zarr.open_group(cache_file_path, mode="r")
        already_processed_list = set(zarr_file["id"])
        metadata = [
            video_info
            for video_info in metadata
            if video_info[0] not in already_processed_list
        ]
    return metadata


class AV1MDataModule(L.LightningDataModule):

    def __init__(
        self, metadata_file, data_root, batch_size, num_workers, cache_file_path
    ):
        super().__init__()
        self.metadata_file = metadata_file
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_file_path = cache_file_path

    def setup(self, stage):
        cache_file = os.path.join(self.data_root, "metadata_cache.json")

        if os.path.exists(cache_file):
            with open(cache_file, "r") as file:
                self.metadata = json.load(file)
            self.metadata = filter_already_processed(
                self.cache_file_path, self.metadata
            )
            random.shuffle(self.metadata)
            return

        with open(os.path.join(self.data_root, self.metadata_file), "r") as file:
            temp_metadata = json.load(file)

        self.metadata = []
        for video_id, video_info in enumerate(
            tqdm(temp_metadata, desc="Preprocessing metadata")
        ):
            video_path = os.path.join(self.data_root, video_info["file"])
            if not os.path.exists(video_path):
                continue

            vc = cv2.VideoCapture(video_path)
            frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
            vc.release()

            if frame_count <= 0:
                continue

            label = [0] * frame_count

            for fake_segment in video_info["fake_segments"]:
                start = int(fake_segment[0] * 25)
                end = min(int(fake_segment[1] * 25) + 1, frame_count)
                for index in range(start, end):
                    label[index] = 1

            self.metadata.append([str(video_id), video_path, label])

        with open(cache_file, "w") as file:
            json.dump(self.metadata, file)

        self.metadata = filter_already_processed(self.cache_file_path, self.metadata)
        random.shuffle(self.metadata)

    def predict_dataloader(self):
        return DataLoader(
            self.metadata,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: batch,
        )


class GenVidBenchDataModule(L.LightningDataModule):

    def __init__(self, data_root, batch_size, num_workers, cache_file_path):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_file_path = cache_file_path

    def setup(self, stage):
        cache_file = os.path.join(self.data_root, "metadata_cache.json")

        if os.path.exists(cache_file):
            with open(cache_file, "r") as file:
                self.metadata = json.load(file)
            self.metadata = filter_already_processed(
                self.cache_file_path, self.metadata
            )
            return

        temp_metadata = []
        with open(os.path.join(self.data_root, "Pair1_labels.txt"), "r") as file:
            temp_metadata += [line.strip().rsplit(" ", 1) for line in file]
        with open(os.path.join(self.data_root, "Pair2_labels.txt"), "r") as file:
            temp_metadata += [line.strip().rsplit(" ", 1) for line in file]

        self.metadata = []
        for video_id, video_info in enumerate(temp_metadata):
            video_path = os.path.join(self.data_root, video_info[0])
            if not os.path.exists(video_path):
                continue

            vc = cv2.VideoCapture(video_path)
            frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
            vc.release()

            if frame_count <= 0:
                continue

            if video_info[1] == "0":
                label = [0] * frame_count
            else:
                label = [1] * frame_count

            self.metadata.append([str(video_id), video_path, label])

        with open(cache_file, "w") as file:
            json.dump(self.metadata, file)

        self.metadata = filter_already_processed(self.cache_file_path, self.metadata)

    def predict_dataloader(self):
        return DataLoader(
            self.metadata,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: batch,
        )


def validate_video(video, zarr_file, interval):
    """Validate a single video entry in parallel."""
    try:
        path = zarr_file["id"][video][0]
        original = zarr_file["original"][video]
        reconstruct = zarr_file["reconstruct"][video]
        visual = zarr_file["visual"][video]
        textual = zarr_file["textual"][video]
        label = zarr_file["label"][video]

        if not all([reconstruct, visual, textual, label]):
            return None

        video_length = original.shape[0]
        if video_length < 10:
            return None
        if "real_video_fake_audio" in path:
            return None
        if (
            video_length != reconstruct.shape[0]
            or video_length != label.shape[0]
            or ceil(video_length / interval) != visual.shape[0]
            or ceil(video_length / interval) != textual.shape[0]
        ):
            return None
        return (video, video_length)
    except Exception:
        return None


class VideoDataset(Dataset):
    def __init__(
        self,
        cache_file_path,
        sample_size=10,
        sample_method="continuous",
        transform_cfg=get_default_transformation_cfg(),
        repeat_sample_prob=0.0,
        interval=200,
        exclude_groups_name=None,
        cache_result_path=None,
        num_workers=8,
    ):
        super().__init__()

        zarr_file = zarr.open_group(cache_file_path, mode="r")
        self.original = zarr_file["original"]
        self.reconstruct = zarr_file["reconstruct"]
        self.visual = zarr_file["visual"]
        self.textual = zarr_file["textual"]
        self.label = zarr_file["label"]

        self.validation_sample_index = (
            {}
        )  # Use stored index for deterministic validation

        self.sample_size = sample_size
        if sample_method not in ["continuous", "entire"]:
            raise ValueError(
                f'Sample method should be either "continuous" or "entire", but not {sample_method}'
            )
        self.sample_method = sample_method
        self.transform = get_video_transformation_from_cfg(transform_cfg)
        self.repeat_sample_prob = repeat_sample_prob
        self.interval = interval
        self.exclude_groups_name = exclude_groups_name
        self.mode = "train"

        self.exclude = set()
        groups = set(zarr_file)
        if self.exclude_groups_name != None:
            for group_name in self.exclude_groups_name:
                if group_name in groups:
                    self.exclude |= set(zarr_file[group_name])

        # Caching setup
        if cache_result_path is None:
            cache_result_path = (
                os.path.splitext(cache_file_path)[0] + "_videos_cache.json"
            )
        self.cache_result_path = cache_result_path

        if os.path.exists(self.cache_result_path):
            print(f"Loading cached video list from {self.cache_result_path}")
            with open(self.cache_result_path, "r") as file:
                self.videos = json.load(file)
        else:
            print("Building video list in parallel...")
            all_videos = list(self.original.keys())

            valid_videos = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(validate_video, video, zarr_file, interval): video
                    for video in all_videos
                    if video not in self.exclude
                }

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Validating"
                ):
                    result = future.result()
                    if result:
                        valid_videos.append(result)

            self.videos = valid_videos

            # Save results to cache
            with open(self.cache_result_path, "w") as file:
                json.dump(self.videos, file)
            print(f"Cached {len(self.videos)} valid videos to {self.cache_result_path}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video, video_length = self.videos[index]

        if self.sample_size > video_length:
            raise ValueError(
                f"The sample size {self.sample_size} is longer than the total frame number {video_length} of video {video}"
            )

        sample_index = []
        if (
            video in self.validation_sample_index
        ):  # Use stored index for deterministic validation
            sample_index = self.validation_sample_index[video]
        else:
            # if self.sample_method == "random":
            #     sample_index = random.sample(range(video_length), self.sample_size)
            if self.sample_method == "continuous":
                sample_start = random.randint(0, video_length - self.sample_size)
                sample_index = list(
                    range(sample_start, sample_start + self.sample_size)
                )
            elif self.sample_method == "entire":
                sample_index = list(range(video_length))
            else:
                raise ValueError(
                    f'Sample method should be either "continuous" or "entire", but not {self.sample_method}'
                )

            if self.mode == "validation":
                self.validation_sample_index[video] = sample_index

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

        if self.sample_method == "continuous":
            visual_textual_feature_index = sample_index[0] // self.interval
            visual_feature = self.visual[video][
                visual_textual_feature_index : visual_textual_feature_index + 1
            ]
            textual_feature = self.textual[video][
                visual_textual_feature_index : visual_textual_feature_index + 1
            ]
        elif self.sample_method == "entire":
            visual_feature = self.visual[video][:]
            textual_feature = self.textual[video][:]
        else:
            raise ValueError(
                f'Sample method should be either "continuous" or "entire", but not {self.sample_method}'
            )

        if self.mode == "predict":
            return (
                video,
                torch.stack(original_frames, dim=0),
                torch.stack(reconstructed_frames, dim=0),
                torch.tensor(visual_feature),
                torch.tensor(textual_feature),
            )
        if self.mode == "test":
            label = self.label[video][sample_index]
            return (
                video,
                torch.stack(original_frames, dim=0),
                torch.stack(reconstructed_frames, dim=0),
                torch.tensor(visual_feature),
                torch.tensor(textual_feature),
                torch.tensor(label, dtype=torch.long),
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

    def __init__(self, dataset, batch_size, num_workers, mode, split=[0.8, 0.2]):
        super().__init__()
        self.dataset_obj = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.split = split

        if mode not in ["train", "test", "predict"]:
            raise ValueError(
                f'Mode should be either "train", "test", or "predict", but not {mode}'
            )

        if mode == "train":
            self.train, self.validation = random_split(dataset, split)
            self.train_indices = self.train.indices
            self.val_indices = self.validation.indices

            self.train.dataset.mode = "train"
            self.validation.dataset.mode = "validation"

        else:
            self.dataset = dataset
            self.dataset.mode = mode
            self.train_indices = None
            self.val_indices = None

    def state_dict(self):
        return {
            "mode": self.mode,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "split": self.split,
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
        }

    def load_state_dict(self, state_dict):
        self.mode = state_dict["mode"]
        self.batch_size = state_dict["batch_size"]
        self.num_workers = state_dict["num_workers"]
        self.split = state_dict["split"]
        self.train_indices = state_dict["train_indices"]
        self.val_indices = state_dict["val_indices"]

        if self.mode == "train":
            self.train = torch.utils.data.Subset(self.dataset_obj, self.train_indices)
            self.validation = torch.utils.data.Subset(
                self.dataset_obj, self.val_indices
            )
            self.train.dataset.mode = "train"
            self.validation.dataset.mode = "validation"
        else:
            self.dataset = self.dataset_obj
            self.dataset.mode = self.mode

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
            torch.stack(visual_feature_list),
            torch.stack(textual_feature_list),
            torch.stack(label_list),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
        )
