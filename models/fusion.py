import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import lightning as L
import torch
import torch.nn as nn
import zarr
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import BinaryAUROC
from tqdm import tqdm


def validate_video(video, path, visual_logits, audio_logits):
    """Validate a single video entry in parallel."""
    try:
        prefix = 50  # len("/scratch/gautschi/nguy1053/AV-Deepfake1M-PlusPlus/")
        suffix = -4  # .mp4
        visual_logits = visual_logits[video]
        video_length = visual_logits.shape[0]

        audio_path = path[prefix:suffix].replace("/", "_")
        audio_16_logits = audio_logits[audio_path]["16"]
        audio_32_logits = audio_logits[audio_path]["32"]
        audio_64_logits = audio_logits[audio_path]["64"]

        if not all([visual_logits, audio_16_logits, audio_32_logits, audio_64_logits]):
            return None

        return (video, video_length, audio_path)
    except Exception:
        return None


class FusionDataset(Dataset):
    def __init__(
        self,
        visual_cache_file_path,
        audio_cache_file_path,
        visual_logits,
        exclude_groups_name=None,
        cache_result_path=None,
        num_workers=8,
    ):
        super().__init__()

        self.mode = "train"

        visual_zarr_file = zarr.open_group(visual_cache_file_path, mode="r")
        self.visual_logits = visual_zarr_file[visual_logits]
        self.label = visual_zarr_file["label"]

        self.audio_logits = zarr.open_group(audio_cache_file_path, mode="r")

        # Caching setup
        if cache_result_path is None:
            cache_result_path = (
                os.path.splitext(visual_cache_file_path)[0]
                + "_fusion_videos_cache.json"
            )
        self.cache_result_path = cache_result_path

        if os.path.exists(self.cache_result_path):
            print(f"Loading cached video list from {self.cache_result_path}")
            with open(self.cache_result_path, "r") as file:
                self.video_dict = json.load(file)
        else:
            print("Building video list in parallel...")
            self.video_dict = {}
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        validate_video,
                        video,
                        path[0],
                        self.visual_logits,
                        self.audio_logits,
                    ): video
                    for video, path in visual_zarr_file["id"].items()
                }

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Validating"
                ):
                    result = future.result()
                    if result:
                        video, video_length, audio_path = result
                        self.video_dict[video] = (video_length, audio_path)

            # Save results to cache
            with open(self.cache_result_path, "w") as file:
                json.dump(self.video_dict, file)
            print(
                f"Cached {len(self.video_dict)} valid videos to {self.cache_result_path}"
            )

        exclude = set()
        groups = set(visual_zarr_file)
        if exclude_groups_name != None:
            for group_name in exclude_groups_name:
                if group_name in groups:
                    exclude |= set(visual_zarr_file[group_name])

        self.video_id_list_predict = list(set(self.video_dict.keys()) - exclude)

        self.video_id_list_train = []
        for video in self.video_id_list_predict:
            self.video_id_list_train += [
                (video, frame, self.video_dict[video][1])
                for frame in range(self.video_dict[video][0])
            ]

    def __len__(self):
        if self.mode == "predict":
            return len(self.video_id_list_predict)
        else:
            return len(self.video_id_list_train)

    def __getitem__(self, index):
        if self.mode == "predict":
            video = self.video_id_list_predict[index]
            _, audio_path = self.video_dict[video]
            return (  # 25 FPS, 160ms 320ms 640ms
                video,
                self.visual_logits[video],
                self.audio_logits[audio_path]["16"][:, :2],
                self.audio_logits[audio_path]["32"][:, :2],
                self.audio_logits[audio_path]["64"][:, :2],
            )
            return
        else:
            video, frame, audio_path = self.video_id_list_train[index]
            return (  # 25 FPS, 160ms 320ms 640ms
                video,
                self.visual_logits[video][frame],
                self.audio_logits[audio_path]["16"][
                    min(frame // 4, self.audio_logits[audio_path]["16"].shape[0] - 1),
                    :2,
                ],
                self.audio_logits[audio_path]["32"][
                    min(frame // 8, self.audio_logits[audio_path]["32"].shape[0] - 1),
                    :2,
                ],
                self.audio_logits[audio_path]["64"][
                    min(frame // 16, self.audio_logits[audio_path]["64"].shape[0] - 1),
                    :2,
                ],
                self.label[video][frame],
            )


class FusionDataModule(L.LightningDataModule):

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

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=lambda x: x,
        )


class Fusion(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.predict_path = config["predict_path"]

        self.model = nn.Linear(8, 2)

        self.train_auc = BinaryAUROC()
        self.validation_auc = BinaryAUROC()

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch):
        (
            video,
            visual_logits,
            audio_16_logits,
            audio_32_logits,
            audio_64_logits,
            label,
        ) = batch

        logits = self.forward(
            torch.cat(
                [visual_logits, audio_16_logits, audio_32_logits, audio_64_logits],
                dim=1,
            )
        )
        loss = torch.nn.functional.cross_entropy(logits, label)

        y_hat = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
        self.train_auc.update(y_hat, label)

        return {"loss": loss}

    def validation_step(self, batch):
        (
            video,
            visual_logits,
            audio_16_logits,
            audio_32_logits,
            audio_64_logits,
            label,
        ) = batch

        logits = self.forward(
            torch.cat(
                [visual_logits, audio_16_logits, audio_32_logits, audio_64_logits],
                dim=1,
            )
        )
        loss = torch.nn.functional.cross_entropy(logits, label)

        y_hat = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
        self.validation_auc.update(y_hat, label)

        return {"loss": loss}

    def predict_step(self, batch):
        (
            video,
            visual_logits,
            audio_16_logits,
            audio_32_logits,
            audio_64_logits,
        ) = batch[0]

        len_audio_16_logits = audio_16_logits.shape[0]
        len_audio_32_logits = audio_32_logits.shape[0]
        len_audio_64_logits = audio_64_logits.shape[0]
        final_logits = []
        for i in range(visual_logits.shape[0]):
            logits = self.forward(
                torch.cat(
                    [
                        torch.tensor(visual_logits[i], device=self.device),
                        torch.tensor(
                            audio_16_logits[min(i // 4, len_audio_16_logits - 1)],
                            device=self.device,
                        ),
                        torch.tensor(
                            audio_32_logits[min(i // 8, len_audio_32_logits - 1)],
                            device=self.device,
                        ),
                        torch.tensor(
                            audio_64_logits[min(i // 16, len_audio_64_logits - 1)],
                            device=self.device,
                        ),
                    ]
                )
            )
            final_logits.append(logits)
        final_logits = torch.stack(final_logits)
        final_logits = final_logits.detach().cpu().numpy()

        return {os.path.join(self.predict_path, video): final_logits}

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
