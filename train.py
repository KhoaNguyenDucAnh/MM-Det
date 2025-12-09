import os

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from dataset import VideoDataModule, VideoDataset
from models import MMDet
from options.train_options import TrainOption
from utils.utils import set_random_seed


def main(args):
    set_random_seed(args["seed"])

    os.makedirs(args["cache_dir"], exist_ok=True)
    cache_file_path = os.path.join(args["cache_dir"], args["cache_file_name"])

    video_dataset = VideoDataset(
        cache_file_path=cache_file_path, interval=args["interval"]
    )
    video_datamodule = VideoDataModule(
        video_dataset,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        mode="train",
        split=[0.8, 0.2],
    )

    model = MMDet(args)

    model_checkpoint = ModelCheckpoint(
        monitor="validation_auc",
        mode="max",
        dirpath=args["ckpt_dir"],
        save_top_k=-1,
        filename="MM-Det-{epoch:02d}-{validation_auc:.5f}",
    )

    trainer = L.Trainer(
        strategy="ddp_find_unused_parameters_true",
        callbacks=[model_checkpoint],
        accumulate_grad_batches=16,
        max_epochs=args["max_epochs"],
    )

    previous_checkpoint_path = ""
    if previous_checkpoint_path != "":
        checkpoint = torch.load(previous_checkpoint_path)
        video_datamodule.load_state_dict(checkpoint["VideoDataModule"])
        trainer.fit(model, video_datamodule, ckpt_path=previous_checkpoint_path)
    else:
        trainer.fit(model, video_datamodule)


if __name__ == "__main__":
    opt = TrainOption()
    args = opt.parse().__dict__

    main(args)
