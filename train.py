import os

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

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
        split=[0.95, 0.05],
    )

    model = MMDet(args)

    model_checkpoint = ModelCheckpoint(
        monitor="validation_auc",
        mode="min",
        dirpath=args["ckpt_dir"],
        save_top_k=3,
        every_n_train_steps=1000,
        filename="MM-Det-{epoch:02d}-{validation_auc:.2f}",
    )
    early_stopping = EarlyStopping(
        monitor="validation_loss", mode="min", patience=5, verbose=False
    )

    trainer = L.Trainer(
        strategy="ddp_find_unused_parameters_true",
        callbacks=[model_checkpoint, early_stopping],
        val_check_interval=1000,
    )

    trainer.fit(model, video_datamodule)


if __name__ == "__main__":
    opt = TrainOption()
    args = opt.parse().__dict__

    main(args)
