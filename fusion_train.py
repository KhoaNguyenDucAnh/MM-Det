import os

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from models.fusion import *
from options.train_options import TrainOption
from utils.utils import set_random_seed


def main(args):
    set_random_seed(args["seed"])

    os.makedirs(args["cache_dir"], exist_ok=True)
    cache_file_path = os.path.join(args["cache_dir"], args["cache_file_name"])

    fusion_dataset = FusionDataset(
        visual_cache_file_path="/scratch/gautschi/nguy1053/cache/av1m.zarr",
        audio_cache_file_path="/scratch/gautschi/nguy1053/cache/audio.zarr",
        visual_logits=args["visual_logits"],
    )
    fusion_datamodule = FusionDataModule(
        fusion_dataset,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        mode="train",
        split=[0.8, 0.2],
    )

    model = Fusion(args)

    model_checkpoint = ModelCheckpoint(
        monitor="validation_auc",
        mode="max",
        dirpath=args["ckpt_dir"],
        save_top_k=-1,
        filename="Fusion-"
        + args["visual_logits"]
        + "-{epoch:02d}-{lr:.6f}-{validation_auc:.6f}",
    )

    trainer = L.Trainer(
        strategy="ddp_find_unused_parameters_true",
        callbacks=[model_checkpoint],
        accumulate_grad_batches=16,
        max_epochs=args["max_epochs"],
    )

    previous_checkpoint_path = ""  # "/scratch/gautschi/nguy1053/checkpoints/"
    if previous_checkpoint_path != "":
        checkpoint = torch.load(previous_checkpoint_path)
        fusion_datamodule.load_state_dict(checkpoint["FusionDataModule"])
        trainer.fit(model, fusion_datamodule, ckpt_path=previous_checkpoint_path)
    else:
        trainer.fit(model, fusion_datamodule)


if __name__ == "__main__":
    opt = TrainOption()
    args = opt.parse().__dict__

    main(args)
