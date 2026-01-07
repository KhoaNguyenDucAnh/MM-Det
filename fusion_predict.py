import os

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from models.fusion import *
from options.test_options import TestOption
from utils.utils import CustomWriter, set_random_seed


def main(args):
    set_random_seed(args["seed"])

    os.makedirs(args["cache_dir"], exist_ok=True)
    cache_file_path = os.path.join(args["cache_dir"], args["cache_file_name"])

    prediction_writer = CustomWriter(
        output_file="/scratch/gautschi/nguy1053/cache/av1m.zarr"
    )

    fusion_dataset = FusionDataset(
        visual_cache_file_path="/scratch/gautschi/nguy1053/cache/av1m.zarr",
        audio_cache_file_path="/scratch/gautschi/nguy1053/cache/audio.zarr",
        visual_logits="onecycle",
        exclude_groups_name=[args["predict_path"]],
    )
    fusion_datamodule = FusionDataModule(
        fusion_dataset,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        mode="predict",
    )

    model = Fusion.load_from_checkpoint(args["ckpt_path"], config=args)

    trainer = L.Trainer(
        strategy="ddp_find_unused_parameters_true", callbacks=[prediction_writer]
    )

    trainer.predict(model, fusion_datamodule)


if __name__ == "__main__":
    opt = TestOption()
    args = opt.parse().__dict__

    main(args)
