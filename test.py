import os
from copy import deepcopy

import lightning as L
import torch
from tqdm import tqdm

from dataset import VideoDataModule, VideoDataset
from models import MMDet
from options.test_options import TestOption
from utils.utils import CustomWriter, set_random_seed


def main(args):
    # logger = get_logger(__name__, args)
    # logger.info(args)
    set_random_seed(args["seed"])

    os.makedirs(args["cache_dir"], exist_ok=True)
    cache_file_path = os.path.join(args["cache_dir"], args["cache_file_name"])
    prediction_writer = CustomWriter(output_file=cache_file_path)

    video_dataset = VideoDataset(
        cache_file_path=cache_file_path, sample_method="entire"
    )
    video_datamodule = VideoDataModule(
        video_dataset,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        mode="test",
    )

    model = MMDet.load_from_checkpoint(args["ckpt_path"], config=args)
    model.eval()

    trainer = L.Trainer(
        strategy="ddp_find_unused_parameters_true", callbacks=[prediction_writer]
    )
    trainer.test(model, video_datamodule)


if __name__ == "__main__":
    opt = TestOption()
    args = opt.parse().__dict__

    main(args)
