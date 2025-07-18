import argparse
import os

import lightning as L
import numpy as np
import torch.utils.data as data

from dataset import ZarrDataset
from models import MMEncoder
from options.base_options import BaseOption
from utils.utils import CustomWriter


def parse_args(opt):
    parser = opt.parser
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="output path",
    )
    parser.add_argument(
        "--output-fn",
        type=str,
        default="output.zarr",
        help="output file name",
    )
    return parser.parse_args()


class MMRepresentationDataModule(L.LightningDataModule):

    def __init__(self, input_file, batch_size, num_workers):
        super().__init__()
        self.input_file = input_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = ZarrDataset(self.input_file, "original")

    def predict_dataloader(self):
        return data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: batch,
        )


if __name__ == "__main__":
    opt = BaseOption()
    args = parse_args(opt)

    os.makedirs(args.output_dir, exist_ok=True)
    zarr_file = os.path.join(args.output_dir, args.output_fn)
    prediction_writer = CustomWriter(output_file=zarr_file)

    mm_representation_datamodule = MMRepresentationDataModule(
        input_file=zarr_file,
        batch_size=6,
        num_workers=6,
    )
    model = MMEncoder(args.__dict__)
    trainer = L.Trainer(callbacks=[prediction_writer])
    trainer.predict(model, mm_representation_datamodule, return_predictions=False)
