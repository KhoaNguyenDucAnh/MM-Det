import os

import lightning as L
import numpy as np
import torch.utils.data as data

from dataset import MMRepresentationDataModule
from models import MMEncoder
from options.base_options import BaseOption
from utils.utils import CustomWriter

def main(args):
    os.makedirs(args.["cache_dir"], exist_ok=True)
    cache_file_path = os.path.join(args["cache_dir"], args["cache_file_name"])
    prediction_writer = CustomWriter(output_file=cache_file_path)

    mm_representation_datamodule = MMRepresentationDataModule(
        cache_file_path=cache_file_path,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
    )
    
    model = MMEncoder(args)
    
    trainer = L.Trainer(strategy="ddp", callbacks=[prediction_writer])
    trainer.predict(model, mm_representation_datamodule, return_predictions=False)

if __name__ == "__main__":
    opt = BaseOption()
    args = opt.parse().__dict__
    
    main(args)
