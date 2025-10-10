import os

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from dataset import VideoDataModule, VideoDataset
from models import MMDet
from options.train_options import TrainOption
from utils.utils import get_logger, set_random_seed


if __name__ == "__main__":
    opt = TrainOption()
    args = opt.parse()

    config = args.__dict__
    logger = get_logger(__name__, config)
    logger.info(config)
    set_random_seed(config["seed"])

    os.makedirs(args.cache_dir, exist_ok=True)
    cache_file_path = os.path.join(args.cache_dir, args.cache_file_name)

    video_dataset = VideoDataset(cache_file_path=cache_file_path)
    video_datamodule = VideoDataModule(
        video_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = MMDet(config)

    trainer = L.Trainer(
        default_root_dir=args.ckpt_dir,
        strategy="ddp_find_unused_parameters_true",
        callbacks=[
            EarlyStopping(
                monitor="validation_loss", patience=5, verbose=False, mode="min"
            )
        ],
    )
    trainer.fit(model, video_datamodule)
