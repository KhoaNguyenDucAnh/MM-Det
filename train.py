import os

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from dataset import VideoDataModule, VideoDataset
from models import MMDet
from options.train_options import TrainOption
from utils.utils import get_logger, set_random_seed


def parse_args(opt):
    parser = opt.parser
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="outputs/",
        help="output path",
    )
    parser.add_argument(
        "-fn",
        "--file-name",
        type=str,
        default="output.zarr",
        help="output file name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opt = TrainOption()
    args = parse_args(opt)

    config = args.__dict__
    logger = get_logger(__name__, config)
    logger.info(config)
    set_random_seed(config["seed"])

    zarr_file = os.path.join(args.output, args.file_name)

    video_dataset = VideoDataset(input_file=zarr_file)
    video_datamodule = VideoDataModule(
        video_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = MMDet(config)

    trainer = L.Trainer(
        default_root_dir=args.ckpt_dir,
        strategy="ddp",
        callbacks=[
            EarlyStopping(
                monitor="validation_loss", patience=5, verbose=False, mode="min"
            )
        ],
    )
    trainer.fit(model, video_datamodule)
