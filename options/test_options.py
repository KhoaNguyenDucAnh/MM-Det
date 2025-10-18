from .base_options import BaseOption


class TestOption(BaseOption):
    def __init__(self):
        super().__init__()
        # self.parser.add_argument(
        #     "-c",
        #     "--classes",
        #     type=str,
        #     nargs="+",
        #     default=["videocrafter1"],
        #     help="the forgery dataset for testing. Real datasets are from 3 datasets (VC1, Zscope, OSora) during evaluation.",
        # )
        self.parser.add_argument("--ckpt-path", type=str, help="checkpoint path")
        self.parser.add_argument(
            "--batch-size", type=int, default=1, help="batch size for testing"
        )
        # self.parser.add_argument("--mode", type=str, default="test", help="mode")
        # self.parser.add_argument(
        #     "--sample-size",
        #     type=int,
        #     help="the number of randomly sampled data for each dataloader",
        # )
        self.parser.add_argument(
            "--num-workers",
            type=int,
            default=4,
            help="number for dataloader workers per gpu",
        )
