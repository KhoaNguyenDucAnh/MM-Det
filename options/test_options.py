from .base_options import BaseOption


class TestOption(BaseOption):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("--ckpt-path", type=str, help="checkpoint path")
        self.parser.add_argument("--predict-path", type=str, help="predict path")
