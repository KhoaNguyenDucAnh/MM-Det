import json
import logging
import os
import random
import sys
from copy import deepcopy

import numpy as np
import torch
import zarr
from lightning.pytorch.callbacks import BasePredictionWriter


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_file):
        super().__init__("epoch")
        self.output_file = output_file

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        if prediction == None:
            return
        file = zarr.open_group(self.output_file, mode="a")
        for key, value in prediction.items():
            file.array(
                name=key,
                data=value,
                shape=value.shape,
                dtype=value.dtype,
                overwrite=True,
            )
