import random

import numpy as np
import torch
import zarr
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import roc_auc_score


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class CustomWriter(Callback):
    def __init__(self, output_file):
        self.output_file = output_file

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        if outputs == None:
            return
        file = zarr.open_group(self.output_file, mode="a")
        for key, value in outputs.items():
            if key == "loss":
                continue
            file.array(
                name=key,
                data=value,
                shape=value.shape,
                dtype=value.dtype,
                overwrite=True,
            )

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        if outputs == None:
            return
        file = zarr.open_group(self.output_file, mode="a")
        for key, value in outputs.items():
            if key == "loss":
                continue
            file.array(
                name=key,
                data=value,
                shape=value.shape,
                dtype=value.dtype,
                overwrite=True,
            )
