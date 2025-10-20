import random

import numpy as np
import torch
import zarr
from lightning.pytorch.callbacks import BasePredictionWriter, Callback
from sklearn.metrics import roc_auc_score


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


class AUCCalculator(Callback):
    def __init__(self, output_file):
        super().__init__()
        self.output_file = output_file

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        y_hat = torch.nn.functional.softmax(outputs["logits"], dim=-1)[:, 1]
        y_hat = y_hat.detach().cpu().numpy()
        y_true = outputs["label"].detach().cpu().numpy()
        auc = roc_auc_score(y_true, y_hat)
        pl_module.log_dict({"train_auc": auc}, sync_dist=True, prog_bar=True)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        y_hat = torch.nn.functional.softmax(outputs["logits"], dim=-1)[:, 1]
        y_hat = y_hat.detach().cpu().numpy()
        y_true = outputs["label"].detach().cpu().numpy()
        auc = roc_auc_score(y_true, y_hat)
        pl_module.log_dict({"train_auc": auc}, sync_dist=True, prog_bar=True)

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        y_hat = torch.nn.functional.softmax(outputs["logits"], dim=-1)[:, :, 1]
        y_hat = y_hat.detach().cpu().numpy()
        y_true = outputs["label"].detach().cpu().numpy()
        auc = roc_auc_score(y_true, y_hat)
        pl_module.log_dict({"train_auc": auc}, sync_dist=True, prog_bar=True)
