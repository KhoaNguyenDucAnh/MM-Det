import lightning as L
import torch.utils.data as data
import zarr
from torch.utils.data import Dataset


class ZarrDataset(Dataset):
    def __init__(self, input_file, data_group_name, exclude_groups_name=None):
        super().__init__()
        zarr_file = zarr.open_group(input_file, mode="r")
        self.data = zarr_file[data_group_name]
        self.video_id_list = list(self.data)
        if exclude_groups_name != None:
            exclude = set()
            groups = set(zarr_file)
            for group_name in exclude_groups_name:
                if group_name in groups:
                    exclude |= set(zarr_file[group_name])
            self.video_id_list = [
                video for video in self.video_id_list if video not in exclude
            ]

    def __len__(self):
        return len(self.video_id_list)

    def __getitem__(self, index):
        video_id = self.video_id_list[index]
        array = self.data[video_id]
        return video_id, array[:]


class ZarrDataModule(L.LightningDataModule):

    def __init__(self, cache_file_path, batch_size, num_workers):
        super().__init__()
        self.cache_file_path = cache_file_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def predict_dataloader(self):
        return data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: batch,
        )


class ReconstructDataModule(ZarrDataModule):

    def __init__(self, cache_file_path, batch_size, num_workers):
        super().__init__(cache_file_path, batch_size, num_workers)

    def setup(self, stage=None):
        self.dataset = ZarrDataset(self.cache_file_path, "original", ["reconstruct"])


class MMRepresentationDataModule(ZarrDataModule):

    def __init__(self, cache_file_path, batch_size, num_workers):
        super().__init__(cache_file_path, batch_size, num_workers)

    def setup(self, stage=None):
        self.dataset = ZarrDataset(
            self.cache_file_path, "original", ["textual", "visual"]
        )
