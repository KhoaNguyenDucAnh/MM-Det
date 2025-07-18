import torch.utils.data as data
import zarr


class ZarrDataset(data.Dataset):
    def __init__(self, input_file, zarr_group_name):
        zarr_file = zarr.open_group(input_file, mode="r")
        self.zarr_group = zarr_file[zarr_group_name]
        self.keys = list(self.zarr_group.array_keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        array = self.zarr_group[key]
        return key, array[:]
