from torch.utils.data import Dataset
import zarr


class ZarrDataset(Dataset):
    def __init__(self, input_file, zarr_group_name):
        super().__init__()
        zarr_file = zarr.open_group(input_file, mode="r")
        self.zarr_group = zarr_file[zarr_group_name]
        self.videos = list(self.zarr_group.array_keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        video = self.videos[index]
        array = self.zarr_group[video]
        return video, array[:]
