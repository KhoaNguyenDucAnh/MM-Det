from torch.utils.data import Dataset
import zarr


class ZarrDataset(Dataset):
    def __init__(self, input_file, data_group_name, exclude_group_name=None):
        super().__init__()
        zarr_file = zarr.open_group(input_file, mode="r")
        self.data = zarr_file[data_group_name]
        self.video_id_list = list(self.data.array_keys())
        if exclude_group_name != None:
            exclude = list(zarr_file[exclude_group_name].array_keys())
            self.video_id_list = [video for video in self.video_id_list if video not in exclude]

    def __len__(self):
        return len(self.video_id_list)

    def __getitem__(self, index):
        video_id = self.video_id_list[index]
        array = self.data[video_id]
        return video_id, array[:]
