import csv
import os
import random
import cv2
import json
import numpy as np
import torch
import torchvision
from copy import deepcopy
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split

from .process import get_image_transformation_from_cfg, get_video_transformation_from_cfg
from .utils import get_default_transformation_cfg


class FolderDataset(Dataset):
    def __init__(self, data_root, supported_exts, transform_cfg, selected_cls_labels=None):
        super(FolderDataset, self).__init__()
        self.data_root = data_root
        self.supported_exts = supported_exts
        self.transform_cfg = transform_cfg
        self.selected_cls_labels = selected_cls_labels
        
    def initialize_data_info(self):
        self.classes, self.class_to_idx = self.__find_classes(self.data_root, self.selected_cls_labels)
        self.data_info = self.__make_dataset(self.data_root, self.classes, self.class_to_idx)    # [(path, label), ...]
        
    def __find_classes(self, dir, selected_cls_labels):
        if selected_cls_labels is not None:    # use the provided class and label pairs.
            classes = [d[0] for d in selected_cls_labels]
            class_to_idx = {d[0]: d[1] for d in selected_cls_labels}
        else:    # use all classes
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
        
    def __make_dataset(self, dir, classes, class_to_idx):
        data_info = []
        for cls in classes:
            for root, _, names in os.walk(os.path.join(dir,cls)):
                for name in sorted(names):
                    if os.path.splitext(name)[-1].lower() in self.supported_exts:
                        data_info.append((os.path.join(root, name), class_to_idx[cls]))
        return data_info
    
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        raise NotImplementedError
        

class ImageFolderDataset(FolderDataset):
    def __init__(self, 
                 data_root, 
                 transform_cfg=get_default_transformation_cfg(), 
                 selected_cls_labels=None,    # only folders in the dict are loaded.
                 supported_exts=['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.bmp'],
                **kwargs
                 ):
        super(ImageFolderDataset, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        
        self.initialize_data_info()
    
    def __getitem__(self, idx):
        """
           image: [C, H, W]
           label: [1]
        """
        transform = get_image_transformation_from_cfg(self.transform_cfg)
        img_pil = Image.open(self.data_info[idx][0]).convert('RGB')
        img_cls = torch.tensor(self.data_info[idx][1], dtype=torch.int64)
        img_t = transform(img_pil)
        
        return img_t, img_cls
    
    
class VideoFolderDataset(FolderDataset):
    def __init__(self, 
                 data_root, 
                 sample_size=10, 
                 sample_method='continuous',    # ['random', 'continuous', 'entire']
                 transform_cfg=get_default_transformation_cfg(),
                 repeat_sample_prob = 0.,
                 selected_cls_labels=None, 
                 supported_exts=['.mp4', '.avi', '.wmv', '.mkv', '.flv'],
                **kwargs
                 ):
        super(VideoFolderDataset, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        
        self.initialize_data_info()
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.repeat_sample_prob = repeat_sample_prob
        if sample_method not in ['random', 'continuous', 'entire']:
            raise ValueError(f"Sample method should be either \"random\" or \"continuous\", but not {self.sample_method}")
    
    def __getitem__(self, idx):
        """
           frames: [L, C, H, W]
           label: [1]
        """
        video = self.data_info[idx]
        video_path = video[0]
        label = torch.tensor(video[1], dtype=torch.int64)
        transform = get_video_transformation_from_cfg(self.transform_cfg)

        cap = cv2.VideoCapture(video_path)
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.sample_size > N:
            print(f"The sample size {self.sample_size} is longer than the total frame number {N} of video {video_path}")
        
        sample_index = []
        if self.sample_method == "random":
            sample_index = random.sample(range(N), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(0, N - self.sample_size)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(0, N))
        else:
            raise ValueError("Sample method should be one of \"random\" , \"continuous\" or \"entire\". Get {}".format(self.sample_method))
            
        count = 0
        frames = []
        for frame_index in range(N):
            _, frameread = cap.read()
            if frame_index in sample_index:
                frameread = cv2.cvtColor(frameread, cv2.COLOR_BGR2RGB)
                frameread = transform(Image.fromarray(np.uint8(frameread)))
                frames.append(frameread)
                count += 1
            if count >= len(sample_index):
                break
        cap.release()
        
        if self.repeat_sample_prob > 0.:
            if random.random() > self.repeat_sample_prob:
                repeated_frame = random.choice(frames)
                repeated_frames = []
                for i in range(len(frames)):
                    repeated_frames.append(repeated_frame.clone())
                frames = repeated_frames
            
        return torch.stack(frames, dim=0), label


class VideoFolderDatasetForRecons(FolderDataset):
    def __init__(self, 
                 data_root, 
                 sample_size=10, 
                 sample_method='continuous',    # ['random', 'continuous', 'entire']
                 transform_cfg=get_default_transformation_cfg(),
                 repeat_sample_prob = 0.,
                 selected_cls_labels=None, 
                 supported_exts=['.jpg'],
                 split_file=None,
                 **kwargs
                 ):
        super(VideoFolderDatasetForRecons, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts,
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.repeat_sample_prob = repeat_sample_prob
        self.split_file = split_file
        if sample_method not in ['random', 'continuous', 'entire']:
            raise ValueError(f"Sample method should be either \"random\" or \"continuous\", but not {self.sample_method}")
        self.initialize_split_data_info()
    
    def initialize_split_data_info(self):
        if self.split_file is None:
            self.classes, self.class_to_idx = self.__find_classes(self.data_root, self.selected_cls_labels)
            self.data_info = self.__make_dataset(self.data_root, self.classes, self.class_to_idx)    # [(path, label), ...]
        else:
            self.data_info = self.__make_split_dataset(self.data_root, self.split_file)
         
    def __find_classes(self, dir, selected_cls_labels):
        if selected_cls_labels is not None:    # use the provided class and label pairs.
            classes = [d[0] for d in selected_cls_labels]
            class_to_idx = {d[0]: d[1] for d in selected_cls_labels}
        else:    # use all classes
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def __make_dataset(self, dir, classes, class_to_idx):
        data_info = []
        for cls in classes:
            ddir = os.path.join(dir, cls)
            if not os.path.exists(ddir):
                continue
            fds = os.listdir(ddir)
            assert('original' in fds and 'recons' in fds)
            data_elems = dict()
            for name in sorted(os.listdir(os.path.join(ddir, 'original'))):
                fn, ext = os.path.splitext(name)
                if ext in self.supported_exts:
                    prefix = fn.rsplit('_', maxsplit=1)[0]
                    if prefix not in data_elems:
                        data_elems[prefix] = 1
                    else:
                        data_elems[prefix] += 1
            for prefix, count in data_elems.items():
                data_info.append(((ddir, prefix, count), class_to_idx[cls]))
        return data_info
    
    def __make_split_dataset(self, data_root, split_file):
        data_info = []
        with open(split_file) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            all_dir_prefixs = dict()
            for row in csv_reader:
                file_name = row['file_name']
                label = int(row['label'])
                base_path = os.path.dirname(file_name)
                fn = os.path.basename(file_name)
                fn_prefix = os.path.splitext(fn)[0]
                if base_path not in all_dir_prefixs:
                    all_dir_prefixs[base_path] = list()
                all_dir_prefixs[base_path].append((fn_prefix, label))
            for base_dir, prefix_info in all_dir_prefixs.items():
                data_elems = dict()
                prefixs = list()
                for prefix_pair in prefix_info:
                    prefixs.append(prefix_pair[0])
                for name in sorted(os.listdir(os.path.join(data_root, base_dir, 'recons'))):
                    fn, ext = os.path.splitext(name)
                    if ext in self.supported_exts:
                        fn_prefix = fn.rsplit('_', maxsplit=1)[0]
                        if fn_prefix in prefixs:
                            if fn_prefix not in data_elems:
                                data_elems[fn_prefix] = 1
                            else:
                                data_elems[fn_prefix] += 1
                for prefix_pair in prefix_info:
                    fn_prefix, label = prefix_pair
                    data_info.append(((os.path.join(data_root, base_dir), fn_prefix, data_elems[fn_prefix]), label))
        return data_info
    
    def __getitem__(self, idx):
        """
           frames: [L, C, H, W]
           label: [1]
        """
        video = self.data_info[idx]
        video_ddir, video_prefix, video_length = video[0]
        label = torch.tensor(video[1], dtype=torch.int64)
        transform = get_video_transformation_from_cfg(self.transform_cfg)

        if self.sample_size > video_length:
            raise ValueError(f"The sample size {self.sample_size} is longer than the total frame number {video_length} of video {os.path.join(video_ddir, video_prefix)}")
        
        sample_index = []
        if self.sample_method == "random":
            sample_index = random.sample(range(1, video_length + 1), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(1, video_length - self.sample_size + 1)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(1, video_length + 1))
        else:
            raise ValueError("Sample method should be one of \"random\" , \"continuous\" or \"entire\". Get {}".format(self.sample_method))
            
        original_frames, recons_frames = [], []
        for frame_index in sample_index:
            original_frame = Image.open(os.path.join(video_ddir, 'original', video_prefix + f'_{frame_index}.jpg')).convert('RGB')
            transformed_frame = transform(original_frame)
            original_frames.append(transformed_frame)
            recons_frame = Image.open(os.path.join(video_ddir, 'recons', video_prefix + f'_{frame_index}.jpg')).convert('RGB')
            transformed_recons_frame = transform(recons_frame)
            recons_frames.append(transformed_recons_frame)
       
        if self.repeat_sample_prob > 0.:
            if random.random() > self.repeat_sample_prob:
                repeated_index = random.choice(range(len(original_frames)))
                repeated_original_frames = []
                for i in range(len(original_frames)):
                    repeated_original_frames.append(original_frames[repeated_index].clone())
                original_frames = repeated_original_frames
                repeated_recons_frames = []
                for i in range(len(recons_frames)):
                    repeated_recons_frames.append(recons_frames[repeated_index].clone())
                recons_frames = repeated_recons_frames
                
        return (torch.stack(original_frames, dim=0), torch.stack(recons_frames, dim=0)), label
    

class VideoFolderDatasetForReconsWithFn(FolderDataset):
    def __init__(
        self, 
        data_root, 
        sample_size=10, 
        sample_method='continuous',    # ['random', 'continuous', 'entire']
        transform_cfg=get_default_transformation_cfg(),
        repeat_sample_prob = 0.,
        selected_cls_labels=None, 
        supported_exts=['.jpg'],
        split=None,
        **kwargs,
    ):
        super(VideoFolderDatasetForReconsWithFn, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.repeat_sample_prob = repeat_sample_prob
        self.split = split
        if sample_method not in ['random', 'continuous', 'entire']:
            raise ValueError(f"Sample method should be either \"random\" or \"continuous\", but not {self.sample_method}")
        self.initialize_split_data_info()
    
    def initialize_split_data_info(self):
        if self.split is None:
            self.classes, self.class_to_idx = self.__find_classes(self.data_root, self.selected_cls_labels)
            self.data_info = self.__make_dataset(self.data_root, self.classes, self.class_to_idx)    # [(path, label), ...]
        else:
            self.data_info = self.__make_split_dataset(self.data_root, self.split)
         
    def __find_classes(self, dir, selected_cls_labels):
        if selected_cls_labels is not None:    # use the provided class and label pairs.
            classes = [d[0] for d in selected_cls_labels]
            class_to_idx = {d[0]: d[1] for d in selected_cls_labels}
        else:    # use all classes
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def __make_dataset(self, dir, classes, class_to_idx):
        data_info = []
        for cls in classes:
            ddir = os.path.join(dir, cls)
            if not os.path.exists(ddir):
                continue
            fds = os.listdir(ddir)
            assert('original' in fds and 'recons' in fds)
            data_elems = dict()
            for name in sorted(os.listdir(os.path.join(ddir, 'original'))):
                fn, ext = os.path.splitext(name)
                if ext in self.supported_exts:
                    prefix = fn.rsplit('_', maxsplit=1)[0]
                    if prefix not in data_elems:
                        data_elems[prefix] = 1
                    else:
                        data_elems[prefix] += 1
            for prefix, count in data_elems.items():
                data_info.append(((ddir, prefix, count), class_to_idx[cls]))
        return data_info
    
    def __make_split_dataset(self, data_root, split):
        data_info = []
        all_dir_prefixs = dict()
        for file_name, label in split.items():
            base_path = os.path.dirname(file_name)
            fn = os.path.basename(file_name)
            fn_prefix = os.path.splitext(fn)[0]
            if base_path not in all_dir_prefixs:
                all_dir_prefixs[base_path] = list()
            all_dir_prefixs[base_path].append((fn_prefix, label))
        for base_dir, prefix_info in all_dir_prefixs.items():
            data_elems = dict()
            prefixs = list()
            for prefix_pair in prefix_info:
                prefixs.append(prefix_pair[0])
            for name in sorted(os.listdir(os.path.join(data_root, base_dir, 'recons'))):
                fn, ext = os.path.splitext(name)
                if ext in self.supported_exts:
                    fn_prefix = fn.rsplit('_', maxsplit=1)[0]
                    if fn_prefix in prefixs:
                        if fn_prefix not in data_elems:
                            data_elems[fn_prefix] = 1
                        else:
                            data_elems[fn_prefix] += 1
            for prefix_pair in prefix_info:
                fn_prefix, label = prefix_pair
                data_info.append(((os.path.join(data_root, base_dir), fn_prefix, data_elems[fn_prefix]), label))
        return data_info
    
    def __getitem__(self, idx):
        """
           frames: [L, C, H, W]
           label: [1]
        """
        video = self.data_info[idx]
        video_ddir, video_prefix, video_length = video[0]
        label = torch.tensor(video[1], dtype=torch.int64)
        transform = get_video_transformation_from_cfg(self.transform_cfg)

        if self.sample_size > video_length:
            raise ValueError(f"The sample size {self.sample_size} is longer than the total frame number {video_length} of video {os.path.join(video_ddir, video_prefix)}")
        
        sample_index = []
        if self.sample_method == "random":
            sample_index = random.sample(range(1, video_length + 1), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(1, video_length - self.sample_size + 1)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(1, video_length + 1))
        else:
            raise ValueError("Sample method should be one of \"random\" , \"continuous\" or \"entire\". Get {}".format(self.sample_method))
            
        original_frames, recons_frames = [], []
        for frame_index in sample_index:
            original_frame = Image.open(os.path.join(video_ddir, 'original', video_prefix + f'_{frame_index}.jpg')).convert('RGB')
            transformed_frame = transform(original_frame)
            original_frames.append(transformed_frame)
            recons_frame = Image.open(os.path.join(video_ddir, 'recons', video_prefix + f'_{frame_index}.jpg')).convert('RGB')
            transformed_recons_frame = transform(recons_frame)
            recons_frames.append(transformed_recons_frame)
       
        if self.repeat_sample_prob > 0.:
            if random.random() > self.repeat_sample_prob:
                repeated_index = random.choice(range(len(original_frames)))
                repeated_original_frames = []
                for i in range(len(original_frames)):
                    repeated_original_frames.append(original_frames[repeated_index].clone())
                original_frames = repeated_original_frames
                repeated_recons_frames = []
                for i in range(len(recons_frames)):
                    repeated_recons_frames.append(recons_frames[repeated_index].clone())
                recons_frames = repeated_recons_frames
                
        return os.path.join(video_ddir, f'{video_prefix}__{sample_index[0]}'), (torch.stack(original_frames, dim=0), torch.stack(recons_frames, dim=0)), label
    
    
def random_split_dataset(dataset, split_ratio, seed=100):
    total_ratio = sum(split_ratio)
    split_ratio = [x / total_ratio for x in split_ratio]
    dataset_len = len(dataset)
    dataset_split_len = [int(x * dataset_len) for x in split_ratio]
    dataset_split_len[-1] = int(dataset_len - sum(dataset_split_len) + dataset_split_len[-1])
    return random_split(dataset=dataset, lengths=dataset_split_len, generator=torch.Generator().manual_seed(seed))


def collate_video_folder_dataset(batch):
    datas = list()
    labels = list()
    for pack in batch:
        data, label = pack
        datas.append(data)
        labels.append(label)
    return torch.stack(datas, dim=0), torch.stack(labels, dim=0)


def collate_video_folder_for_recons_dataset_with_fn(batch):
    fns = list()
    original_data = list()
    recons_data = list()
    labels = list()
    for pack in batch:
        fn, data, label = pack
        fns.append(fn)
        original_data.append(data[0])
        recons_data.append(data[1])
        labels.append(label)
    return fns, (torch.stack(original_data, dim=0), torch.stack(recons_data, dim=0)), torch.stack(labels, dim=0)


__DATASET = {
    "VideoFolderDataset": VideoFolderDataset,
    "VideoFolderDatasetForReconsWithFn": VideoFolderDatasetForReconsWithFn,
}


__COLLATE_FN = {
    "VideoFolderDataset": collate_video_folder_dataset,
    "VideoFolderDatasetForReconsWithFn": collate_video_folder_for_recons_dataset_with_fn,
}


def get_dataloader(dataset, mode='train', bs=32, workers=4, **kwargs):
    params = {'batch_size': bs,
            'shuffle': (mode=='train'),
            'num_workers': workers,
            'drop_last' : (mode=='train')
    }
    for k, v in kwargs.items():
        params[k] = v
    return DataLoader(dataset, **params)


def get_train_dataloader(config):
    train_datasets = []
    val_datasets = []
    _dataset_type = None
    for dname, dargs in config["datasets"].items():
        transformation_cfg = get_default_transformation_cfg(mode=config["mode"])
        if 'transform' in dargs:
            for k, v in dargs['transform'].items():
                transformation_cfg[k] = v
        dataset_type = dargs['dataset_type']
        if _dataset_type is None:
            _dataset_type = dataset_type
        elif _dataset_type != dataset_type:
            raise ValueError(f'Expect all datasets in the same type. Get "{_dataset_type}" and "{dataset_type}".')           
        dargs['transform_cfg'] = transformation_cfg
        if 'split' in dargs:
            train_dargs = deepcopy(dargs)
            train_dargs['split'] = dargs['split']['train']
            train_datasets.append(__DATASET[dataset_type](**train_dargs))
            val_dargs = deepcopy(dargs)
            val_dargs['split'] = dargs['split']['val']
            val_datasets.append(__DATASET[dataset_type](**val_dargs))
        else:
            train_datasets.append(__DATASET[dataset_type](**dargs))
    train_datasets = ConcatDataset(train_datasets)
    if len(val_datasets) == 0:    # not split yet
        train_datasets, val_datasets = random_split_dataset(train_datasets, config["split_ratio"])
    else:
        val_datasets = ConcatDataset(val_datasets)
    train_params = {'batch_size': config["bs"],
            'shuffle': (config["mode"]=='train'),
            'num_workers': config["num_workers"] * config["gpus"],
            'drop_last' : (config["mode"]=='train'),
            'collate_fn': __COLLATE_FN[dataset_type],
    }   
    val_params = {'batch_size': 1,
            'shuffle': False,
            'num_workers': 1,
            'drop_last' : False,
            'collate_fn': __COLLATE_FN[dataset_type],
    }
    return DataLoader(train_datasets, **train_params), DataLoader(val_datasets, **val_params)


def get_test_dataloader(test_dataset_config):
    test_datasets = []
    _dataset_type = None
    for dname, dargs in test_dataset_config.items():
        transformation_cfg = get_default_transformation_cfg(mode='test')
        dataset_type = dargs['dataset_type']
        if _dataset_type is None:
            _dataset_type = dataset_type
        elif _dataset_type != dataset_type:
            raise ValueError(f'Expect all datasets in the same type. Get "{_dataset_type}" and "{dataset_type}".')           
        dargs['transform_cfg'] = transformation_cfg
        test_datasets.append(__DATASET[dataset_type](**dargs))
    test_datasets = ConcatDataset(test_datasets)
    test_params = {'batch_size': 1,
            'shuffle': True,
            'num_workers': 1,
            'drop_last' : False,
            'collate_fn': __COLLATE_FN[dataset_type],
    }
    return DataLoader(test_datasets, **test_params)
