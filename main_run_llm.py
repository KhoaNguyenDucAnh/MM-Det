# coding: utf-8
# author: Xiao Guo
import json
import os
import numpy as np
import random
import subprocess
from datetime import datetime
import logging
import sys,signal
from torch.utils import data
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import datetime
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, utils
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from dataset import ImageFolderDataset, VideoFolderDataset, VideoFolderDatasetRestricted, VideoFolderDatasetCachedForRecons, VideoFolderDatasetSplit, get_dataloader, VideoFolderDatasetSplitFixedSample, random_split_dataset, VideoFolderDatasetCachedForReconsSplitFn
from earlystop import EarlyStopping

from sequence.models.test_model_llm import AdaptiveSTViTReconsWithSeparateTemporal_LlavaSE, AdaptiveSTViTReconsWithSeparateTemporal_LlavaAddCLIP, AdaptiveSTViTReconsWithSeparateTemporal_LlavaCatCLIP, AdaptiveSTViTReconsWithSeparateTemporal_LlavaCatCLIPLlava, AdaptiveSTViTReconsWithSeparateTemporal_LlavaSECLIPLlava, \
    AdaptiveSTViTReconsWithSeparateTemporal_LlavaLayerSECLIPLlava, AdaptiveSTViTReconsWithSeparateTemporal_LlavaLayerSECLIPLlava_OnlyLLM, AdaptiveSTViTReconsWithSeparateTemporal_LlavaWeightedFusion
from sequence.torch_utils import eval_model,display_eval_tb,train_logging,get_lr_blocks,associate_param_with_lr,lrSched_monitor, step_train_logging, Metrics
from sequence.runjobs_utils import init_logger,Saver,DataConfig,torch_load_model,get_iter,get_data_to_copy_str


def get_train_transformation_cfg():
    cfg = {
        # 'resize': {
        #     'img_size': 224
        # },
        'post': {
            'blur': {
                'prob': 0.1,
                'sig': [0.0, 3.0]
            },
            'jpeg': {
                'prob': 0.1,
                'method': ['cv2', 'pil'],
                'qual': [30, 100]
            },
            'noise':{
                'prob': 0.0,
                'var': [0.01]
            }
        },
        'crop': {
            'img_size': 224,
            'type': 'random'    # ['center', 'random'], according to 'train' or 'test' mode
        },
        'flip': True,    # set false when testing
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    return cfg


def get_val_transformation_cfg():
    cfg = {
        # 'resize': {
        #     'img_size': 224
        # },
        'post': {
            'blur': {
                'prob': 0.0,
                'sig': [0.0, 3.0]
            },
            'jpeg': {
                'prob': 0.0,
                'method': ['cv2', 'pil'],
                'qual': [30, 100]
            },
            'noise':{
                'prob': 0.0,
                'var': [0.01]
            }
        },
        'crop': {
            'img_size': 224,
            'type': 'center'    # ['center', 'random'], according to 'train' or 'test' mode
        },
        'flip': False,    # set false when testing
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    return cfg


starting_time = datetime.datetime.now()

## Deterministic training
_seed_id = 100
torch.backends.cudnn.deterministic = True
torch.manual_seed(_seed_id)

exp_name = '0001_recons_w10_mmfr'
model_name = exp_name
model_path = './expts_mmdet_svd'
model_path = os.path.join(model_path, model_name)

# Create the model path if doesn't exists
if not os.path.exists(model_path):
    subprocess.call(f"mkdir -p {model_path}", shell=True)
    
datasets = ['original', 'svd']
manipulations_names = [n for c, n in enumerate(datasets) if n != 'original']
manipulations_dict = {n : c  for c, n in enumerate(manipulations_names) }
manipulations_dict['original'] = 255

for key, value in manipulations_dict.items():
	print(key, value)

gpus = 1

# logger for training
logger = init_logger(__name__)
logger.setLevel(logging.INFO)
out_handler = logging.FileHandler(filename=os.path.join(model_path, 'train.log'))
out_handler.setLevel(level=logging.INFO)
logger.addHandler(out_handler)


## Hyper-params #######################
hparams = {
            'epochs': 200, 'batch_size': 2, 'basic_lr': 1e-4, 'fine_tune': True, 'use_laplacian': True, 'step_factor': 0.2, 
            'patience': 4, 'weight_decay': 1e-06, 'lr_gamma': 2.0, 'use_magic_loss': True, 'feat_dim': 2048, 'drop_rate': 0.2, 
            'skip_valid': False, 'rnn_type': 'LSTM', 'rnn_hidden_size': 256, 'num_rnn_layers': 1, 'rnn_drop_rate': 0.2, 
            'bidir': True, 'merge_mode': 'concat', 'perc_margin_1': 0.95, 'perc_margin_2': 0.95, 'soft_boundary': False, 
            'dist_p': 2, 'radius_param': 0.84, 'strat_sampling': True, 'normalize': True, 'window_size': 10, 'hop': 1, 
            'valid_step': 1000, 'display_step': 50, 'use_sched_monitor': True, 'level': 'video', 'save_epoch': 5
            }
batch_size = hparams['batch_size']
basic_lr = hparams['basic_lr']
fine_tune = hparams['fine_tune']
use_laplacian = hparams['use_laplacian']
step_factor = hparams['step_factor']
patience = hparams['patience']
weight_decay = hparams['weight_decay']
lr_gamma = hparams['lr_gamma']
use_magic_loss = hparams['use_magic_loss']
feat_dim = hparams['feat_dim']
drop_rate = hparams['drop_rate']
rnn_type = hparams['rnn_type']
rnn_hidden_size = hparams['rnn_hidden_size']
num_rnn_layers = hparams['num_rnn_layers']
rnn_drop_rate = hparams['rnn_drop_rate']
bidir = hparams['bidir']
merge_mode = hparams['merge_mode']
perc_margin_1 = hparams['perc_margin_1']
perc_margin_2 = hparams['perc_margin_2']
dist_p = hparams['dist_p']
radius_param = hparams['radius_param']
strat_sampling = hparams['strat_sampling']
normalize = hparams['normalize']
window_size = hparams['window_size']
hop = hparams['hop']
soft_boundary = hparams['soft_boundary']
use_sched_monitor = hparams['use_sched_monitor']
level = hparams['level']    # 'frame' or 'video'
valid_step = hparams['valid_step']

logger.info(hparams)
########################################

accum_grad_loop = batch_size
workers_per_gpu = 4


v2_data_root = "/home/sxf/data/svd_v2"
v2_partition_data_root = "/home/sxf/data/svd_v2"
v3_data_root = "/home/sxf/data/svd_v3"
zeroscope_data_root = "/home/sxf/data/zeroscope"
v3_partition_data_root = "/home/sxf/data/svd_v3_partition"
videocrafter_v2_path = f'/home/sxf/data/videocrafter_v2'

split_files = [
    '/home/sxf/data/svd_v3_partition/split/split_1.csv',
    '/home/sxf/data/svd_v3_partition/split/split_2.csv',
    '/home/sxf/data/svd_v3_partition/split/split_3.csv',
    '/home/sxf/data/svd_v3_partition/split/split_4.csv',
    '/home/sxf/data/svd_v3_partition/split/split_5.csv',
    '/home/sxf/data/svd_v3_partition/split/split_6.csv'
]
split_v2_files = [
    '/home/sxf/data/svd_v3_partition/split/split_fake_v2.csv',
    '/home/sxf/data/svd_v3_partition/split/split_real_v2.csv'
]
split_short_v2_file = '/home/sxf/data/svd_v3_partition/split/split_real_short_v2.csv'
sub_split_files = [
    '/home/sxf/data/svd_v3_partition/split/split_1_1.csv'
]
split_only_sample_files = [
    '/home/sxf/data/svd_v3_partition/split/split_1_only_samples.csv'
]
split_v2_val_files = [
    "/home/sxf/data/svd_v3_partition/split/split_val_real_short_v2.csv",
    "/home/sxf/data/svd_v3_partition/split/split_val_fake_v2.csv"
]
split_high_level_train_files = [
    "/home/sxf/data/svd_v3_partition/split/split_train_real_high_level.csv",
    "/home/sxf/data/svd_v3_partition/split/split_train_fake_high_level.csv"
]
split_high_level_val_files = [
    "/home/sxf/data/svd_v3_partition/split/split_val_real_high_level.csv",
    "/home/sxf/data/svd_v3_partition/split/split_val_fake_high_level.csv"
]
zeroscope_split_files = [
    "/home/sxf/data/zeroscope/split_train/split_1.csv"
]
zeroscope_partition_data_root = "/home/sxf/data/zeroscope/train"
# train_transformation_cfg = get_train_transformation_cfg()
# train_dataset = VideoFolderDatasetRestricted(data_root=f'{data_root}/train', sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg, repeat_sample_prob=1.0, restricted_ref='/home/sxf/data/svd_v3/level_1.json')
# val_transformation_cfg = get_val_transformation_cfg()
# val_dataset = VideoFolderDatasetRestricted(data_root=f'{data_root}/val', sample_size=window_size, sample_method='continuous', transform_cfg=val_transformation_cfg, repeat_sample_prob=1.0, restricted_ref='/home/sxf/data/svd_v3/level_1.json')

train_transformation_cfg = get_train_transformation_cfg()

# split_index = 0
# split_dataset = VideoFolderDatasetSplit(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg, split_file=split_files[split_index])
# sub_split_dataset_1 = VideoFolderDatasetSplit(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg, split_file=sub_split_files[0])
# split_only_sample_dataset = VideoFolderDatasetSplit(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg, split_file=split_only_sample_files[0])
# v3_dataset = VideoFolderDatasetSplit(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg)
# v3_split_fake_dataset_v2 = VideoFolderDatasetSplit(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg, split_file=split_v2_files[0])
# v3_split_real_dataset_v2 = VideoFolderDatasetSplit(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg, split_file=split_v2_files[1])

# v3_split_real_fixed_sample_dataset_v2 = VideoFolderDatasetSplitFixedSample(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg, split_file=split_v2_files[1])
# v3_split_real_short_dataset_v2 = VideoFolderDatasetSplit(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg, split_file=split_short_v2_file)
# v3_real_train_high_level_dataset = VideoFolderDatasetSplit(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg, split_file=split_high_level_train_files[0])
# v3_fake_train_high_level_dataset = VideoFolderDatasetSplit(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg, split_file=split_high_level_train_files[1])


# v2_real_train_dataset = VideoFolderDatasetRestricted(data_root=f'{v2_partition_data_root}/train', sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg, selected_cls_labels=[('0_real', 0)])
# v2_fake_train_dataset = VideoFolderDatasetRestricted(data_root=f'{v2_partition_data_root}/train', sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg, selected_cls_labels=[('1_fake', 1)])
# v3_train_dataset = VideoFolderDatasetRestricted(data_root=f'{data_root}/train', sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg)
# v3_real_train_dataset = VideoFolderDatasetRestricted(data_root=f'{v3_data_root}/train', sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg, selected_cls_labels=[('0_real', 0)], repeat_sample_prob=0.05, restricted_ref='/home/sxf/data/svd_v3/level_1.json')
# v3_fake_train_dataset = VideoFolderDatasetRestricted(data_root=f'{v3_data_root}/train', sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg, selected_cls_labels=[('1_fake', 1)], repeat_sample_prob=0.05, restricted_ref='/home/sxf/data/svd_v3/level_1.json')
# zeroscope_train_dataset = VideoFolderDatasetRestricted(data_root=f'{zeroscope_data_root}/train', sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg)
# zeroscope_val_dataset = VideoFolderDatasetRestricted(data_root=f'{zeroscope_data_root}/val', sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg)
# videocrafter_v2_dataset = VideoFolderDataset(data_root='/home/aya/workspace/data/videocrafter_v2', sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg)

v2_recons_dataset_real = VideoFolderDatasetCachedForReconsSplitFn(data_root=f'/home/songxiufeng/data/all_recons/svd_v2_recons', sample_size=window_size, transform_cfg=train_transformation_cfg, selected_cls_labels=[('0_real', 0)])
v3_recons_dataset_fake = VideoFolderDatasetCachedForReconsSplitFn(data_root=f'/home/songxiufeng/data/all_recons/svd_v3_recons', sample_size=window_size, transform_cfg=train_transformation_cfg, selected_cls_labels=[('1_fake', 1)])

# train_dataset = VideoFolderDatasetRestricted(data_root=f'{data_root}/train', sample_size=window_size, sample_method='continuous', transform_cfg=train_transformation_cfg, repeat_sample_prob=1.0)
val_transformation_cfg = get_val_transformation_cfg()

# former_val_dataset = VideoFolderDatasetSplit(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=val_transformation_cfg, split_file=split_files[split_index - 1])
# latter_val_dataset = VideoFolderDatasetSplit(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=val_transformation_cfg, split_file=split_files[split_index + 1])
# v3_split_val_real_short_dataset_v2 = VideoFolderDatasetSplit(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=val_transformation_cfg, split_file=split_v2_val_files[0])
# v3_split_val_fake_dataset_v2 = VideoFolderDatasetSplit(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=val_transformation_cfg, split_file=split_v2_val_files[1])
# v3_val_real_high_level_dataset = VideoFolderDatasetSplit(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=val_transformation_cfg, split_file=split_high_level_val_files[0])
# v3_val_fake_high_level_dataset = VideoFolderDatasetSplit(data_root=v3_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=val_transformation_cfg, split_file=split_high_level_val_files[1])
# zeroscope_val_dataset = VideoFolderDatasetSplit(data_root=zeroscope_partition_data_root, sample_size=window_size, sample_method='continuous', transform_cfg=val_transformation_cfg, split_file=zeroscope_split_files[0])

# val_dataset = VideoFolderDatasetRestricted(data_root=f'{data_root}/val', sample_size=window_size, sample_method='continuous', transform_cfg=val_transformation_cfg, repeat_sample_prob=1.0)

train_dataset = torch.utils.data.ConcatDataset([v2_recons_dataset_real, v3_recons_dataset_fake])
train_dataset, val_dataset = random_split_dataset(train_dataset, [0.9, 0.1], _seed_id)



def collate_fn_recons_dataset(batch):
    fns = list()
    original_datas = list()
    recons_datas = list()
    labels = list()
    for pack in batch:
        fn, data, label = pack
        fns.append(fn)
        original_datas.append(data[0])
        recons_datas.append(data[1])
        labels.append(label)
    return fns, (torch.stack(original_datas, dim=0), torch.stack(recons_datas, dim=0)), torch.stack(labels, dim=0)

# test_transformation_cfg = get_val_transformation_cfg()
# test_svd_v2_dataset = VideoFolderDataset(data_root=f'{data_root}/test', sample_size=5, sample_method='continuous', transform_cfg=test_transformation_cfg)
train_generator = get_dataloader(dataset=train_dataset, mode='train', bs=batch_size, drop_last=True, workers=workers_per_gpu * gpus, collate_fn=collate_fn_recons_dataset)
val_generator = get_dataloader(dataset=val_dataset, mode='test', bs=batch_size, workers=workers_per_gpu * gpus, collate_fn=collate_fn_recons_dataset)
# val_generator = train_generator

# data_root = "/home/sxf/data/svd_v2_recons"
# train_transformation_cfg = get_train_transformation_cfg()
# train_svd_v2_dataset = VideoFolderDatasetCachedForRecons(data_root=f'{data_root}/train', sample_size=8, sample_method='continuous', transform_cfg=train_transformation_cfg)
# val_transformation_cfg = get_val_transformation_cfg()
# val_svd_v2_dataset = VideoFolderDatasetForRecons(data_root=f'{data_root}/val', sample_size=8, sample_method='continuous', transform_cfg=val_transformation_cfg)
# test_transformation_cfg = get_val_transformation_cfg()
# test_svd_v2_dataset = VideoFolderDatasetForRecons(data_root=f'{data_root}/test', sample_size=8, sample_method='continuous', transform_cfg=test_transformation_cfg)
# train_generator = get_dataloader(dataset=train_svd_v2_dataset, mode='train', bs=batch_size, workers=workers_per_gpu * gpus)
# test_generator = get_dataloader(dataset=test_svd_v2_dataset, mode='test', bs=batch_size, workers=workers_per_gpu * gpus)
# val_generator = train_generator

## Model definition
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = ViT_hybrid_H(residual=True)
# model = ViTHybridFreqCLIP_V()
# model = ViTHybridFreqCLIP_V_Twostage()
# model = STViTFreq()
# model = AdaptiveSTViTWithSeparateTemporal(window_size=window_size)
# model = AdaptiveSTViTWithSeparateTemporal_CLIP(window_size=window_size)
# model = AdaptiveSTViTReconsWithSeparateTemporal_CLIP(window_size=window_size)
# model = AdaptiveSTViTReconsWithSeparateTemporal_LlavaCat(window_size=window_size)
# model = AdaptiveSTViTReconsWithSeparateTemporal_LlavaAddCLIP(window_size=window_size, load_llm=False)
# model = AdaptiveSTViTReconsWithSeparateTemporal_LlavaCatCLIPLlava(window_size=window_size, load_llm=False)
# model = AdaptiveSTViTReconsWithSeparateTemporal_LlavaLayerSECLIPLlava(window_size=window_size, load_llm=False)

# model = AdaptiveSTViTReconsWithSeparateTemporal_LlavaLayerSECLIPLlava_OnlyLLM(window_size=window_size, load_llm=False)
model = AdaptiveSTViTReconsWithSeparateTemporal_LlavaWeightedFusion(window_size=window_size, load_llm=False)

# model = Resnet50(window_size=window_size)
# model = Resnet_stem_spatial(window_size=window_size)
# model = AdaptiveSTViTWithSeparateTemporal_NOCLIP(window_size=window_size)
# model = AdaptiveSTViTWithSeparateTemporal_OnlyCLIP(window_size=window_size)
# model = AdaptiveSTViTFreq(window_size=window_size)
# model = ViTHybridFreq_V_CNNDet(window_size=window_size)
# model = ViT_hybrid_Two_Branch()
# model = AdaptiveViT_CLIP_Abla(window_size=window_size)
# model = ViT_hybrid_Two_Branch_Recons()
logger.info(model)
model = model.to(device)
model = torch.nn.DataParallel(model).to(device)

## Fine-tuning functions
params_to_optimize = model.parameters()
## Option 1:
# optimizer = torch.optim.Adam(params_to_optimize, lr=basic_lr, weight_decay=weight_decay)
## Option 2:

## Attention!!! Only the parameters that is declared in the models "assign_lr_dict_list()" will be optimized.
optimizer = torch.optim.Adam(model.module.assign_lr_dict_list(lr=basic_lr), weight_decay=weight_decay)

lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=step_factor, min_lr=1e-08, patience=patience, cooldown=5, verbose=True)
# lr_scheduler = StepLR(optimizer, step_size=500, gamma=0.1, last_epoch=-1)
criterion = nn.CrossEntropyLoss()


## Re-loading the model in case
epoch_init=epoch=ib=ib_off=before_train=0

# model_ckpt_path = model_path
model_ckpt_path = 'expts_all_abla/0021_llm_layer_se'
load_model_path = os.path.join(model_ckpt_path,'current_model_10.pth')
val_loss = np.inf
if os.path.exists(load_model_path):
    logger.info(f'Loading weights, optimizer and scheduler from {load_model_path}...')
    ib_off, epoch_init, scheduler, val_loss = torch_load_model(model, optimizer, load_model_path, strict=False)

## Saver object and data config
data_config = DataConfig(model_path, model_name)
saver = Saver(model, optimizer, lr_scheduler, data_config, starting_time, hours_limit=23, mins_limit=0)

## Writer summary for tb
tb_folder = os.path.join(model_path, 'tb_logs',model_name)
writer = SummaryWriter(tb_folder)
log_string_config = '  '.join([k+':'+str(v) for k,v in hparams.items()])
writer.add_text('config : %s' % model_name, log_string_config, 0)


cached_feature_path = '/home1/songxiufeng/workspace/imdl/Rec_Video_Det/outputs/llava_dump_nips24'
real_dir = 'svd_v2_real'
fake_dir = 'svd_v3_fake'
representation_fn = 'llava_representation.pth'
cached_representation = {
    "real": torch.load(os.path.join(cached_feature_path, real_dir, representation_fn)),
    "fake": torch.load(os.path.join(cached_feature_path, fake_dir, representation_fn))
}

cached_representation_index = {
    'real': {},
    'fake': {}
}

for label in ['real', 'fake']:
    for fn in cached_representation[label].keys():
        base_fn = os.path.splitext(fn)[0]
        prefix = base_fn.rsplit('_', maxsplit=1)[0]
        index = base_fn.rsplit('_',maxsplit=1)[-1]
        if prefix not in cached_representation_index[label].keys():
            cached_representation_index[label][prefix] = [index]
        else:
            cached_representation_index[label][prefix].append(index)


def find_nearest_key(cached_all_indexs, prefix, index):
    all_indexs = cached_all_indexs[prefix]
    previous_indexs = []
    for i in all_indexs:
        if i <= index:
            previous_indexs.append(i)
    return max(previous_indexs)
    
    
if epoch_init == 0:
    model.zero_grad()

## Start training
tot_iter = 0
for epoch in range(epoch_init,hparams['epochs']):
    logger.info(f'Epoch ############: {epoch}')
    train_loss = 0
    total_loss = 0
    total_accu = 0
    logger.info(f"Epoch: {epoch}, learning_rate: {optimizer.param_groups[0]['lr']}")

    for ib, (fns, img_batch, true_labels) in enumerate(train_generator, 1):
        llava_representations = []
        clip_representations = []
        for fn in fns:
            label = 'real' if 'real/' in fn else 'fake'
            base_fn = os.path.basename(fn)
            prefix = base_fn.rsplit('__', maxsplit=1)[0]
            index = base_fn.rsplit('__',maxsplit=1)[-1]
            cached_index = find_nearest_key(cached_representation_index[label], prefix, index)
            clip_representations.append(cached_representation[label][f'{prefix}_{cached_index}.jpg']['clip'])
            llava_representations.append(cached_representation[label][f'{prefix}_{cached_index}.jpg']['llava'])
        clip_representations = torch.stack(clip_representations, dim=0).to(device)
        llava_representations = torch.stack(llava_representations, dim=0).to(device)

        original, recons = img_batch
        original, recons = original.float().to(device), recons.float().to(device)
        B, L, C, H, W = original.shape
        img_batch = original, recons
        
        
        # img_batch = img_batch.float().to(device)
        # B, L, C, H, W = img_batch.shape
        
        
        true_labels = true_labels.long().to(device)
        if level == 'frame':
            true_labels = true_labels.repeat_interleave(L, dim=0)
        optimizer.zero_grad()
        # pred_labels = model((img_rgb, img_resi))
        pred_labels = model(img_batch, cached_features={
            "clip": clip_representations,
            "llava": llava_representations
        })
        loss = criterion(pred_labels, true_labels)
        total_loss += loss.item()
        log_probs = F.softmax(pred_labels, dim=-1)
            
        res_probs = torch.argmax(log_probs, dim=-1)
        summation = torch.sum(res_probs == true_labels)
        accu = summation / true_labels.shape[0]
        total_accu += accu
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if tot_iter % hparams['display_step'] == 0:
            train_logging(
                        'loss/train_loss_iter', writer, logger, epoch, saver, 
                        tot_iter, total_loss/hparams['display_step'], 
                        total_accu/hparams['display_step'], lr_scheduler
                        )
            # step_train_logging(
            #             'loss/train_loss_iter', writer, logger, epoch, saver, 
            #             tot_iter, total_loss/hparams['display_step'], 
            #             total_accu/hparams['display_step'], lr_scheduler   
            # )
            total_loss = 0
            total_accu = 0
        # lr_scheduler.step()
        tot_iter += 1
        if (tot_iter + 1) % hparams['valid_step'] == 0:
            model.eval()
            with torch.no_grad():
                frame_metrics = Metrics()
                nonoverlapped_video_metrics = Metrics()
                for idx, val_batch in tqdm(enumerate(val_generator, 1), total=len(val_generator), desc='valid'):
                    fns, val_img_batch, val_true_labels = val_batch
                    val_true_labels = val_true_labels.long().to(device)
                    if isinstance(val_img_batch, tuple) or isinstance(val_img_batch, list):
                        B, L, C, H, W = val_img_batch[0].shape
                    else:
                        B, L, C, H, W = val_img_batch.shape
                        
                    llava_representations = []
                    clip_representations = []
                    for fn in fns:
                        label = 'real' if 'real/' in fn else 'fake'
                        base_fn = os.path.basename(fn)
                        prefix = base_fn.rsplit('__', maxsplit=1)[0]
                        index = base_fn.rsplit('__',maxsplit=1)[-1]
                        cached_index = find_nearest_key(cached_representation_index[label], prefix, index)
                        clip_representations.append(cached_representation[label][f'{prefix}_{cached_index}.jpg']['clip'])
                        llava_representations.append(cached_representation[label][f'{prefix}_{cached_index}.jpg']['llava'])
                    clip_representations = torch.stack(clip_representations, dim=0).to(device)
                    llava_representations = torch.stack(llava_representations, dim=0).to(device)
                    
                    for index in range(L - window_size + 1):
                        if isinstance(val_img_batch, tuple) or isinstance(val_img_batch, list):
                            video_clip = val_img_batch[0][:, index: index + window_size, :, :, :], val_img_batch[1][:, index: index + window_size, :, :, :]
                        else:
                            video_clip = val_img_batch[:, index: index + window_size, :, :, :]
                        val_preds = model(video_clip, cached_features={
                            "clip": clip_representations,
                            "llava": llava_representations
                        })
                        frame_val_loss = criterion(val_preds, val_true_labels)
                        frame_log_probs = F.softmax(val_preds, dim=-1)
                        frame_res = torch.argmax(frame_log_probs, dim=-1)
                        frame_samples = frame_res.shape[0]
                        
                        frame_matching_num = (frame_res == val_true_labels).sum().item()
                        frame_metrics.roc.predictions.extend(frame_res.tolist())
                        frame_metrics.roc.pred_proba.extend(frame_log_probs[:,0].tolist())
                        frame_fixed_labels = 1 - val_true_labels
                        frame_metrics.roc.gt.extend(frame_fixed_labels[:].tolist())
                        frame_metrics.update(frame_matching_num, frame_val_loss.item(), frame_samples)
                        
                        if index % window_size == 0 or index == L - window_size:
                            nonoverlapped_video_metrics.roc.predictions.extend(frame_res.tolist())
                            nonoverlapped_video_metrics.roc.pred_proba.extend(frame_log_probs[:,0].tolist())
                            nonoverlapped_video_metrics.roc.gt.extend(frame_fixed_labels[:].tolist())
                            nonoverlapped_video_metrics.update(frame_matching_num, frame_val_loss.item(), frame_samples)
                
            ## Setting the model back to train mode
            model.train()
            # frame_metrics, video_metrics = eval_model(model, val_generator, criterion, window_size=window_size, device=device, desc='valid', debug_mode=False, level=level)
            video_val_loss = nonoverlapped_video_metrics.get_avg_loss()
            lr_scheduler.step(video_val_loss)
            writer.add_scalar('loss/val_loss_iter', video_val_loss, tot_iter)
            logger.info(f'Val loss: {video_val_loss}')
            logger.info(f'Patience: {lr_scheduler.num_bad_epochs} / {patience}')
    if epoch % hparams['save_epoch'] == 0:
        saver.save_model(epoch,tot_iter,total_loss,before_train,force_saving=True)
    