# [NeurIPS 2024] On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection

This repository is the official implementation of [MM-Det](https://arxiv.org/abs/2410.23623) [NeurIPS 2024 Poster].

[![arxiv](https://img.shields.io/badge/arXiv-2310.23623-b31b1b.svg)](https://arxiv.org/abs/2410.23623)

- We develop an effective detector, MM-Det, based on multimodal forgery representation. 

- We release Diffusion Video Forensics (DVF) as a diffusion-generated video dataset for forgery detection on diffusion videos. We provide the dataset links of DVF here. [BaiduNetDisk](https://pan.baidu.com/s/1vDCocTRWedmktmzcP903pQ?pwd=KuRo)(Code: KuRo). [google driver](https://drive.google.com/drive/folders/1NxCvJVPSxV2Mib5NaNj5Cf2WnnjrqpMb?usp=drive_link)

<table class="center">
    <tr>
    <td><img src="assets/overview_of_main_method.png" width="600">></td>
    </tr>
</table>


## Contents

- [Environment](#environment)
- [Dataset](#diffusion-video-forensics-dataset)
- [Data Preparation](#preparation)
  - [Reconstruction Process](#reconstruction-dataset)
  - [Caching Multi-Modal Forgery Representation](#caching-multi-modal-forgery-representation)
  - [Pre-trained Weights](#pre-trained-weights)
- [Evaluation](#evaluation)
- [Training](#training)

## Environment

1. Install basic packages
```bash
conda create -n MM_Det python=3.10
conda activate 
pip install -r requirements.txt
cd LLaVA
pip install -e .
```
2. For training cases, install additional packages
```bash
cd LLaVA
pip install --upgrade pip
pip install -e ".[train]"
pip install flash-attn==2.5.8 --no-build-isolation
```

## Diffusion Video Forensics Dataset
We release Diffusion Video Forensics (DVF) as the benchmark for forgery video detection.

<table class="center">
    <tr>
    <td><img src="assets/dvf_dataset_samples.png"></td>
    </tr>
</table>

DVF contains 8 diffusion generative methods, including [Stable Diffusion](https://github.com/comfyanonymous/ComfyUI), [VideoCrafter1](https://github.com/AILab-CVC/VideoCrafter), [Zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w), [Sora](https://openai.com/index/sora/), [Pika](https://pika.art/), [OpenSora](https://github.com/hpcaitech/Open-Sora), [Stable Video](https://stability.ai/stable-video), and [Stable Video Diffusion](https://github.com/Stability-AI/generative-models).

<table class="center">
    <tr>
    <td><img src="assets/dvf_dataset_statistics.png"></td>
    </tr>
</table>

## Preparation

### Reconstruction Dataset

Based on the findings([DIRE](https://github.com/ZhendongWang6/DIRE)) that generative methods always fail to reconstruct details in real videos, we extend this method by utilizing a VQVAE trained on ImageNet to reconstruct each frame. The reconstruction dataset structure is as follows. For all videos in DVF, we provide a ready reconstruction dataset at [BaiduNetDisk](https://pan.baidu.com/s/1oJarzo09jx8Tc1L3GihdSA?pwd=moyu) (Code: moyu).

```
--$RECONSTRUCTION_DATASET_ROOT
  | -- dataset A
    | -- class A1
      | -- original    # frame sequences for original videos
        | -- {video_id_1}_1.jpg
        ...
        | -- {video_id_M}_{frame_id_N}.jpg
      | -- recons    # frame sequences for reconstructed videos
        | -- {video_id_1}_1.jpg
        ...
        | -- {video_id_M}_{frame_id_N}.jpg
    | -- class A2
      | -- original    # frame sequences for original videos
        | -- {video_id_1}_1.jpg
        ...
        | -- {video_id_M}_{frame_id_N}.jpg
      | -- recons    # frame sequences for reconstructed videos
        | -- {video_id_1}_1.jpg
        ...
        | -- {video_id_M}_{frame_id_N}.jpg
  | -- dataset B
      ...
```

For reconstruction datasets of all videos in DVF, we will provide readily available paired dataset sooner for evaluation on MM-Det.

### Caching Multi-Modal Forgery Representation

Our method takes advantage of Multi-Modal Forgery Representation (MMFR) based on finetuned LLaVA-1.5 for forgery detection. Since the representation is fixed during training and inference, it is recommended to cache the representation before the overall training to reduce time cost. For all videos in DVF, we provide a ready dataset for cached MMFR at [BaiduNetDisk](https://pan.baidu.com/s/1ODAfIMRzXlroXG30i5_Bcg?pwd=Haru) (Code: Haru).

### Pre-trained Weights
We provide the [weights](https://huggingface.co/sparklexfantasy/llava-7b-1.5-rfrd) for our fine-tuned large multi-modal model, which is based on llava-v1.5-Vicuna-7b from [LLaVA](https://github.com/haotian-liu/LLaVA). The weights will be automatically downloaded. Besides, the overall weights for MM-Det without the LMM can be achieved from [weights](https://drive.google.com/drive/folders/1RRNS8F7ETZWrcBu8fvB3pM9qHbmSEEzy?usp=sharing) at `MM-Det/current_model.pth`. Please download and put the weights at `./weights/`.

## Evaluation

For datasets in DVF, the reconstructed datasets, as well as the cached MMFR, will be provided. (We will make it available soon.) Set `$RECONSTRUCTION_DATASET_ROOT` as `DVF_recons` and `$MM_REPRESENTATION_ROOT` as `mm_representations`.

For customized datasets, prepare the test dataset frames as well as the cached MMFR by following [Data Preparation](#data-preparation). Set `$RECONSTRUCTION_DATASET_ROOT` and `$MM_REPRESENTATION_ROOT` to both data roots. `--cache-mm` is also recommended for save the computational and memory cost of LMM branch.

Make sure the pretrained weights are organize at `./weights`. Then, run the following script for testing on 7 datasets, respectively. Since the entire evaluation is time-costing, `sample-size` can be specified (e.g., 1,000) to reduce time by conducting inference only on limited (1,000) videos. To finish the entire evaluation, please set `sample-size` as `-1`.

```bash
python test.py \
--classes videocrafter1 zeroscope opensora sora pika stablediffusion stablevideo \
--ckpt ./weights/MM-Det/current_model.pth \
--data-root $RECONSTRUCTION_DATASET_ROOT \
--cache-mm \
--mm-root $MM_REPRESENTATION_ROOT\
# when sample-size > 0, only [sample-size] videos are evaluated for each dataset.
--sample-size -1
```

## Training

Known Issues:

- From the feedback we’ve received, we noticed a deviation in the training process when fine-tuning the large language model, making it difficult to reproduce our reported results fully in some cases. We are now resolving this issue and will share the updated training scripts soon. Currently, we provide the inference interface first.


## Acknowledgement

We express our sincere appreciation to the following projects.

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- [pytorch-vqvae](https://github.com/ritheshkumar95/pytorch-vqvae)
- [Stable Diffusion](https://github.com/comfyanonymous/ComfyUI)
- [VideoCrafter1](https://github.com/AILab-CVC/VideoCrafter)
- [Zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w)
- [OpenSora](https://github.com/hpcaitech/Open-Sora)
- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models).

## Citation

```
@misc{song2024learningmultimodalforgeryrepresentation,
      title={On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection}, 
      author={Xiufeng Song and Xiao Guo and Jiache Zhang and Qirui Li and Lei Bai and Xiaoming Liu and Guangtao Zhai and Xiaohong Liu},
      year={2024},
      eprint={2410.23623},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.23623}, 
}
```
