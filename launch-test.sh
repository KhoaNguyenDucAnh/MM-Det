CUDA_VISIBLE_DEVICES=3
python test.py \
    --classes stablediffusion sora\
    --ckpt weights/MM-Det/current_model.pth \
    --mm-root ./outputs/mm_representations \
    --cache-mm \
    --sample-size -1