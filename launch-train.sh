python train.py \
    --data-root ./data/DVF_recons \
    --classes youtube stablevideodiffusion \
    --fix-split \
    --split-path ./splits \
    --cache-mm \
    --mm-root ./data/DVF_mm_representations \
    --expt MM_Det_01 \