python train.py \
    --data-root ./data/dvf_recons \
    --classes youtube stablevideodiffusion \
    --fix-split \
    --split ./splits \
    --cache-mm \
    --mm-root ./data/mm_representations \
    --expt MM_Det_01 \