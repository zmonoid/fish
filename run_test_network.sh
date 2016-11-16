#!/usr/bin/env sh
NETWORK=inception-v3
DATASHAPE=337
CROPSHAPE=299
echo $NETWORK

python test_imagenet.py \
    --load-prefix checkpoints/${NETWORK}-$1 \
    --load-epoch 10 \
    --save-prefix data/${NETWORK}_out \
    --batch-size 128 \
    --gpus $2 \
    --shape ${CROPSHAPE} \
    --data-shape ${DATASHAPE}
