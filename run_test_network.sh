#!/usr/bin/env sh
NETWORK=inception-v3
DATASHAPE=337
CROPSHAPE=299
echo $NETWORK


task(){
    echo "$1";
    for index in 0 1 2 3; do
        python test_imagenet.py \
            --load-prefix checkpoints/${NETWORK}_$1 \
            --load-epoch 21 \
            --save-prefix data/${NETWORK}-prob-$1-$index \
            --batch-size 128 \
            --gpus $1 \
            --shape ${CROPSHAPE} \
            --data-shape ${DATASHAPE}
    done
}
for index in 0; do
    task "$index" &
done
wait
